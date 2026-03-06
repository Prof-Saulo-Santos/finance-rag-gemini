import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Procura o arquivo .env na pasta atual e nas pastas pai (como a raiz do projeto)
load_dotenv(find_dotenv())

# Configuração da página do Streamlit
st.set_page_config(page_title="RAG Financeiro", page_icon="📈", layout="centered")
st.title("Sistema RAG: Relatório Financeiro 📈")

# Puxa a chave explicitamente para podermos verificar se ela carregou
gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("⚠️ Chave de API não encontrada! Verifique se seu arquivo `.env` está na raiz do projeto contendo `GOOGLE_API_KEY=sua_chave`.")
    st.stop() # Interrompe a execução do Streamlit aqui mesmo para evitar erros feios no console

# Definir o caminho correto para o banco de dados vetorial
FAISS_INDEX_PATH = "faiss_index"

# ==========================================
# 1. Funções de Configuração e Cache
# ==========================================
# Usamos @st.cache_resource para não recarregar os modelos a cada interação na tela
@st.cache_resource
def get_vectorstore():
    # Inicializa os Embeddings limitados à CPU para compatibilidade de hardware local
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Carrega o banco FAISS previamente salvo
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Necessário para carregar FAISS local seguro
        )
        return vectorstore
    else:
        st.error("Banco de dados vetorial não encontrado! Execute 'uv run python src/ingest.py' primeiro.")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_rag_chain(api_key):
    vectorstore = get_vectorstore()
    if not vectorstore:
        return None

    # Configura o recuperador (retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # Conecta com a LLM do Gemini da Google passando a chave explicitamente
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=api_key
    )

    # Criação do Prompt
    template = """Você é um assistente financeiro especializado em análise de relatórios.
    Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto fornecido do relatório financeiro.
    Se o contexto não contiver a resposta para a pergunta, diga claramente "O relatório não contém essa informação."
    Não invente informações (não produza alucinações).

    Contexto do relatório:
    {context}

    Pergunta do usuário: {input}

    Resposta detalhada (em português):"""

    prompt = PromptTemplate.from_template(template)

    # Cadeia RAG moderna usando LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Inicializar o RAG
chain = get_rag_chain(gemini_api_key)

# ==========================================
# 2. Interface de Chat (Streamlit)
# ==========================================
# Variável na sessão (session_state) para guardar o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens já enviadas na tela (do mais antigo ao mais novo)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Caixa de input para o usuário digitar a pergunta
if user_input := st.chat_input("Pergunte algo sobre o relatório (ex: Qual a receita total?)"):
    # 1. Salva e exibe a pergunta do usuário na tela
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Se a pipeline RAG carregou, tenta gerar a resposta
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("Analisando o relatório financeiro..."):
                try:
                    # Executa a busca e gera a resposta via LCEL
                    answer = chain.invoke(user_input)

                    st.markdown(answer)

                    # Salva a resposta no histórico da tela para persistência
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Erro ao processar com a API do Gemini: {e}")
                    st.info("Verifique se seu GOOGLE_API_KEY no arquivo .env na raiz do projeto está correto.")
