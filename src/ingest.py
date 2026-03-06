import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Caminho para o arquivo PDF (definido no início do projeto)
PDF_PATH = "data/rag-financeiro.pdf"
# Pasta onde o banco de dados vetorial será salvo
FAISS_INDEX_PATH = "faiss_index"

def main():
    print(f"Iniciando a ingestão do documento: {PDF_PATH}")

    # 1. Carregar o documento PDF
    if not os.path.exists(PDF_PATH):
        print(f"ERRO: Arquivo não encontrado em {PDF_PATH}")
        return

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Documento carregado. Páginas totais: {len(documents)}")

    # 2. Dividir o texto em pedaços menores (chunks)
    # Isso é essencial para que a LLM consiga processar o contexto sem exceder seu limite de tokens
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, # Aumentado para pegar tabelas inteiras ou grandes seções de uma vez
        chunk_overlap=400, # Aumentado para garantir que informações na transição não se percam
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documento dividido em {len(chunks)} pedaços maiores (chunks de 2500 caracteres).")

    # 3. Criar os Embeddings
    # Usando o modelo sentence-transformers 'all-MiniLM-L6-v2' (grátis e roda localmente)
    # E ativando explicitamente o uso da CPU para evitar erros com GPUs antigas ou incompatíveis.
    print("Inicializando o modelo de Embeddings (HuggingFace) em CPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Criar e salvar o banco de dados vetorial (FAISS)
    print("Criando o banco de dados vetorial FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Salvando o banco de dados em '{FAISS_INDEX_PATH}'...")
    vectorstore.save_local(FAISS_INDEX_PATH)

    print("Ingestão concluída com sucesso! Banco de dados pronto para uso.")

if __name__ == "__main__":
    main()
