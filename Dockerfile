# Usa a imagem oficial leve do Python 3.13
FROM python:3.13-slim

# Instala ferramentas do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o binário do 'uv' para acelerar a instalação
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de gerenciamento de dependências
COPY pyproject.toml uv.lock ./

# Instala os pacotes
RUN uv sync --frozen

# Copia o restante do projeto
COPY . /app

# Expõe a porta obrigatória do Hugging Face Spaces
EXPOSE 7860

# Comando de inicialização
# Gera o banco local FAISS e logo depois sobe o Streamlit
CMD ["bash", "-c", "uv run python src/ingest.py && uv run streamlit run src/app.py --server.port=7860 --server.address=0.0.0.0"]
