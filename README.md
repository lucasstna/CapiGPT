# CapiGPT

Um sistema de perguntas e respostas baseado em LLMs para auxiliar com dúvidas sobre documentos da UFMS. Utiliza RAG (Retrieval Augmented Generation) para fornecer respostas precisas com referências adequadas aos documentos.

## Descrição

A aplicação web permite aos usuários:
- Fazer upload de documentos PDF da UFMS
- Fazer perguntas sobre estes documentos
- Obter respostas com referências às seções específicas

## Estrutura do Projeto

```
CapiGPT/
├── src/                       # Código fonte
│   ├── app.py                 # Aplicação FastAPI
│   ├── rag.py                 # Implementação RAG
│   ├── reranker.py            # Lógica de reranking
│   ├── vectorstore.py         # Operações da vector store
│   └── models/                # Modelos Pydantic
...
├── database/                  # Armazenamento dos documentos disponíveis para pergunta
    ├── edital1
│   |   └── document_index/    # Índices do vector store
|   ... 
├── datasets/                  # Conjuntos de dados de teste
├── tests/                     # Resultados dos testes conduzidos
└── notebooks/                 # Notebooks Jupyter usados no desenvolvimeto da solução final
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/lucasstna/CapiGPT.git
cd CapiGPT
```

2. Instale as dependências:
```bash
pip install -r requirements-semantic-rag.txt
```

3. Instale e inicie o Ollama:
   - Acesse [ollama.com/download/linux](https://ollama.com/download/linux) para instruções de instalação

4. Baixe o modelo necessário:
```bash
ollama pull llama3.1:8b
```

## Uso

1. Inicie a aplicação:
```bash
python src/app.py
```

2. Abra seu navegador e acesse:
```
http://localhost:8000
```