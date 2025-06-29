from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models.generate import QueryRequest

from rag import RAG
from vectorstore import load_vectorstore, create_vectorstore_from_dir
from reranker import create_reranker

# Create FastAPI instance
app = FastAPI(title="CapiGPT API", description="Uma aplicação para consulta de informações em documentos da UFMS", version="1.0.0")

ollama_base_url = 'http://localhost:11435'

model = Ollama(base_url=ollama_base_url, model="llama3.1:latest", temperature=0)
embeddings_model = HuggingFaceEmbeddings(model_name='stjiris/bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-MetaKD-v0')

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    """
    Root endpoint that returns the HTML landing page
    """
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/info")
async def get_info():
    """
    Info endpoint that returns application information
    """
    return {
        "app_name": "CapiGPT",
        "version": "1.0.0",
        "description": "A FastAPI Hello World application"
    }

@app.post("/generate")
async def generate_response(request: QueryRequest):
    """
    Generate a response based on the provided parameters
    """
    try:
        print("entrou")
        vectorstore = create_vectorstore_from_dir('edital1', embeddings_model, documents_dir_path='database/edital1', persist=False)

        print("carregou vectorstore")
        # Create retriever from vectorstore
        retriever = create_reranker(vectorstore)
        # Initialize RAG with retriever and model
        print(f"Received request: {request.query}")
        rag = RAG(retriever=retriever, model=model)

        print(f"Received request: {request.query}")

        return {"response": rag.generate_response(request.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)