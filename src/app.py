from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models.query import QueryRequest
from fastapi import UploadFile

import os
from typing import List

from rag import RAG
from vectorstore import load_vectorstore, create_vectorstore_from_dir
from reranker import create_reranker

# Create FastAPI instance
app = FastAPI(title='CapiGPT API', description='Uma aplicação para consulta de informações em documentos da UFMS', version='1.0.0')

ollama_base_url = 'http://localhost:11435'

model = Ollama(base_url=ollama_base_url, model='llama3.1:latest', temperature=0)
embeddings_model = HuggingFaceEmbeddings(model_name='stjiris/bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-MetaKD-v0')

# Mount static files directory
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
async def read_root():
    '''
    Root endpoint that returns the HTML landing page
    '''
    return FileResponse('static/index.html')

@app.post('/generate')
async def generate_response(request: QueryRequest):
    '''
    Generate a response based on the provided parameters
    '''
    try:

        vectorstore = load_vectorstore(f'database/{request.file}', embeddings_model)

        # Create retriever from vectorstore
        retriever = create_reranker(vectorstore)

        # Initialize RAG with retriever and model
        rag = RAG(retriever=retriever, model=model)

        return {'response': rag.generate_response(request.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/upload')
async def upload_file(file: UploadFile):
    '''
    Endpoint to handle file uploads
    '''
    try:
        # Read file content
        contents = await file.read()

        dir = f'database/{file.filename.split('.')[0]}'
        os.mkdir(dir)

        # Save the uploaded file
        with open(f'{dir}/{file.filename}', 'wb') as f:
            f.write(contents)
        
        create_vectorstore_from_dir(file.filename, embeddings_model, documents_dir_path=dir, persist=True)

        return {'filename': file.filename, 'message': 'Arquivo enviado com sucesso.'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/list-documents', response_model=List[str])
async def list_documents():
    '''
    List all available documents in the database directory
    '''
    try:
        documents = []
        for dir_name in os.listdir('database'):
            if os.path.isdir(os.path.join('database', dir_name)):
                documents.append(dir_name)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)