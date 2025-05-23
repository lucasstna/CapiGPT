from langchain_community.vectorstores import FAISS
from typing import List
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker


def create_vectorstore_from_dir(
    doc_folder: str,
    embeddings_model: Embeddings,
    documents_dir_path: str = '../database/{doc_folder}',
    persist: bool = True
) -> FAISS:
    """
    Creates a FAISS vectorstore from PDF documents in a directory and saves it to disk.
    
    Args:
        doc_folder: Name of the folder containing PDF documents
        embeddings_model: The embeddings model to use
        documents_dir_path: Path to the directory containing the documents
        persist: Whether to save the vectorstore to disk or not 
        
    Returns:
        FAISS vectorstore object
    """
    documents_dir_path = documents_dir_path.format(doc_folder=doc_folder)

    print(f'Loading documents from {documents_dir_path}')
    
    # Load PDF documents from directory
    loader = DirectoryLoader(
        documents_dir_path, 
        glob='./*.pdf', 
        loader_cls=PyPDFLoader
    )
    loaded_pdfs = loader.load()
    
    # Split documents using semantic chunker
    text_splitter = SemanticChunker(
        embeddings=embeddings_model
    )
    pages = text_splitter.split_documents(loaded_pdfs)
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(
        pages,
        embeddings_model
    )

    # Save vectorstore
    if persist:
        print(f'Saving vectorstore to {documents_dir_path}/document_index')

        vectorstore.save_local(f'{documents_dir_path}/document_index')
    
    return vectorstore

def load_vectorstore(
    doc_folder: str,
    embeddings_model: Embeddings
) -> FAISS:
    """
    Loads a FAISS vectorstore from disk.
    
    Args:
        doc_folder: Folder name where the vectorstore is saved
        embeddings_model: The embeddings model to use
        
    Returns:
        FAISS vectorstore object
    """

    doc_folder = f'{doc_folder}/document_index'

    return FAISS.load_local(
        doc_folder  ,
        embeddings_model,
        allow_dangerous_deserialization=True
    )