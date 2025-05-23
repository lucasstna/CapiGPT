from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.docstore.document import Document

def create_reranker(vectorstore: FAISS) -> ContextualCompressionRetriever:
    """
    Creates a reranker retriever using FlashrankRerank.
    
    Args:
        retriever: Base retriever to compress/rerank results
        
    Returns:
        FAISS vectorstore object
    """

    retriever = vectorstore.as_retriever()

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )
    return compression_retriever

def rerank_search(
    compression_retriever: ContextualCompressionRetriever,
    query: str
) -> List[Document]:
    """
    Performs reranked search using the compression retriever.
    
    Args:
        compression_retriever: The reranking retriever to use
        query: Search query string
        
    Returns:
        List of retrieved Documents
    """
    return compression_retriever.invoke(query)