import json

from langchain.chains.question_answering import load_qa_chain

from langchain_community.llms import Ollama
# from langchain_community.embeddings.laser import LaserEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatMaritalk

from langchain_core.prompts.chat import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_experimental.text_splitter import SemanticChunker

import numpy as np

import pandas as pd

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

import os

from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

def reranker_retrieval(compression_retriever, query):
    return compression_retriever.invoke(query)


def assistant(content: str):
    return ("assistant", content)

def user(content: str):
    return ("user", content)

def main():
    # ollama_base_url= 'http://tempestade.facom.ufms.br:11435'
    ollama_base_url = 'http://localhost:11435'
    # model = Ollama(base_url=ollama_base_url, model="phi3:medium", temperature=0)

    # ollama_base_url = 'http://localhost:11434' 
    model = Ollama(base_url=ollama_base_url, model="llama3.3:latest", temperature=0)
    # model = Ollama(base_url=ollama_base_url, model="phi3:medium", temperature=0)

    # embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    embeddings_model = HuggingFaceEmbeddings(model_name='stjiris/bert-large-portuguese-cased-legal-tsdae-gpl-nli-sts-MetaKD-v0')

    prompt_structure = '''
        Baseado nos seguintes documentos:
        {context}
        Responda a pergunta abaixo:
        {query}
        '''

    # few-shot prompting
    qa_prompt = ChatPromptTemplate.from_messages([
            user('''Você é um funcionário da Universidade Federal de Mato Grosso do Sul que tem conhecimento
            de todo o documento apresentado como contexto e 
            responde todas as perguntas em Portugues do Brasil. Você responde a perguntas sobre o documento apresentado, usando o contexto fornecido.
            Você sempre cita INTEGRALMENTE o item do edital que contém a resposta desejada. Se não souber a resposta, responda "Não consigo encontrar essa informação no documento". 
            Você cita seomente item necessário para resposta direta da pergunta e nada mais. Você SEMPRE cita o número do item que contém a resposta. Aqui estão alguns exemplos:'''),
            user('''Como será a lista de espera?'''),
            assistant('''De acordo com o item 3.4 do edital, a lista de espera será definida pela ordem de cadastro aprovado 
            e permanecerá para o atendimento por meio da liberação de novas vagas pelo MEC.'''),
            user('''Qual o número mínimo de membros das comissões temporárias constituídas pelo Conselho?'''),
            assistant('''De acordo com o Art. 61, as comissões temporárias deverão ser constituídas por, no mínimo, três membros.'''),
            user(prompt_structure)
    ])

    chain = load_qa_chain(model, chain_type='stuff', verbose=True, prompt=qa_prompt)

    editais = ['edital1', 'edital2', 'regulamento']

    mapeamento = {
        'edital1': 'edital-70',
        'edital2': 'edital-9',
        'regulamento': 'regulamento'
    }

    queries = pd.read_csv(f'../datasets/dataset-ragas-openai.csv')

    editais = ['edital1', 'edital2', 'regulamento']

    os.environ["OPENAI_API_KEY"] = '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'

    for edital in editais:
 
        answers = []
        contexts = []

        vectorstore = FAISS.load_local(f'../database/{edital}/document_index', embeddings_model, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever()

        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        questions = queries[queries['name'] == edital]

        print(f'Processing {edital} with {len(questions)} queries')

        for question in questions.user_input.tolist():

            _ = []
            # query = f'De acordo com o edital {num_doc}. ' + question

            # vectorstore = FAISS.load_local(f'../database/{map_editais[num_doc]}/document_index', embeddings_model, allow_dangerous_deserialization=True)

            retrieval_result = reranker_retrieval(compression_retriever, question)

            output = chain.invoke(
                {'input_documents':retrieval_result, 'query':question}
            )

            for page in retrieval_result:
                _.append(page.page_content)
                
            contexts.append(_)
            answers.append(output['output_text'])

        result = pd.DataFrame(
            {
                'question':questions.user_input.tolist(),
                'answer':answers,
                'contexts':contexts,
                'reference':questions.reference_contexts.tolist(),
                'ground_truths':questions.reference.tolist()
            }           
        )

        result.to_csv(f'../tests/llama/{model.model}-few-shot-reranker-bertimbau-semantic-split-{mapeamento[edital]}.csv', index=False)

        dataset = Dataset.from_pandas(result[['question', 'answer', 'contexts', 'reference', 'ground_truths']])

        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        for m in metrics:
            # change LLM for metric
            m.__setattr__("llm", llm)

            # check if this metric needs embeddings
            if hasattr(m, "embeddings"):
                # if so change with VertexAI Embeddings
                m.__setattr__("embeddings", embeddings)

        result = evaluate(
            dataset = dataset, 
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ]
        )

        df = result.to_pandas()
        df.to_csv(f'../tests/llama/{model.model}-few-shot-reranker-bertimbau-semantic-split-{mapeamento[edital]}-metrics.csv', index=False)

    


if __name__ == "__main__":
    main()
