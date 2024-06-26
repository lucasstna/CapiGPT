from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts.chat import ChatPromptTemplate

ollama = Ollama(base_url='http://localhost:11434', model="llama3", template="")

doc_path = '/home/lucass/rag/resultados/EDITAL-PROAES-N-70-DE-JUNHO-DE-2024_CADASTRO-PARA-BOLSA-PERMANENCIA-DO-MEC-PARA-ESTUDANTES-INDIGENAS-E-QUILOMBOLA-MA-GRADUACAO-PROGRAMA-BPMEC-2024.pdf'

# loader = TextLoader(doc_path)
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=30,
#     length_function=len
# )
# all_splits = text_splitter.split_documents(data)

pdf_loader = PyPDFLoader(doc_path)

loaded_pdf = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=['\n', ' ', '']
)

pages = text_splitter.split_documents(loaded_pdf)

retriever = BM25Retriever.from_documents(pages)

# oembed = OllamaEmbeddings(base_url="http://localhost:11434")
# vectorstore = Chroma.from_documents(documents=pages, embedding=oembed)

# question='''
#         <s>[INST] <<SYS>>
#         Você é um funcionário da Universidade Federal de Mato Grosso do Sul que tem conhecimento de todo o documento apresentado como contexto e  
#         responde todas as perguntas em Portugues do Brasil.
#         <</SYS>>

#         Qual o número mínimo de membros das comissões temporárias constituídas pelo Conselho? [/INST]
#         De acordo com o Art. 61, as comissões temporárias deverão ser constituídas por, no mínimo, três membros.

#         [INST]
#         A quem deverão ser endereçados os pareceres das comissões? [/INST]
#         De acordo com o Art. 64, os pareceres das comissões temporárias deverão ser endereçados ao Presidente do Conselho
#         e enviados para a Unidade de assessoramento e aos Órgãos Colegiados Superiores da UFMS.
#         </s>

#         <s> [INST]
#         Por quem será composto o Conselho Universitário?
#         '''
# docs = vectorstore.similarity_search(question, k=10)
prompt_structure = '''
        Baseado nos seguintes documentos:
        {context}
        Responda a pergunta abaixo:
        {query}
        '''


qa_prompt = ChatPromptTemplate.from_messages([
        ('system', '''Você é um funcionário da Universidade Federal de Mato Grosso do Sul que tem conhecimento
         de todo o documento apresentado como contexto e 
         responde todas as perguntas em Portugues do Brasil. Você responde a perguntas sobre o documento apresentado, usando o contexto fornecido.
         Use a seguinte estrutura para responder as perguntas:'''),
         ('human', '''Como será a lista de espera?'''),
         ('ai', '''De acordo com o item 3.4 do edital, a lista de espera será definida pela ordem de cadastro aprovado 
          e permanecerá para o atendimento por meio da liberação de novas vagas pelo MEC.'''),
        # ('human', '''Qual o número mínimo de membros das comissões temporárias constituídas pelo Conselho?'''),
        # ('ai', '''De acordo com o Art. 61, as comissões temporárias deverão ser constituídas por, no mínimo, três membros.'''),
        ('human', prompt_structure)
])
# qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
chain = load_qa_chain(ollama, chain_type='stuff', verbose=True, prompt=qa_prompt)

query = input()

docs = retriever.invoke(query)

output = chain.invoke(
        {'input_documents':docs, 'query':query}
)

print(output['output_text'])

# model_output = qachain.invoke({"query": question})

# print("USUÁRIO:\n", model_output['query'], '\nSISTEMA:\n', model_output['result'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# import json
# import requests

# # NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
# model = 'llama3' # TODO: update this for whatever model you wish to use

# def generate(prompt, context):
#     r = requests.post('http://localhost:11434/api/generate',
#                       json={
#                           'model': model,
#                           'prompt': prompt,
#                           'context': context,
#                       },
#                       stream=True)
#     r.raise_for_status()

#     for line in r.iter_lines():
#         body = json.loads(line)
#         response_part = body.get('response', '')
#         # the response streams one token at a time, print that as we receive it
#         print(response_part, end='', flush=True)

#         if 'error' in body:
#             raise Exception(body['error'])

#         if body.get('done', False):
#             return body['context']

# def main():
#     context = [] # the context stores a conversation history, you can use this to make the model more context aware
#     while True:
#         user_input = input("Enter a prompt: ")
#         if not user_input:
#             exit()
#         print()
#         context = generate(user_input, context)
#         print()

# if __name__ == "__main__":
#     main()