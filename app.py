from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts.chat import ChatPromptTemplate

ollama = Ollama(base_url='http://localhost:11434', model="llama3")

doc_path = '/home/lucass/rag/resultados/EDITAL-PROAES-N-70-DE-JUNHO-DE-2024_CADASTRO-PARA-BOLSA-PERMANENCIA-DO-MEC-PARA-ESTUDANTES-INDIGENAS-E-QUILOMBOLA-MA-GRADUACAO-PROGRAMA-BPMEC-2024.pdf'

pdf_loader = PyPDFLoader(doc_path)

loaded_pdf = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=['\n', ' ', '']
)

pages = text_splitter.split_documents(loaded_pdf)

retriever = BM25Retriever.from_documents(pages)

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

chain = load_qa_chain(ollama, chain_type='stuff', verbose=True, prompt=qa_prompt)

query = input()

docs = retriever.invoke(query)

output = chain.invoke(
        {'input_documents':docs, 'query':query}
)

print(output['output_text'])