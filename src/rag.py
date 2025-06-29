from langchain_core.prompts.chat import ChatPromptTemplate

from langchain.chains.question_answering import load_qa_chain

from reranker import rerank_search


class RAG:
    
    @staticmethod
    def assistant(content: str):
        return ("assistant", content)

    @staticmethod
    def user(content: str):
        return ("user", content)
    
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model
        
        prompt_structure = '''
            Baseado nos seguintes documentos:
            {context}
            Responda a pergunta abaixo:
            {query}
            '''

        self.prompt = ChatPromptTemplate.from_messages([
            RAG.user('''Você é um funcionário da Universidade Federal de Mato Grosso do Sul que tem conhecimento
            de todo o documento apresentado como contexto e 
            responde todas as perguntas em Portugues do Brasil. Você responde a perguntas sobre o documento apresentado, usando o contexto fornecido.
            Você sempre cita INTEGRALMENTE o item do edital que contém a resposta desejada. Se não souber a resposta, responda "Não consigo encontrar essa informação no documento". 
            Você cita seomente item necessário para resposta direta da pergunta e nada mais. Você SEMPRE cita o número do item que contém a resposta. Aqui estão alguns exemplos:'''),
            RAG.user('''Como será a lista de espera?'''),
            RAG.assistant('''De acordo com o item 3.4 do edital, a lista de espera será definida pela ordem de cadastro aprovado 
            e permanecerá para o atendimento por meio da liberação de novas vagas pelo MEC.'''),
            RAG.user('''Qual o número mínimo de membros das comissões temporárias constituídas pelo Conselho?'''),
            RAG.assistant('''De acordo com o Art. 61, as comissões temporárias deverão ser constituídas por, no mínimo, três membros.'''),
            RAG.user(prompt_structure)
        ])

        self.chain = load_qa_chain(model, chain_type='stuff', verbose=True, prompt=self.prompt)
    

    def generate_response(self, query):

        retrieval_result = rerank_search(self.retriever, query)

        response = self.chain.invoke(
            {'input_documents':retrieval_result, 'query':query}
        )
    
        return response['output_text']

