from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

from langchain_core.prompts import PromptTemplate

import streamlit as st
import os

#--- Configuração da página do Streamlit ---
st.set_page_config(
    page_title="Chat Imobiliário",
    page_icon="🏢")

CAMINHO_DB = "db"

#--- Carregamento utilizando a cache, para evitar recarregar o DB a cada clique ---
@st.cache_resource

def carregar_sistema():
    print("--- Carregando os recursos do Sistema... ---")

    # 1. Modelo de Embedding - Utilizando o mesmo da criação
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    funcao_embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 2. Banco de Dados
    if not os.path.exists(CAMINHO_DB):
        st.error(f"Pasta '{CAMINHO_DB}' não encontrada! Execute 'criar_db.py' primeiro.")
        return None, None
    
    db = Chroma(persist_directory = CAMINHO_DB, embedding_function=funcao_embedding)
    print("Banco de Dados Carregado.")

    # 3. Configuração do Self-Querying
    # Definindo para o LLM quais campos de metadados existem e o que eles significam:
    metadata_field_info = [
    # --- Dados do CSV (Localização) ---
        AttributeInfo(
            name="bairro",
            description="O nome do bairro onde o imóvel está localizado (ex: Centro, Aldeota)",
            type="string",
        ),
        AttributeInfo(
            name="cidade",
            description="A cidade onde o prédio está localizado (ex: Caucaia, Fortaleza)",
            type="string",
        ),
        AttributeInfo(
            name="estado",
            description="A sigla do estado (ex: CE, SP)",
            type="string",
        ),
        AttributeInfo(
            name="cep", 
            description="O CEP do imóvel com pontuação (ex: 60000-000)", 
            type="string"
        ),
        #Endereço (rua). Útil se o usuário busca pelo nome da rua.
        AttributeInfo(
            name="endereco",
            description="O nome da rua, avenida ou endereço do imóvel",
            type="string",
        ),

        # --- Dados do JSON (Técnicos) ---
        AttributeInfo(
            name="quantidade_de_salas",
            description="O número de salas ou escritórios no prédio",
            type="integer",
        ),
        AttributeInfo(
            name="tamanho_m2",
            description="O tamanho total do prédio em metros quadrados",
            type="integer",
        ),
        AttributeInfo(
            name="preco_estimado",
            description="O valor ou preço de venda do imóvel",
            type="float",
        ),
        AttributeInfo(
            name="ano_construcao",
            description="O ano em que o prédio foi construído",
            type="string", 
        ),
    ]

    document_content_description = "Ficha técnica de prédios comerciais"

    # 4. LLM para o Filtro (Llama 3)
    llm_filter = Ollama(model="llama3", temperature=0) # Temp 0 para precisão lógica

    # 5. Criação do Retriever
    # Instanciando o retriever aqui dentro para ele ficar salvo no cache
    retriever = SelfQueryRetriever.from_llm(
        llm_filter,
        db,
        document_content_description,
        metadata_field_info,
        verbose=True # True para ver o filtro sendo criado no console
    )

    # 6. LLM para a resposta final
    llm_resposta = Ollama(model="llama3", temperature=0.7)

    return retriever, llm_resposta

# --- Função da Interface do Streamlit ---
def app():
    st.header("🏢 Imobiliária Inteligente", divider=True)
    st.caption("Pergunte sobre localização, preço, tamanho ou características específicas.")

    # Inicializa o sistema (Recupera do cache se já estiver carregado)
    retriever, llm_resposta = carregar_sistema()
    
    if not retriever:
        st.stop()

    # Inicializa histórico de mensagens na sessão
    if "mensagens" not in st.session_state:
        st.session_state["mensagens"] = [{"usuario": "assistant", "texto": "Olá! Como posso ajudar você a encontrar o imóvel ideal?"}]

    # Exibindo as mensagens anteriores
    for mensagem in st.session_state["mensagens"]:
        
        avatar = "🧑‍💻" if mensagem["usuario"] == "user" else "🤖"
        with st.chat_message(mensagem["usuario"], avatar=avatar):
            st.write(mensagem["texto"])

    # Campo de entrada do usuário
    mensagem_usuario = st.chat_input("Digite sua pergunta aqui...")

    if mensagem_usuario:
        # 1. Mostra a mensagem do usuário na tela e salva no histórico
        st.session_state["mensagens"].append({"usuario": "user", "texto": mensagem_usuario})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(mensagem_usuario)

        # 2. Processa a resposta do sistema
        with st.chat_message("assistant", avatar="🤖"):
            placeholder_resposta = st.empty() # Placeholder para atualizar depois
            
            with st.spinner("Analisando filtros e buscando imóveis..."):
                try:
                    # --- Lógica do RAG (SelfQuery) ---
                    docs = retriever.invoke(mensagem_usuario)
                    
                    if not docs:
                        texto_resposta = "Não encontrei nenhum imóvel com esses filtros exatos na nossa base de dados."
                    else:
                        # Prepara o contexto
                        contexto = "\n\n".join([d.page_content for d in docs])
                        
                        # Template de Resposta
                        template_resposta = """
                        Você é um assistente imobiliário. Use as informações abaixo para responder.
                        
                        Imóveis Encontrados:
                        {context}

                        Pergunta do Usuário: {question}
                        """
                        prompt_final = PromptTemplate.from_template(template_resposta)
                        
                        # Gera a resposta final
                        chain_input = {"context": contexto, "question": mensagem_usuario}
                        texto_resposta = llm_resposta.invoke(prompt_final.format(**chain_input))

                except Exception as e:
                    texto_resposta = f"Ocorreu um erro ao processar sua solicitação: {str(e)}"
            
            # 3. Mostra a resposta e salva no histórico
            placeholder_resposta.write(texto_resposta)
            st.session_state["mensagens"].append({"usuario": "assistant", "texto": texto_resposta})

# Executa o app
if __name__ == "__main__":
    app()