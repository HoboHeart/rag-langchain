import streamlit as st
import os
import json

# --- Importações ---
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Chat Imobiliário", page_icon="🏢")

CAMINHO_DB = "db"

# --- Carregamento com Cache ---
@st.cache_resource
def carregar_sistema():
    print("--- Carregando recursos do sistema (Cache) ---")
    
    # 1. Embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Banco de Dados
    if not os.path.exists(CAMINHO_DB):
        st.error(f"Pasta '{CAMINHO_DB}' não encontrada!")
        return None, None, None, None
        
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embedder)

    # 3. LLMs
    # LLM 1: Reformulador 
    llm_reformulador = Ollama(model="llama3", temperature=0.1)
    
    # LLM 2: Extrator de Filtros (Formato JSON)
    llm_filtro = Ollama(model="llama3", temperature=0, format="json")
    
    # LLM 3: Resposta Final
    llm_resposta = Ollama(model="llama3", temperature=0.7)

    return db, llm_reformulador, llm_filtro, llm_resposta

# --- Função de Memória (Reformulação) ---
def reformular_pergunta(pergunta_atual, historico_mensagens, llm):
    """Reescreve a pergunta baseada no contexto anterior."""
    if len(historico_mensagens) < 2:
        return pergunta_atual

    # Pega as últimas 4 mensagens para contexto
    historico_texto = ""
    for msg in historico_mensagens[-4:]:
        role = "Human" if msg["usuario"] == "user" else "AI"
        historico_texto += f"{role}: {msg['texto']}\n"

    template = """
    Reformule a 'Pergunta Atual' para que ela seja completa e independente, baseada no 'Histórico'.
    Substitua pronomes (ele, dela, lá) pelos nomes reais (cidade, prédio) mencionados antes.
    Retorne APENAS a pergunta reformulada em Português. Nada mais.

    Histórico:
    {historico}

    Pergunta Atual: {pergunta}
    """
    try:
        nova_pergunta = llm.invoke(template.format(historico=historico_texto, pergunta=pergunta_atual))
        return nova_pergunta.strip()
    except:
        return pergunta_atual

# --- Função de Lógica (Extração de Filtros Manual) ---
def extrair_filtros(pergunta, llm):
    """Extrai filtros JSON manualmente com descrições ricas dos campos."""
    
    template = """
    Você é um especialista em banco de dados. Sua tarefa é analisar a pergunta do usuário e transformar em filtros de busca estruturados (JSON).
    Retorne APENAS o JSON. Não explique nada.
    
    SCHEMA DOS DADOS (Campos que você pode usar):
    
    - cidade (string): A cidade onde o prédio está localizado (ex: "Caucaia", "Fortaleza").
    - bairro (string): O nome do bairro (ex: "Centro", "Aldeota", "Jardim").
    - estado (string): A sigla do estado, use sempre maiúsculo (ex: "CE", "SP").
    - cep (string): O código postal. Mantenha a pontuação se houver (ex: "60000-000").
    - endereco (string): Nome da rua ou avenida.
    
    - quantidade_de_salas (int): Número total de salas/escritórios. Use operadores de comparação.
    - tamanho_m2 (int): Área total do imóvel em metros quadrados.
    - preco_estimado (float): Valor de venda do imóvel em Reais.
    - ano_construcao (string): O ano de construção. Note que é uma string no banco.

    SINTAXE DE OPERADORES (MongoDB Style):
    - Igualdade: "campo": "valor"
    - Maior que: "campo": {{ "$gt": 10 }}
    - Menor que: "campo": {{ "$lt": 500000 }}
    - Maior ou igual: "campo": {{ "$gte": 2020 }}
    - Menor ou igual: "campo": {{ "$lte": 100 }}

    EXEMPLOS DE "SHOTS":
    
    User: "Prédios em Itapipoca acima de 10 salas"
    AI: {{ "cidade": "Itapipoca", "quantidade_de_salas": {{ "$gt": 10 }} }}

    User: "Imóvel no bairro Centro que custe menos de 1 milhão"
    AI: {{ "bairro": "Centro", "preco_estimado": {{ "$lt": 1000000 }} }}

    User: "Qual o prédio do CEP 93260-708?"
    AI: {{ "cep": "93260-708" }}

    User: "Prédios construídos depois de 2010"
    AI: {{ "ano_construcao": {{ "$gt": "2010" }} }}

    Pergunta Atual: {pergunta}
    JSON:
    """
    try:
        json_str = llm.invoke(template.format(pergunta=pergunta))
        # Tenta limpar caso o LLM coloque markdown (```json ... ```)
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        filtros = json.loads(json_str)
        return {k: v for k, v in filtros.items() if v}
    except Exception as e:
        print(f"Erro ao extrair filtro: {e}")
        return {}

# --- 4. Interface Streamlit ---
def app():
    st.header("🏢 Imobiliária Inteligente", divider=True)

    db, llm_reformulador, llm_filtro, llm_resposta = carregar_sistema()
    
    if not db:
        st.stop()

    if "mensagens" not in st.session_state:
        st.session_state["mensagens"] = [{"usuario": "assistant", "texto": "Olá! Como posso ajudar você a encontrar o imóvel ideal?"}]

    # Renderiza mensagens
    for mensagem in st.session_state["mensagens"]:
        avatar = "🧑‍💻" if mensagem["usuario"] == "user" else "🤖"
        with st.chat_message(mensagem["usuario"], avatar=avatar):
            st.write(mensagem["texto"])

    # Input do Usuário
    mensagem_usuario = st.chat_input("Digite sua pergunta aqui...")

    if mensagem_usuario:
        st.session_state["mensagens"].append({"usuario": "user", "texto": mensagem_usuario})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(mensagem_usuario)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            
            # Status visual
            status = st.status("Processando...", expanded=True)
            
            try:
                # PASSO 1: Reformulação (Memória)
                status.write("🧠 Lendo histórico...")
                historico_atual = st.session_state["mensagens"][:-1]
                pergunta_final = reformular_pergunta(mensagem_usuario, historico_atual, llm_reformulador)
                status.write(f"**Entendido:** {pergunta_final}")

                # PASSO 2: Filtros (Lógica)
                status.write("🔍 Identificando critérios...")
                filtros = extrair_filtros(pergunta_final, llm_filtro)
                if filtros:
                    status.write(f"**Filtros:** {filtros}")
                else:
                    status.write("**Busca Semântica:** (Sem filtros exatos)")

                # PASSO 3: Busca (Chroma)
                status.write("📂 Consultando banco de dados...")
                docs = db.similarity_search(
                    pergunta_final, 
                    k=4, 
                    filter=filtros if filtros else None
                )
                
                if not docs:
                    texto_resposta = "Não encontrei imóveis correspondentes a esses critérios específicos."
                    status.update(label="Sem resultados", state="error")
                else:
                    # PASSO 4: Resposta (Geração)
                    status.write("✍️ Escrevendo resposta...")
                    contexto = "\n\n".join([d.page_content for d in docs])
                    
                    template_resp = """
                    Você é um corretor imobiliário. Responda à pergunta do usuário usando os imóveis abaixo.
                    
                    Imóveis Encontrados:
                    {context}

                    Histórico da Conversa:
                    {pergunta_original}
                    
                    Pergunta Final: {pergunta_reformulada}
                    
                    Responda de forma natural:
                    """
                    
                    chain_input = {
                        "context": contexto, 
                        "pergunta_original": mensagem_usuario,
                        "pergunta_reformulada": pergunta_final
                    }
                    
                    texto_resposta = llm_resposta.invoke(template_resp.format(**chain_input))
                    status.update(label="Concluído!", state="complete", expanded=False)

            except Exception as e:
                texto_resposta = f"Ocorreu um erro técnico: {str(e)}"
                status.update(label="Erro", state="error")

            placeholder.write(texto_resposta)
            st.session_state["mensagens"].append({"usuario": "assistant", "texto": texto_resposta})

if __name__ == "__main__":
    app()