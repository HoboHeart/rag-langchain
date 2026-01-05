import streamlit as st
import os
import json

# --- Importações ---
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Chat Imobiliário (Smart Router)", page_icon="🏢")

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
        return None, None, None
        
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embedder)

    # 3. LLMs
    # LLM 1: Reformulador (O Cérebro da Memória)
    llm_reformulador = Ollama(model="llama3", temperature=0)

    # LLM 2: Extrator de Filtros (JSON)
    llm_filtro = Ollama(model="llama3", temperature=0, format="json")
    
    # LLM 3: Resposta Final
    llm_resposta = Ollama(model="llama3", temperature=0.7)

    return db, llm_reformulador, llm_filtro, llm_resposta

def reformular_pergunta(pergunta_atual, historico_mensagens, llm):
    """
    Reescreve a pergunta.
    CORREÇÃO: Agora detecta mudança de assunto para não misturar filtros antigos.
    """
    if len(historico_mensagens) < 2:
        return pergunta_atual

    historico_texto = ""
    for msg in historico_mensagens[-4:]: 
        role = "Human" if msg["usuario"] == "user" else "AI"
        historico_texto += f"{role}: {msg['texto']}\n"

    template = """
    Aja como um interpretador de intenção. 
    Reescreva a 'Pergunta Atual' para torná-la independente, usando o histórico APENAS SE NECESSÁRIO.

    REGRAS OBRIGATÓRIAS:
    1. RESOLUÇÃO DE PRONOMES: Se o usuário usar "ele", "dela", "o mais barato", "o primeiro", substitua pelo sujeito do histórico.
    
    2. MUDANÇA DE TÓPICO (CRÍTICO): 
       - Se a 'Pergunta Atual' menciona uma NOVA LOCALIZAÇÃO (Cidade/Bairro) diferente do histórico, IGNORE a localização antiga.
       - Exemplo: Histórico="Em Sobral" -> Pergunta="E em Fortaleza?" -> Saída="Quais imóveis em Fortaleza?" (NÃO misture as cidades).
       
    3. MUDANÇA DE CARACTERÍSTICA:
       - Se o usuário mudar o filtro (ex: "Agora com 3 quartos"), mantenha a cidade mas atualize os quartos.

    Histórico:
    {historico}

    Pergunta Atual: {pergunta}
    
    Pergunta Reescrita (em Português):
    """
    
    try:
        # Temperatura 0 ajuda, mas o prompt explícito é o que resolve.
        return llm.invoke(template.format(historico=historico_texto, pergunta=pergunta_atual)).strip()
    except:
        return pergunta_atual

def classificar_intencao(pergunta_reformulada, llm):
    """
    Decide se o usuário quer buscar novos dados (NOVA_BUSCA) 
    ou analisar o que já foi mostrado (ANALISE_CONTEXTO).
    """
    template = """
    Analise a pergunta do usuário: "{pergunta}"
    
    Classifique a intenção em APENAS UMA das opções abaixo:
    
    1. "NOVA_BUSCA": Se o usuário quer filtrar, mudar de cidade, ver outros imóveis ou buscar algo novo. 
       (Ex: "Mostre em Sobral", "Quero com 3 quartos", "Tem piscina?", "Busque outros").
       
    2. "ANALISE_CONTEXTO": Se o usuário está pedindo uma comparação, ordenação, resumo ou detalhe sobre os imóveis JÁ listados na resposta anterior. 
       (Ex: "Qual desses é o mais barato?", "O primeiro é novo?", "Qual o maior deles?", "Me fale mais sobre o segundo").
    
    Responda APENAS a palavra-chave (NOVA_BUSCA ou ANALISE_CONTEXTO).
    """
    try:
        # Usamos o llm_reformulador (temp=0) pois ele é rápido e preciso
        decisao = llm.invoke(template.format(pergunta=pergunta_reformulada))
        decisao = decisao.strip().upper()
        # Tratamento de erro básico caso o LLM fale frases inteiras
        if "ANALISE" in decisao or "CONTEXTO" in decisao:
            return "ANALISE_CONTEXTO"
        return "NOVA_BUSCA"
    except:
        return "NOVA_BUSCA" # Na dúvida, vai no banco.

# --- Função de Lógica (Extração de Filtros Manual - COM DESCRIÇÕES RICAS) ---
# --- Função Auxiliar para Sintaxe do Chroma (Mantém essa) ---
def corrigir_sintaxe_chroma(filtros):
    """
    Corrige limitações do ChromaDB e remove operadores alucinados (ex: $min, $max).
    """
    # Lista oficial de operadores suportados pelo ChromaDB
    OPERADORES_VALIDOS = ["$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin"]
    
    lista_condicoes = []

    for campo, criterio in filtros.items():
        # Caso 1: Critério é um dicionário (ex: {"$gte": 100})
        if isinstance(criterio, dict):
            # Filtra apenas os operadores que existem no Chroma
            criterio_limpo = {op: val for op, val in criterio.items() if op in OPERADORES_VALIDOS}
            
            # Se sobrou algo válido, processa
            if criterio_limpo:
                # Se tiver mais de um operador válido (Range), explode em lista
                if len(criterio_limpo) > 1:
                    for op, val in criterio_limpo.items():
                        lista_condicoes.append({campo: {op: val}})
                else:
                    # Se for só um, adiciona direto
                    lista_condicoes.append({campo: criterio_limpo})
        
        # Caso 2: Critério é valor direto (Igualdade implícita)
        else:
            lista_condicoes.append({campo: criterio})

    # Monta a estrutura final
    if len(lista_condicoes) > 1:
        return {"$and": lista_condicoes}
    elif len(lista_condicoes) == 1:
        return lista_condicoes[0]
    else:
        return {}

# --- Função Principal de Filtros (Versão Completa) ---
def extrair_filtros(pergunta, llm):
    """
    Extrai filtros JSON combinando:
    1. Schema rico (para entender o significado dos campos).
    2. Lógica avançada (para lidar com aproximações, sufixos 'k/mi' e tempo).
    """
    
    template = """
    Você é um especialista em banco de dados e normalização de dados.
    Analise a pergunta e transforme em filtros de busca estruturados (JSON).
    Retorne APENAS o JSON. Não explique nada.
    
    SCHEMA DOS DADOS (Use estas definições para entender o contexto):
    
    - predio_id (string): O código único do imóvel (ex: "PREDIO_0001").
    - cidade (string): A cidade onde o prédio está localizado (ex: "Caucaia", "Fortaleza").
    - bairro (string): O nome do bairro (ex: "Centro", "Aldeota", "Jardim").
    - estado (string): A sigla do estado, use sempre maiúsculo (ex: "CE", "SP").
    - cep (string): O código postal. Mantenha a pontuação se houver (ex: "60000-000").
    - endereco (string): Nome da rua ou avenida.
    
    - quantidade_de_salas (int): Número total de salas/escritórios. Use operadores de comparação.
    - tamanho_m2 (int): Área total do imóvel em metros quadrados.
    - preco_estimado (float): Valor de venda do imóvel em Reais.
    - - ano_construcao (int): O ano de construção. Note que é um int no banco. Trate ano como NÚMERO (ex: 2010), não string.

    ---------------------------------------------------------
    REGRAS DE LÓGICA AVANÇADA (Siga estritamente NA ORDEM):
    ---------------------------------------------------------
    1. ID ESPECÍFICO:
       - Se o usuário citar um código/ID (ex: "PREDIO_0025"), filtre APENAS pelo predio_id e ignore o resto.

    2. SUPERLATIVOS ("O mais barato", "O maior", "O mais recente"):
       - Prioridade MÁXIMA. Se for uma pergunta de ordenação ("Qual é o mais..."), PARE.
       - NÃO gere filtro de valor numérico para o campo perguntado. Deixe vazio ou filtre apenas a cidade.
       - Ex: "O mais barato de Caucaia" -> {{ "cidade": "Caucaia" }} (Sem filtro de preço).
       - Ex: "Qual o mais recente?" -> {{ }} (Sem filtro de ano).

    3. VALORES APROXIMADOS ("Por volta de", "Cerca de", "Na faixa de"):
       - NUNCA use igualdade para valores aproximados.
       - Crie um intervalo de -20% e +20%.
       - Ex: "Por volta de 1000" -> {{ "$gte": 800, "$lte": 1200 }}
       - Ex: "Uns 2 milhões" -> {{ "$gte": 1600000, "$lte": 2400000 }}

    4. ABREVIAÇÕES NUMÉRICAS:
       - Converta texto para número puro.
       - "k" = mil (ex: 500k -> 500000)
       - "mi", "milhão", "milhões" = 10^6 (ex: 2mi -> 2000000)

    5. CONCEITOS TEMPORAIS ("Novo", "Recente", "Antigo"):
       - APLIQUE APENAS SE NÃO FOR SUPERLATIVO (Veja Regra 1).
       - Se for busca genérica:
       - "Novo" ou "Recente" -> ano_construcao >= 2020
       - "Antigo" -> ano_construcao <= 2010
       - "Anos 90" -> ano_construcao >= 1990 e <= 1999

    6. PROIBIDO: NUNCA use operadores como $min, $max, $avg, $orderby. 
       Se o usuário pedir "o mais barato", NÃO filtre o preço, filtre apenas a cidade/local.

    ---------------------------------------------------------
    EXEMPLOS (Few-Shot):

    User: "Prédios em Itapipoca com mais de 10 salas"
    AI: {{ "cidade": "Itapipoca", "quantidade_de_salas": {{ "$gt": 10 }} }}

    User: "Imóvel no bairro Centro que custe por volta de 2 milhões"
    AI: {{ "bairro": "Centro", "preco_estimado": {{ "$gte": 1600000, "$lte": 2400000 }} }}

    User: "Qual o prédio do CEP 93260-708?"
    AI: {{ "cep": "93260-708" }}

    User: "Prédios novos acima de 500m2"
    AI: {{ "ano_construcao": {{ "$gte": "2020" }}, "tamanho_m2": {{ "$gt": 500 }} }}

    User: "Qual o mais recente de Sobral?"
    AI: {{ "cidade": "Sobral" }}

    Pergunta Atual: {pergunta}
    JSON:
    """
    try:
        json_str = llm.invoke(template.format(pergunta=pergunta))
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        filtros = json.loads(json_str)
        filtros_limpos = {k: v for k, v in filtros.items() if v}
                
        campos_inteiros = ["ano_construcao", "quantidade_de_salas", "tamanho_m2"]
        
        for campo in campos_inteiros:
            if campo in filtros_limpos:
                valor = filtros_limpos[campo]
                
                # Caso 1: Filtro simples (ex: "ano_construcao": "2010")
                if isinstance(valor, (str, float)):
                    filtros_limpos[campo] = int(valor)
                    
                # Caso 2: Filtro com operador (ex: "ano_construcao": {"$gt": "2010"})
                elif isinstance(valor, dict):
                    for op, v in valor.items():
                        # Converte o valor dentro do operador para int
                        filtros_limpos[campo][op] = int(v)
        # ---------------------------------------

        return corrigir_sintaxe_chroma(filtros_limpos)
        
    except Exception as e:
        print(f"Erro ao extrair filtro: {e}")
        return {}

# --- 3. Interface Streamlit ---
def app():
    st.header("🏢 Imobiliária Inteligente", divider=True)

    db, llm_reformulador, llm_filtro, llm_resposta = carregar_sistema()
    
    if not db:
        st.stop()

    if "mensagens" not in st.session_state:
        st.session_state["mensagens"] = [{"usuario": "assistant", "texto": "Olá! Sou seu corretor virtual. Onde você busca seu imóvel?"}]

    for mensagem in st.session_state["mensagens"]:
        avatar = "🧑‍💻" if mensagem["usuario"] == "user" else "🤖"
        with st.chat_message(mensagem["usuario"], avatar=avatar):
            st.write(mensagem["texto"])

    mensagem_usuario = st.chat_input("Digite sua pergunta aqui...")

    if mensagem_usuario:
        st.session_state["mensagens"].append({"usuario": "user", "texto": mensagem_usuario})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(mensagem_usuario)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            status = st.status("Processando...", expanded=True)
            
            try:
                # --- PASSO 1: MEMÓRIA (Reformulação) ---
                status.write("🧠 Entendendo contexto...")
                historico = st.session_state["mensagens"][:-1]
                pergunta_final = reformular_pergunta(mensagem_usuario, historico, llm_reformulador)
                
                if pergunta_final.lower() != mensagem_usuario.lower():
                    status.write(f"**Interpretei:** *{pergunta_final}*")

                #  --- DECISOR DE ROTA --- 
                intencao = classificar_intencao(pergunta_final, llm_reformulador)
                status.write(f"🧭 Rota definida: **{intencao}**")

                contexto_final = ""
                
                # === ROTA A: NOVA BUSCA (Vai no ChromaDB) ===
                if intencao == "NOVA_BUSCA":
                    status.write("🔍 Consultando banco de dados...")
                    
                    filtros = extrair_filtros(pergunta_final, llm_filtro)
                    if filtros:
                        status.write(f"**Filtros:** {filtros}")
                    
                    docs = db.similarity_search(pergunta_final, k=4, filter=filtros if filtros else None)
                    
                    if not docs:
                        texto_resposta = "Não encontrei imóveis com esses critérios."
                        status.update(label="Sem resultados", state="error")
                        placeholder.write(texto_resposta)
                        st.session_state["mensagens"].append({"usuario": "assistant", "texto": texto_resposta})
                        st.stop()
                    
                    contexto_final = "\n\n".join([d.page_content for d in docs])

                # === ROTA B: ANÁLISE DE CONTEXTO (Lê o Histórico) ===
                else:
                    status.write("📖 Lendo conversa anterior...")
                    
                    # Recupera a última fala da IA (onde estão os dados dos prédios)
                    ultima_resposta_ia = ""
                    for msg in reversed(st.session_state["mensagens"]):
                        if msg["usuario"] == "assistant":
                            ultima_resposta_ia = msg["texto"]
                            break
                    
                    if not ultima_resposta_ia:
                        contexto_final = "Não há contexto anterior disponível."
                    else:
                        contexto_final = f"Informações apresentadas anteriormente pelo Assistente:\n{ultima_resposta_ia}"

                # === GERAÇÃO DA RESPOSTA FINAL (Comum às duas rotas) ===
                status.write("✍️ Escrevendo resposta...")
                
                template_resp = """
                Você é um corretor imobiliário prestativo.
                
                CONTEXTO (Informações do Banco ou da Conversa Anterior):
                {context}

                PERGUNTA DO USUÁRIO (Reformulada): {pergunta_reformulada}
                
                INSTRUÇÃO:
                - Se o contexto tiver imóveis, responda a pergunta usando esses dados.
                - Se o usuário pediu "o mais barato" ou "o mais recente", analise os dados no contexto e conclua.
                - Seja direto e natural.
                """
                
                chain_input = {
                    "context": contexto_final, 
                    "pergunta_reformulada": pergunta_final
                }
                
                texto_resposta = llm_resposta.invoke(template_resp.format(**chain_input))
                status.update(label="Concluído!", state="complete", expanded=False)

            except Exception as e:
                texto_resposta = f"Erro técnico: {str(e)}"
                status.update(label="Erro", state="error")

            placeholder.write(texto_resposta)
            st.session_state["mensagens"].append({"usuario": "assistant", "texto": texto_resposta})

if __name__ == "__main__":
    app()