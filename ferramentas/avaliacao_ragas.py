import os

# 1. Correção do Git (Mantida)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# --- NOVAS IMPORTAÇÕES DO RAGAS (ATUALIZADO) ---
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# --- CONFIGURAÇÕES ---
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
CAMINHO_DB = os.path.join(RAIZ_PROJETO, "db")
ARQUIVO_TESTES = os.path.join(DIRETORIO_ATUAL, "perguntas_auto.json")
ARQUIVO_RESULTADO_CSV = os.path.join(DIRETORIO_ATUAL, "resultado_ragas.csv")

# ==============================================================================
# 1. FUNÇÕES DO RAG (Cópia da lógica do main.py)
# ==============================================================================

def corrigir_sintaxe_chroma(filtros):
    lista_condicoes = []
    for campo, criterio in filtros.items():
        if isinstance(criterio, dict) and len(criterio) > 1:
            for operador, valor in criterio.items():
                lista_condicoes.append({campo: {operador: valor}})
        else:
            lista_condicoes.append({campo: criterio})
    if len(lista_condicoes) > 1:
        return {"$and": lista_condicoes}
    elif len(lista_condicoes) == 1:
        return lista_condicoes[0]
    else:
        return {}

def extrair_filtros(pergunta, llm):
    template = """
    Você é um especialista em banco de dados e normalização de dados.
    Analise a pergunta e transforme em filtros de busca estruturados (JSON).
    Retorne APENAS o JSON. Não explique nada.
    
    SCHEMA DOS DADOS (Use estas definições para entender o contexto):
    
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
    REGRAS DE LÓGICA AVANÇADA (Siga estritamente):
    ---------------------------------------------------------

    1. VALORES APROXIMADOS ("Por volta de", "Cerca de", "Na faixa de"):
       - NUNCA use igualdade para valores aproximados.
       - Crie um intervalo de -20% e +20%.
       - Ex: "Por volta de 1000" -> {{ "$gte": 800, "$lte": 1200 }}
       - Ex: "Uns 2 milhões" -> {{ "$gte": 1600000, "$lte": 2400000 }}

    2. ABREVIAÇÕES NUMÉRICAS:
       - Converta texto para número puro.
       - "k" = mil (ex: 500k -> 500000)
       - "mi", "milhão", "milhões" = 10^6 (ex: 2mi -> 2000000)

    3. CONCEITOS TEMPORAIS ("Novo", "Recente", "Antigo"):
       - "Novo" ou "Recente" -> ano_construcao >= "2020"
       - "Antigo" -> ano_construcao <= "2010"
       - "Anos 90" -> ano_construcao >= "1990" e <= "1999"

    4. SUPERLATIVOS ("O mais barato", "O maior"):
       - NÃO gere filtro de valor para o campo superlativo. Deixe vazio ou filtre apenas a cidade.
       - Ex: "O mais barato de Caucaia" -> {{ "cidade": "Caucaia" }} (Sem filtro de preço).

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
                if isinstance(valor, (str, float)):
                    filtros_limpos[campo] = int(valor)
                elif isinstance(valor, dict):
                    for op, v in valor.items():
                        filtros_limpos[campo][op] = int(v)
                        
        return corrigir_sintaxe_chroma(filtros_limpos)
    except:
        return {}

def gerar_resposta_rag(pergunta, db, llm_filtro, llm_resposta):
    filtros = extrair_filtros(pergunta, llm_filtro)
    try:
        docs = db.similarity_search(pergunta, k=4, filter=filtros if filtros else None)
    except:
        docs = []
        
    if not docs:
        return "Não encontrei imóveis com esses critérios.", []

    contexto_str = "\n\n".join([d.page_content for d in docs])
    template_resp = "Use os imóveis abaixo para responder a pergunta: {pergunta}\n\nImóveis:\n{context}"
    resposta = llm_resposta.invoke(template_resp.format(pergunta=pergunta, context=contexto_str))
    
    lista_contextos = [d.page_content for d in docs]
    return resposta, lista_contextos

# ==============================================================================
# 2. EXECUÇÃO DO RAGAS
# ==============================================================================

def rodar_avaliacao_ragas():
    print("🚀 Iniciando Avaliação com RAGAS...")
    
    # --- Configuração dos Wrappers (A CORREÇÃO ESTÁ AQUI) ---
    print("Carregando modelos...")
    
    # 1. LLM Wrapper
    ollama_model = Ollama(model="llama3", temperature=0)
    ragas_llm = LangchainLLMWrapper(ollama_model) 
    
    # 2. Embeddings Wrapper
    langchain_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    # --- Configuração do RAG do Projeto ---
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=langchain_embeddings)
    llm_filtro = Ollama(model="llama3", temperature=0, format="json")
    llm_resposta = Ollama(model="llama3", temperature=0.7)

    # --- Carregar Perguntas ---
    if not os.path.exists(ARQUIVO_TESTES):
        print("Arquivo de testes não encontrado.")
        return

    with open(ARQUIVO_TESTES, "r", encoding="utf-8") as f:
        dados_json = json.load(f)

    perguntas_validas = []
    for nome, dados in dados_json.items():
        if isinstance(dados, dict) and dados.get("expectativa") == "ENCONTRADO":
            perguntas_validas.append(dados["pergunta"])
    
    # Limita a 3 perguntas para teste rápido (remova depois)
    #perguntas_validas = perguntas_validas[:3]
    print(f"Selecionadas {len(perguntas_validas)} perguntas para avaliação.")
    
    

    # --- Gerar Dataset ---
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        # ground_truth é opcional para Fidelidade e Relevância, então deixamos vazio
    }

    print("Gerando respostas do RAG...")
    for i, pergunta in enumerate(perguntas_validas):
        print(f"[{i+1}/{len(perguntas_validas)}] Processando: {pergunta[:30]}...")
        resposta, contextos = gerar_resposta_rag(pergunta, db, llm_filtro, llm_resposta)
        
        data_samples['question'].append(pergunta)
        data_samples['answer'].append(resposta)
        data_samples['contexts'].append(contextos)

    dataset = Dataset.from_dict(data_samples)

    # --- Executar Avaliação ---
    print("\n🧐 O Ragas está avaliando... (Pode demorar)")
    
    metrics = [faithfulness, answer_relevancy]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,           # Passamos o wrapper novo
        embeddings=ragas_embeddings # Passamos o wrapper novo
    )

    print("\n📊 RESULTADOS DO RAGAS:")
    print(results)
    
    df = results.to_pandas()
    df.to_csv(ARQUIVO_RESULTADO_CSV, index=False)
    print(f"Relatório salvo em: {ARQUIVO_RESULTADO_CSV}")

if __name__ == "__main__":
    rodar_avaliacao_ragas()