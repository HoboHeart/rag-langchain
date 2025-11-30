from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

from langchain_core.prompts import PromptTemplate

import json
from langchain_core.prompts import PromptTemplate

CAMINHO_DB = "db"

# Modelo de Embedding - Utilizando o mesmo da criação
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

funcao_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = Chroma(persist_directory = CAMINHO_DB, embedding_function=funcao_embedding)
print("Banco de Dados Carregado.")

# ---Configuração dos Filtros de forma Manual ---

# Em vez de usar a classe SelfQueryRetriever, usamos um prompt direto.
# Deste modo podemos adquirir os scores após a resposta
template_filtro = """
Você é um tradutor de filtros para banco de dados imobiliário.
Analise a pergunta e retorne APENAS um objeto JSON com os filtros.

Campos disponíveis:
- cidade (string)
- estado (string)
- quantidade_de_salas (int) -> use $gt, $lt, $gte, $lte se necessário
- tamanho_m2 (int) -> use $gt, $lt, $gte, $lte se necessário
- preco_estimado (float) -> use $gt, $lt, $gte, $lte se necessário

Regras:
- Para igualdade: {{ "campo": "valor" }}
- Para "maior que": {{ "campo": {{ "$gt": valor }} }}
- Para "menor que": {{ "campo": {{ "$lt": valor }} }}
- Se não houver filtro claro, retorne {{}}

Exemplos:
P: "Prédios em Caucaia com mais de 20 salas"
R: {{ "cidade": "Caucaia", "quantidade_de_salas": {{ "$gt": 20 }} }}

P: "Imóveis baratos no Ceará"
R: {{ "estado": "CE" }}

Pergunta: {pergunta}
JSON:
"""

prompt_filtro = PromptTemplate.from_template(template_filtro)
llm_filtro = Ollama(model="llama3", temperature=0, format="json")

# --- Configurando o Prompt de Resposta Final ---

template_resposta = """
Você é um assistente imobiliário. Use as informações abaixo para responder.
Se não houver imóveis listados, diga que não encontrou nada com essas características.

Imóveis Encontrados:
{context}

Pergunta do Usuário: {question}
"""
prompt_final = PromptTemplate.from_template(template_resposta)
llm_resposta = Ollama(model="llama3", temperature=0.7)

print("\n--- Sistema Pronto (Com Visualização de Scores) ---")

# --- Loop de Perguntas ---

while True:
    pergunta = input("\nEscreva sua pergunta (ou 'sair'): ")
    if pergunta.lower() == 'sair':
        break

    print(f"1. Gerando filtros inteligentes...")
    
    try:
        # PASSO 1: O LLM cria o filtro JSON
        json_filtro = llm_filtro.invoke(prompt_filtro.format(pergunta=pergunta))
        filtros = json.loads(json_filtro)
        
        # Limpeza simples de chaves vazias
        filtros = {k: v for k, v in filtros.items() if v}
        
        if filtros:
            print(f"   -> Filtro Aplicado: {filtros}")
        else:
            print("   -> Busca Semântica Pura (Sem filtros de metadados)")

        # PASSO 2: Busca no Chroma COM SCORES e COM FILTRO
        # Aqui está a mágica que você queria:
        resultados_com_score = db.similarity_search_with_relevance_scores(
            pergunta, 
            k=3,                 # Top 3 resultados
            filter=filtros if filtros else None  # Aplica o filtro do LLM
        )

        # PASSO 3: Visualização Técnica 
        print("\n--- 🔍 DETALHES DA BUSCA (DEBUG) ---")
        if not resultados_com_score:
            print("Nenhum resultado encontrado com esses filtros/keywords.")
        else:
            print(f"Encontrados {len(resultados_com_score)} documentos.\n")
            for i, (doc, score) in enumerate(resultados_com_score):
                # Nota: Com 'hnsw:space': 'cosine', o score já vem convertido para Relevância (0 a 1)
                # Se vier Distância, Relevância = 1 - score.
                print(f"RESULTADO #{i+1}:")
                print(f"   Score: {score:.4f}") # Mostra 4 casas decimais
                print(f"   ID: {doc.metadata.get('source', 'N/A')}")
                print(f"   Trecho: {doc.page_content[:100].replace(chr(10), ' ')}...") # Só os primeiros 100 chars
                print("-" * 30)

        # PASSO 4: Gerar Resposta Final (Se tiver resultados)
        if resultados_com_score:
            print("\n--- 🤖 RESPOSTA DO RAG ---")
            contexto = "\n\n".join([doc.page_content for doc, score in resultados_com_score])
            
            chain_input = {"context": contexto, "question": pergunta}
            resposta_final = llm_resposta.invoke(prompt_final.format(**chain_input))
            
            print(resposta_final)
            
    except Exception as e:
        print(f"Erro no processamento: {e}")