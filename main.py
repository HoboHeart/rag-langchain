from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

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

# --- Configuração do Self-Querying ---

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
    # Opcional: Endereço (rua). Útil se o usuário busca pelo nome da rua.
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

# Carregando o Llama 3 para ser o "Tradutor de Filtros"
llm_filter = Ollama(model="llama3", temperature=0) # Temp 0 para precisão lógica

# Criamos o Retriever Inteligente
retriever = SelfQueryRetriever.from_llm(
    llm_filter,
    db,
    document_content_description,
    metadata_field_info,
    verbose=True # Deixe True para ver o filtro sendo criado no console!
)

# --- Loop de Perguntas ---

# Template para a resposta final
template_resposta = """
Você é um assistente imobiliário. Use as informações abaixo para responder.
Se não houver imóveis listados, diga que não encontrou nada com essas características.

Imóveis Encontrados:
{context}

Pergunta do Usuário: {question}
"""
prompt_final = PromptTemplate.from_template(template_resposta)
llm_resposta = Ollama(model="llama3", temperature=0.7) # Temp maior para texto fluido

print("\n--- Sistema Pronto ---")

while True:
    pergunta = input("\nEscreva sua pergunta (ou 'sair'): ")
    if pergunta.lower() == 'sair':
        break

    print(f"Analisando filtros e buscando...")
    
    try:
        #O retriever aplica o filtro antes da busca
        docs = retriever.invoke(pergunta)
        
        if not docs:
            print(">>> Nenhum imóvel encontrado com esses filtros exatos.")
        else:
            print(f">>> Encontrados {len(docs)} imóveis compatíveis.")
            
            # Preparar contexto para o LLM responder
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # Gerar resposta final
            chain_input = {"context": contexto, "question": pergunta}
            resposta_final = llm_resposta.invoke(prompt_final.format(**chain_input))
            
            print("\n--- Resposta ---")
            print(resposta_final)
            
    except Exception as e:
        print(f"Erro ao processar filtros: {e}")
        print("Tentando busca simples...")
        docs = db.similarity_search(pergunta)
        # ... fallback para busca simples se o filtro falhar ...




#resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)

#print(resultados)
#print(len(resultados))