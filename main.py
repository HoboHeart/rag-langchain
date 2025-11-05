from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CAMINHO_DB = "db"

prompt_template = """
Responda a pergunta do usuário:
{pergunta}

 com base nessas informações:
 {base_conhecimento}"""

pergunta = input("Escreva sua pergunta: ")

#Carregar o Banco de dados:
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

funcao_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)


db = Chroma(persist_directory = CAMINHO_DB, embedding_function=funcao_embedding)
print("Banco de Dados Carregado.")

resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)

print(resultados)
print(len(resultados))