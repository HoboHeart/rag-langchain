import os
from langchain_community.document_loaders import CSVLoader, JSONLoader
from typing import List, Dict, Any
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



# --- Definindo os caminhos ---
PASTA_BASE = "base"
CAMINHO_CSV = os.path.join(PASTA_BASE,"base_1_locacoes.csv")
CAMINHO_JSON = os.path.join(PASTA_BASE,"base_2_detalhes_predios.json")

def criar_db():
    #carregar documentos
    documentos_brutos = carregar_documentos()
    #print(documentos)

    documentos_unificados = unificar_documentos(documentos_brutos)

    #dividir os chunks
    chunks = dividir_chunks(documentos_unificados)

    #vetorizar os chunks com o processo de embedding
    vetorizar_chunks(chunks)


# --- Carregador do JSON ---
def extrair_metadados_json(record: Dict[str, Any], default_metadata: Dict[str, Any]) -> Dict[str, Any]:
    #Extrai todos os campos como metadados e define 'predio_id' como a fonte

    metadata = default_metadata.copy()
    metadata.update(record)

    #Remove o campo 'descricao' dos metadados, pois ela estará no page_content
    metadata.pop("descricao", None)

    #Define o 'predio_id' como a 'source'
    if "predio_id" in metadata:
        metadata["source"] = metadata["predio_id"]
    
    return metadata

def carregar_documentos_json() -> List[Document]:
    #Carrega documentos do arquivo JSON com controle de conteúdo e metadados.

    try:
        carregador_json = JSONLoader(
            file_path = CAMINHO_JSON,
            jq_schema ='.[]',
            content_key = 'descricao',
            metadata_func = extrair_metadados_json,
        )
        documentos = carregador_json.load()
        print(f"Carregados {len(documentos)} documentos do JSON.")
        
        return documentos
    
    except Exception as e:
        print(f"Erro ao carregar JSON: {e}")
        return []

def carregar_documentos_csv() -> List[Document]:
    #Carrega documentos do arquivo CSV com controle de conteúdo e metadados.

    colunas_conteudo = ["endereco", "bairro", "cidade"]
    colunas_metadata = ["predio_id", "cep", "estado"]

    try:
        carregador_csv = CSVLoader(
            file_path = CAMINHO_CSV,
            encoding = "utf-8-sig",
            source_column = "predio_id",
            content_columns = colunas_conteudo,
            metadata_columns = colunas_metadata 
        )
        documentos = carregador_csv.load()
        print(f"Carregados {len(documentos)} documentos do CSV.")
        return documentos
    
    except Exception as e:
        # --- MELHOR RELATÓRIO DE ERRO ---
        # Isso nos dará o erro real, e não a mensagem genérica
        print(f"\n--- ERRO DETALHADO AO CARREGAR CSV ---")
        print(f"{repr(e)}")
        print("--------------------------------------\n")
        
        # Dicas comuns baseadas no erro
        if "does not exist in" in str(e):
            print(">>> DICA: O erro indica que um NOME DE COLUNA está errado. Verifique os arrays 'colunas_conteudo' e 'colunas_metadata' e compare com seu arquivo CSV.")
        
        return []

def carregar_documentos():
    
    #1. Carregando os documentos CSV
    documentos_csv = carregar_documentos_csv()

    #2. Carregando os documentos JSON
    documentos_json = carregar_documentos_json()

    #3. Combinando documentos
    documentos_totais = documentos_csv + documentos_json

    print(f"--- Sucesso! Total de {len(documentos_totais)} documentos carregados. ---")
    return documentos_totais

def unificar_documentos(documentos: List[Document]) -> List[Document]:
    """
    Combina os documentos CSV e JSON usando o 'predio_id' como chave.
    Cria um 'page_content' rico (Ficha Técnica) para a busca.
    """
    print(f"\nIniciando a Unificação de {len(documentos)} documentos...")

    # 1. Separando os documentos por fonte e indexando pelo predio_id
    docs_csv = {}
    docs_json = {}

    for doc in documentos:
        predio_id = doc.metadata.get("source")
        if not predio_id:
            continue
        
        if "tamanho_m2" in doc.metadata:
            docs_json[predio_id] = doc
        elif "cep" in doc.metadata:
            docs_csv[predio_id] = doc
    
    # 2. Criando a lista final de documentos unificados
    documentos_unificados = []

    # Iterando pelos prédios que temos no CSV
    for predio_id, doc_csv in docs_csv.items():
        doc_json = docs_json.get(predio_id)
        
        if not doc_json:
            print(f"Aviso: Prédio {predio_id} do CSV não encontrado no JSON")
            continue

        # --- CORREÇÃO DA LÓGICA DE COLETA ---

        # 3. Unifica todos os METADADOS primeiro
        metadata_final = doc_csv.metadata.copy()
        metadata_final.update(doc_json.metadata)

        # 4. Pega os dados dos locais corretos:
        
        # 'endereco' VEM DO page_content DO DOCUMENTO CSV
        endereco = doc_csv.page_content 
        
        # 'descricao' VEM DO page_content DO DOCUMENTO JSON
        descricao = doc_json.page_content 
        
        # O resto VEM DOS METADADOS unificados
        ano = metadata_final.get('ano_construcao', 'N/A')
        tamanho = metadata_final.get('tamanho_m2', 'N/A')
        salas = metadata_final.get('quantidade_de_salas', 'N/A')
        preco = metadata_final.get('preco_estimado', 0)

        # 5. CRIA O NOVO PAGE_CONTENT (A "Ficha Técnica")
        page_content_final = f"""
Ficha do Prédio: {predio_id}
Localização: {endereco.replace("\n", ", ")}
Ano de Construção: {ano}
Tamanho: {tamanho} m²
Quantidade de Salas: {salas}
Preço Estimado: R$ {preco:,.2f}
Descrição: {descricao}
"""
        # 6. (Opcional, mas bom) Adiciona a descrição e endereço
        #    também aos metadados para referência
        metadata_final["endereco_completo"] = endereco
        metadata_final["descricao"] = descricao

        # 7. Cria o novo documento unificado
        documentos_unificados.append(
            Document(
                page_content=page_content_final,
                metadata=metadata_final
            )
        )

    print(f"Tudo Certo! {len(documentos_unificados)} documentos unificados criados.")
    return documentos_unificados

def dividir_chunks(documentos):
    #Dividindo os documentos, agora que a base está unificada e combinada, 
    # podemos utilizar um chunk_size mais curto, pois os dados estarão próximos um do outro
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True
    )

    print(f"Dividindo {len(documentos)} documentos em chunks...")
    chunks = separador_documentos.split_documents(documentos)
    print(f"Divisão concluída. Total de {len(chunks)} chunks criados.")

    return chunks

def vetorizar_chunks(chunks: List[Document]):
    
    #1. Definindo o modelo de embedding local (como os dados estão em pt-br escolhi um multilingual)

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'} #força o uso da cpu
    encode_kwargs = {'normalize_embeddings': True}

    embeddings_locais = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    #2. definindo o diretório
    db_directory = "db"

    #3. Criando o banco de dados
    db = Chroma.from_documents(
        chunks,
        embeddings_locais,
        persist_directory=db_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vetorização concluída! Banco de dados salvo em '{db_directory}'.")

criar_db()