import os
import json
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
CAMINHO_DB = os.path.join(RAIZ_PROJETO, "db")
ARQUIVO_TESTES = os.path.join(DIRETORIO_ATUAL, "perguntas_auto.json")
MODELO_LLM = "llama3" # O modelo que você está usando no Ollama

# Metadados que o LLM precisa para criar filtros
metadata_field_info = [
    AttributeInfo(name="bairro", description="O nome do bairro", type="string"),
    AttributeInfo(name="cidade", description="A cidade", type="string"),
    AttributeInfo(name="estado", description="A sigla do estado", type="string"),
    AttributeInfo(name="cep", description="O CEP", type="string"),
    AttributeInfo(name="endereco", description="O nome da rua, avenida ou endereço do imóvel",type="string",),
    AttributeInfo(name="quantidade_de_salas", description="Número de salas", type="integer"),
    AttributeInfo(name="tamanho_m2", description="O tamanho total do prédio em metros quadrados", type="integer",),
    AttributeInfo(name="preco_estimado", description="O valor de venda", type="float"),
    AttributeInfo(name="ano_construcao", description="O ano de construção", type="string"),
]

# --- 1. FUNÇÃO DE BUSCA ---
def buscar_contexto(pergunta: str, db: Chroma, llm_filter: Ollama) -> List[Document]:
    """Executa a busca usando SelfQueryRetriever."""
    document_content_description = "Ficha técnica de prédios comerciais"
    
    retriever = SelfQueryRetriever.from_llm(
        llm_filter,
        db,
        document_content_description,
        metadata_field_info,
        verbose=False 
    )
    try:
        return retriever.invoke(pergunta)
    except Exception as e:
        print(f"   [ERRO RETRIEVER]: {e}")
        return []

# --- 2. FUNÇÃO DE AUDITORIA (LLM JUIZ) ---
def auditar_contexto(pergunta: str, contexto: str, llm_auditor: Ollama) -> str:
    """Verifica se o contexto responde a pergunta."""
    prompt = PromptTemplate(
        template="""
        Você é um auditor. Responda APENAS com 'SIM' ou 'NÃO'.
        
        O 'CONTEXTO' abaixo contém a informação exata para responder a 'PERGUNTA'?
        
        CONTEXTO: {contexto}
        PERGUNTA: {pergunta}
        
        Resposta (SIM/NÃO):
        """,
        input_variables=["contexto", "pergunta"]
    )
    return llm_auditor.invoke(prompt.format(contexto=contexto, pergunta=pergunta)).strip().upper()

# --- 3. LOOP DE TESTES ---
def rodar_testes():
    print(f"Carregando testes de: {ARQUIVO_TESTES}...")
    
    if not os.path.exists(ARQUIVO_TESTES):
        print("❌ Arquivo de testes não encontrado! Rode o gerador primeiro.")
        return

    with open(ARQUIVO_TESTES, "r", encoding="utf-8") as f:
        testes = json.load(f)

    # Setup do RAG
    llm_base = Ollama(model=MODELO_LLM, temperature=0)
    llm_auditor = Ollama(model=MODELO_LLM, temperature=0)
    
    funcao_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    print("="*60)
    print(f"INICIANDO BATERIA DE {len(testes)} TESTES")
    print("="*60)

    placar = {"APROVADO": 0, "REPROVADO": 0}

    for nome_teste, dados in testes.items():
        # Extrai os dados do novo formato JSON
        pergunta = dados["pergunta"]
        expectativa = dados["expectativa"] # "ENCONTRADO" ou "NAO_ENCONTRADO"

        print(f"\n🔸 TESTE: {nome_teste}")
        print(f"   Pergunta: '{pergunta}'")
        print(f"   Expectativa: {expectativa}")

        # 1. Executa a Busca
        docs = buscar_contexto(pergunta, db, llm_base)
        qtd_docs = len(docs)
        
        resultado_teste = "INCONCLUSIVO"

        # 2. Validação Lógica
        if expectativa == "NAO_ENCONTRADO":
            # Cenário Negativo (Ex: Prédio em Marte)
            if qtd_docs == 0:
                print("   ✅ APROVADO: Sistema não retornou nada (Correto).")
                resultado_teste = "APROVADO"
            else:
                print(f"   ❌ REPROVADO: Sistema encontrou {qtd_docs} docs (Alucinação).")
                # Opcional: Imprimir o que ele achou para debug
                # print(f"      Achou: {docs[0].page_content[:50]}...")
                resultado_teste = "REPROVADO"

        elif expectativa == "ENCONTRADO":
            # Cenário Positivo (Ex: Prédio em Caucaia)
            if qtd_docs == 0:
                print("   ❌ REPROVADO: Sistema não encontrou nada (Deveria existir).")
                resultado_teste = "REPROVADO"
            else:
                # Se achou algo, precisamos ver se é relevante (Contextual Relevancy)
                contexto_str = "\n".join([d.page_content for d in docs])
                verificacao = auditar_contexto(pergunta, contexto_str, llm_auditor)
                
                if "SIM" in verificacao:
                    print("   ✅ APROVADO: Documentos relevantes encontrados.")
                    resultado_teste = "APROVADO"
                else:
                    print(f"   ❌ REPROVADO: Achou docs, mas o conteúdo não responde a pergunta.")
                    resultado_teste = "REPROVADO"

        placar[resultado_teste] += 1

    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print(f"✅ Aprovados: {placar['APROVADO']}")
    print(f"❌ Reprovados: {placar['REPROVADO']}")
    taxa = (placar['APROVADO'] / len(testes)) * 100
    print(f"📊 Taxa de Sucesso: {taxa:.1f}%")
    print("="*60)

if __name__ == "__main__":
    rodar_testes()