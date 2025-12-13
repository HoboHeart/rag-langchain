import os
import json
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Importações do LangChain ---
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- Configurações de Caminho ---
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
CAMINHO_DB = os.path.join(RAIZ_PROJETO, "db")
ARQUIVO_TESTES = os.path.join(DIRETORIO_ATUAL, "perguntas_auto.json")
ARQUIVO_RESULTADOS_CSV = os.path.join(DIRETORIO_ATUAL, "resultados_tcc.csv")

# ==============================================================================
# 1. FUNÇÕES DO SISTEMA (Cópia fiel do main.py para validade do teste)
# ==============================================================================

def corrigir_sintaxe_chroma(filtros):
    """Corrige estrutura para o Chroma ($and)."""
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
    """
    Extrai filtros JSON combinando Schema Rico e Blindagem de Tipos.
    """
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
        
        # --- BLINDAGEM DE TIPOS (Igual ao main.py) ---
        campos_inteiros = ["ano_construcao", "quantidade_de_salas", "tamanho_m2"]
        for campo in campos_inteiros:
            if campo in filtros_limpos:
                valor = filtros_limpos[campo]
                # Caso 1: Valor direto
                if isinstance(valor, (str, float)):
                    filtros_limpos[campo] = int(valor)
                # Caso 2: Operador (Dicionário)
                elif isinstance(valor, dict):
                    for op, v in valor.items():
                        filtros_limpos[campo][op] = int(v)
        # ---------------------------------------------

        return corrigir_sintaxe_chroma(filtros_limpos)
        
    except Exception as e:
        print(f"   [!] Erro na extração de filtros: {e}")
        return {}

# ==============================================================================
# 2. FUNÇÕES DE AUDITORIA E BUSCA
# ==============================================================================

def buscar_contexto_manual(pergunta, db, llm_filtro):
    """Executa o pipeline: Filtro -> Busca"""
    filtros = extrair_filtros(pergunta, llm_filtro)
    try:
        # Tenta buscar. Se o filtro for inválido, o Chroma pode dar erro.
        docs = db.similarity_search(pergunta, k=4, filter=filtros if filtros else None)
        return docs, filtros
    except Exception as e:
        return [], filtros

def auditar_relevancia(pergunta, docs, llm_auditor):
    """
    Usa um LLM (Gemma 2) para verificar se o texto encontrado realmente tem a ver com a pergunta.
    Isso evita que o sistema ache um prédio aleatório e conte como 'Acerto'.
    """
    if not docs:
        return "NÃO"
        
    contexto = "\n".join([d.page_content for d in docs])
    
    prompt = PromptTemplate.from_template(
        """
        Você é um auditor rigoroso. Responda APENAS 'SIM' ou 'NÃO'.
        
        O CONTEXTO abaixo contém as informações solicitadas na PERGUNTA?
        (Verifique se a cidade, bairro, preço ou características batem com o pedido).

        PERGUNTA: {pergunta}
        CONTEXTO ENCONTRADO:
        {contexto}
        
        Resposta (SIM/NÃO):
        """
    )
    return llm_auditor.invoke(prompt.format(contexto=contexto, pergunta=pergunta)).strip().upper()

# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================

def gerar_relatorio_tcc():
    print("\n📊 INICIANDO ANÁLISE ESTATÍSTICA (TCC) 📊")
    print("--------------------------------------------------")
    
    # 1. Configuração dos Modelos
    # O Sistema (Aluno) usa Llama 3 (igual ao app real)
    llm_filtro = Ollama(model="llama3", temperature=0, format="json")
    
    # O Auditor (Professor) usa Gemma 2 (para evitar viés)
    # Se ficar muito lento, você pode mudar para "llama3", mas gemma2 é melhor pro TCC.
    llm_auditor = Ollama(model="gemma2", temperature=0) 
    
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if not os.path.exists(CAMINHO_DB):
        print(f"❌ Erro Crítico: Banco de dados não encontrado em {CAMINHO_DB}")
        print("   Por favor, rode 'python criar_db.py' antes.")
        return

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embedder)

    # 2. Carregar Testes
    with open(ARQUIVO_TESTES, "r", encoding="utf-8") as f:
        testes = json.load(f)

    y_true = [] # Gabarito
    y_pred = [] # Resposta do Sistema
    detalhes = []

    print(f"Processando {len(testes)} cenários de teste...")
    
    for nome, dados in testes.items():
        pergunta = dados["pergunta"]
        expectativa = dados["expectativa"]
        
        print(f"\n🔹 Testando: {nome}")
        print(f"   Pergunta: {pergunta}")
        
        # --- A. Execução do RAG ---
        docs, filtros_usados = buscar_contexto_manual(pergunta, db, llm_filtro)
        qtd_docs = len(docs)
        print(f"   Filtros Gerados: {filtros_usados}")
        print(f"   Documentos Retornados: {qtd_docs}")

        # --- B. Avaliação do Resultado ---
        
        # Cenário 1: Esperava encontrar (Teste Positivo)
        if expectativa == "ENCONTRADO":
            valor_esperado = 1
            
            if qtd_docs == 0:
                print("   ❌ Falha: Não retornou documentos.")
                valor_predito = 0
            else:
                # Auditoria: Achou documentos, mas são os certos?
                veredicto = auditar_relevancia(pergunta, docs, llm_auditor)
                if "SIM" in veredicto:
                    print("   ✅ Sucesso: Documentos relevantes encontrados.")
                    valor_predito = 1
                else:
                    print("   ⚠️ Falha: Retornou documentos, mas o Auditor disse que são irrelevantes.")
                    valor_predito = 0

        # Cenário 2: NÃO esperava encontrar (Teste Negativo)
        else: # "NAO_ENCONTRADO"
            valor_esperado = 0
            
            if qtd_docs == 0:
                print("   ✅ Sucesso: Corretamente não retornou nada.")
                valor_predito = 0 # O sistema previu "classe negativa" corretamente
            else:
                print("   ❌ Falha (Alucinação): Retornou documentos quando não deveria.")
                valor_predito = 1 # O sistema previu "classe positiva" incorretamente

        # --- C. Registro ---
        y_true.append(valor_esperado)
        y_pred.append(valor_predito)

        detalhes.append({
            "Teste": nome,
            "Pergunta": pergunta,
            "Filtros_JSON": str(filtros_usados),
            "Docs_Retornados": qtd_docs,
            "Esperado": valor_esperado,
            "Predito": valor_predito,
            "Acertou": valor_esperado == valor_predito
        })

    # 3. Cálculo das Métricas
    try:
        acuracia = accuracy_score(y_true, y_pred)
        precisao = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception as e:
        print(f"Erro ao calcular métricas: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
        acuracia, precisao, recall, f1 = 0, 0, 0, 0

    # 4. Exibição Final
    print("\n" + "="*50)
    print("📊 RESULTADOS ESTATÍSTICOS FINAIS (TCC)")
    print("="*50)
    print(f"Total de Casos: {len(y_true)}")
    print("-" * 30)
    print(f"✅ Acurácia (Accuracy):  {acuracia:.2%}")
    print(f"🎯 Precisão (Precision): {precisao:.2%}")
    print(f"🔎 Revocação (Recall):   {recall:.2%}")
    print(f"⚖️ F1-Score:             {f1:.2%}")
    print("-" * 30)
    print("MATRIZ DE CONFUSÃO:")
    print(f"[TP] Verdadeiros Positivos (Achou o certo):   {tp}")
    print(f"[TN] Verdadeiros Negativos (Ignorou o errado): {tn}")
    print(f"[FP] Falsos Positivos (Alucinou resultados):  {fp}")
    print(f"[FN] Falsos Negativos (Não achou o que tinha): {fn}")
    print("="*50)

    # 5. Salvar CSV
    if detalhes:
        keys = detalhes[0].keys()
        with open(ARQUIVO_RESULTADOS_CSV, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(detalhes)
        print(f"\nArquivo detalhado salvo em: {ARQUIVO_RESULTADOS_CSV}")

if __name__ == "__main__":
    gerar_relatorio_tcc()