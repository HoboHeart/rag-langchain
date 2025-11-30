import json
import csv
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
PASTA_BASE = os.path.join(RAIZ_PROJETO, "base")
ARQUIVO_CSV = os.path.join(PASTA_BASE, "base_1_locacoes.csv")
ARQUIVO_JSON_AVALIAR = os.path.join(PASTA_BASE, "base_2_detalhes_predios_V2.json")
ARQUIVO_RELATORIO = os.path.join(PASTA_BASE, "relatorio_auditoria.json")

# --- Configurando o Auditor ---
#Temperature 0.0 -> Faz com que a IA seja rigorosa e analítica, não criativa
print("Carregando o Auditor (Llama 3)...")
llm = Ollama(model="llama3", temperature=0.0)

# --- Carregando os dados ---
print("Lendo arquivos...")

# Carrega CSV (Verdade Absoluta 1)
dados_csv = {}
with open(ARQUIVO_CSV, mode='r', encoding='utf-8-sig') as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        id_limpo = linha['predio_id'].strip()
        dados_csv[id_limpo] = linha

# Carrega JSON V2 (Verdade Absoluta 2 + Descrição para Avaliar)
with open(ARQUIVO_JSON_AVALIAR, 'r', encoding='utf-8') as f:
    imoveis_para_avaliar = json.load(f)

# --- Prompt de Auditoria ---
template_auditoria = """
Você é um Auditor de Qualidade de Dados Imobiliários. Sua tarefa é verificar se uma descrição gerada corresponde aos fatos técnicos.

DADOS TÉCNICOS REAIS (A Verdade):
- Localização: {cidade} - {estado}
- Tamanho: {tamanho} m²
- Preço: R$ {preco}
- Salas: {salas}

DESCRIÇÃO GERADA (Para Validar):
"{descricao}"

REGRAS DE VALIDAÇÃO:
1. Verifique se a descrição cita números diferentes dos dados técnicos (ex: preço errado, tamanho errado).
2. Verifique se a descrição inventa características físicas específicas que NÃO existem nos dados (ex: "tem piscina", "tem elevador panorâmico", "tem 10 andares"). 
3. Adjetivos subjetivos (ex: "lindo", "espaçoso", "excelente oportunidade") SÃO PERMITIDOS e corretos.

Responda estritamente neste formato:
STATUS: [APROVADO ou REPROVADO]
MOTIVO: [Se Reprovado, explique o erro factual. Se Aprovado, escreva apenas "OK"]
"""

prompt_auditor = PromptTemplate(
    template=template_auditoria, 
    input_variables=["cidade", "estado", "tamanho", "preco", "salas", "descricao"]
)

# --- Loop de Auditoria ---
print(f"Auditando {len(imoveis_para_avaliar)} descrições...")

relatorio_erros = []
total_aprovados = 0

for imovel in tqdm(imoveis_para_avaliar):
    predio_id = imovel.get('predio_id')
    info_csv = dados_csv.get(predio_id)
    
    if info_csv:
        # Prepara os dados para o Auditor
        texto_prompt = prompt_auditor.format(
            cidade=info_csv['cidade'],
            estado=info_csv['estado'],
            tamanho=imovel['tamanho_m2'],
            preco=f"{imovel['preco_estimado']:,.2f}",
            salas=imovel['quantidade_de_salas'],
            descricao=imovel['descricao'] # A descrição que geramos antes
        )
        
        # O Auditor analisa
        analise = llm.invoke(texto_prompt)
        
        # Processa a resposta simples
        if "STATUS: REPROVADO" in analise.upper():
            print(f"\n[ALERTA] Problema encontrado no {predio_id}")
            print(analise) # Mostra o motivo no terminal
            relatorio_erros.append({
                "predio_id": predio_id,
                "analise_ia": analise,
                "descricao_suspeita": imovel['descricao']
            })
        else:
            total_aprovados += 1

# --- 5. Resumo Final ---
print("\n" + "="*30)
print(f"AUDITORIA CONCLUÍDA")
print(f"Total Auditado: {len(imoveis_para_avaliar)}")
print(f"Aprovados: {total_aprovados}")
print(f"Suspeitos/Reprovados: {len(relatorio_erros)}")
print("="*30)

if relatorio_erros:
    print(f"Salvando relatório de erros em: {ARQUIVO_RELATORIO}")
    with open(ARQUIVO_RELATORIO, 'w', encoding='utf-8') as f:
        json.dump(relatorio_erros, f, indent=4, ensure_ascii=False)
else:
    print("Parabéns! Nenhuma inconsistência encontrada.")