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
ARQUIVO_JSON_ENTRADA = os.path.join(PASTA_BASE, "base_2_detalhes_predios.json")
ARQUIVO_JSON_SAIDA = os.path.join(PASTA_BASE, "base_2_detalhes_predios_V2.json")

# --- Configurando o Ollama ---
print("Carregando o Ollama...")
llm = Ollama(model="llama3", temperature = 0.7)

# --- Carregamento dos Dados ---
print("Lendo os arquivos...")

# Carrega CSV em um dicionário para busca rápida pelo ID
dados_csv = {}
with open(ARQUIVO_CSV, mode='r', encoding='utf-8-sig') as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        # Ajuste a chave 'predio_id' se necessário (ex: remover espaços)
        id_limpo = linha['predio_id'].strip()
        dados_csv[id_limpo] = linha

# Carrega JSON
with open(ARQUIVO_JSON_ENTRADA, 'r', encoding='utf-8') as f:
    lista_imoveis = json.load(f)

# --- Prompt Utilizado: ---
template = """
Você é um corretor de imóveis experiente especializado em imóveis comerciais.
Escreva uma descrição atraente, profissional e vendedora (em Português do Brasil) para o seguinte imóvel.
Não invente dados que não estão listados, mas use adjetivos adequados ao preço e tamanho.

Dados do Imóvel:
- Tipo: Prédio Comercial
- Localização: {endereco}, {bairro}, {cidade} - {estado}
- Tamanho: {tamanho} m²
- Salas: {salas} salas
- Ano de Construção: {ano}
- Preço de Venda: R$ {preco}

A descrição deve ter no máximo 4 frases. Foque nos pontos fortes.
Descrição:
"""

prompt = PromptTemplate(template=template, input_variables=["endereco", "bairro", "cidade", "estado", "tamanho", "salas", "ano", "preco"])

# --- Geração das descrições ---
print(f"Gerando descrições para {len(lista_imoveis)} imóveis...")

novos_imoveis = []

for imovel in tqdm(lista_imoveis):
    predio_id = imovel.get('predio_id')
    
    # Busca dados do CSV correspondente
    info_csv = dados_csv.get(predio_id)
    
    if info_csv:
        # Formata o prompt com os dados reais
        texto_prompt = prompt.format(
            endereco=info_csv['endereco'],
            bairro=info_csv['bairro'],
            cidade=info_csv['cidade'],
            estado=info_csv['estado'],
            tamanho=imovel['tamanho_m2'],
            salas=imovel['quantidade_de_salas'],
            ano=imovel['ano_construcao'],
            preco=f"{imovel['preco_estimado']:,.2f}"
        )
        
        # Pede para a IA gerar
        nova_descricao = llm.invoke(texto_prompt)
        
        # Limpa quebras de linha extras
        nova_descricao = nova_descricao.strip()
        
        # Atualiza o objeto imóvel
        imovel['descricao'] = nova_descricao
    
    novos_imoveis.append(imovel)

# --- Salvando novo arquivo: ---
print("Salvando novo arquivo JSON...")
with open(ARQUIVO_JSON_SAIDA, 'w', encoding='utf-8') as f:
    json.dump(novos_imoveis, f, indent=4, ensure_ascii=False)

print(f"Sucesso! Novo arquivo salvo em: {ARQUIVO_JSON_SAIDA}")
print("!!!Lembrar de atualizar o CAMINHO_JSON para usar esse arquivo V2.!!!")