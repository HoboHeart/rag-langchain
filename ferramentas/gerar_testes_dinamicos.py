import json
import csv
import os
import random

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)
PASTA_BASE = os.path.join(RAIZ_PROJETO, "base")
ARQUIVO_CSV = os.path.join(PASTA_BASE, "base_1_locacoes.csv")
ARQUIVO_JSON = os.path.join(PASTA_BASE, "base_2_detalhes_predios_V2.json")

def carregar_dados_reais():
    
    dados_completos = []
    locais = {}
    with open(ARQUIVO_CSV, mode='r', encoding='utf-8-sig') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            locais[linha['predio_id']] = linha

    with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
        detalhes = json.load(f)

    for item in detalhes:
        pid = item.get('predio_id')
        local = locais.get(pid, {})
        dados_completos.append({**item, **local})
        
    return dados_completos

def gerar_testes_mistos(dados, qtd_positivos=10, qtd_negativos=5):
    """Gera uma mistura de testes que DEVEM funcionar e testes que DEVEM falhar."""
    
    testes_finais = {}
    
    # --- 1. Gerar Casos Positivos (Baseado na Verdade) ---
    templates_pos = [
        ("Positivo: Cidade", "Quais imóveis estão em {valor}?", lambda d: d.get('cidade')),
        ("Positivo: Salas", "Mostre prédios com mais de {valor} salas.", lambda d: int(d.get('quantidade_de_salas', 0)) - 2),
        ("Positivo: Bairro", "Imóveis no bairro {valor}.", lambda d: d.get('bairro')),
    ]
    
    for i in range(qtd_positivos):
        tipo, frase, extrator = random.choice(templates_pos)
        imovel = random.choice(dados)
        try:
            val = extrator(imovel)
            pergunta = frase.format(valor=val)
            
            testes_finais[f"Teste #{i+1} (Esperado: SIM)"] = {
                "pergunta": pergunta,
                "expectativa": "ENCONTRADO" # Esperamos que o RAG ache algo
            }
        except: continue

    # --- 2. Gerar Casos Negativos (Alucinação/Filtro Vazio) ---
    # Cidades e Bairros que sabemos que NÃO estão na base
    lugares_falsos = ["Tóquio", "Nova York", "Gotham City", "Terra Média", "Bairro do Dono Menino"]
    
    templates_neg = [
        ("Negativo: Cidade Falsa", "Quero comprar um prédio em {valor}."),
        ("Negativo: Preço Impossível", "Procuro um prédio comercial por menos de R$ 100,00."),
        ("Negativo: Salas Absurdas", "Você tem algum prédio com mais de 5000 salas?"),
        ("Negativo: Ano Futuro", "Mostre prédios construídos em 2050.")
    ]

    for i in range(qtd_negativos):
        nome_tipo, frase = random.choice(templates_neg)
        
        # Escolhe um valor falso
        valor_falso = random.choice(lugares_falsos) 
        pergunta = frase.format(valor=valor_falso)
        
        testes_finais[f"Teste Negativo #{i+1} (Esperado: NÃO)"] = {
            "pergunta": pergunta,
            "expectativa": "NAO_ENCONTRADO" # Esperamos que o RAG retorne vazio
        }

    return testes_finais

# --- Execução ---
if __name__ == "__main__":
    print("Carregando dados...")
    dados = carregar_dados_reais()
    
    # Gera 15 positivos e 5 negativos
    bateria_testes = gerar_testes_mistos(dados, qtd_positivos=15, qtd_negativos=5)
    
    with open("perguntas_auto.json", "w", encoding="utf-8") as f:
        json.dump(bateria_testes, f, indent=4, ensure_ascii=False)
        
    print(f"Gerados {len(bateria_testes)} testes em 'perguntas_auto.json'.")