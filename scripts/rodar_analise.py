"""Roda a análise completa e grava as tabelas (reports/resultados.json) e as figuras do README."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from lead_scoring import analise, graficos, site
from lead_scoring.dados import ALVO, ARQUIVO_PADRAO, baixar, carregar

SAIDA = Path("reports")
FIGURAS = SAIDA / "figuras"


def _para_json(tabela: pd.DataFrame) -> dict:
    return json.loads(tabela.to_json(orient="index", force_ascii=False))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # Nos primeiros meses ninguém tinha contato anterior, então `pdays` chega todo ausente em
    # algumas janelas de treino e o imputador avisa que vai ignorá-lo. É o comportamento esperado.
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    # Com janela de 12 meses, o bloco de teste pode trazer um mês do ano que não estava no
    # treino. O codificador zera essa categoria (handle_unknown="ignore"), que é o que queremos.
    warnings.filterwarnings("ignore", message="Found unknown categories")
    FIGURAS.mkdir(parents=True, exist_ok=True)

    baixar()
    bruto = pd.read_csv(ARQUIVO_PADRAO, sep=";")
    df = carregar()

    vazamento = analise.demonstrar_vazamento(bruto)
    promessa = analise.aleatoria_x_temporal(df)
    teto = analise.teto_do_regime_novo(df)
    juntas, por_bloco = analise.previsoes_no_tempo(df)
    resumo, diferencas = analise.comparar_concorrentes(juntas)
    calibracao = analise.calibracao_no_tempo(juntas, por_bloco[analise.ESCOLHIDO])
    capacidade = analise.ganho_com_capacidade(juntas)
    curva = analise.curva_de_ganho(juntas)
    grade = analise.grade_de_configuracoes(df)

    figuras = {
        "promessa_x_realidade": graficos.promessa_x_realidade(promessa, teto),
        "auc_por_bloco": graficos.auc_por_bloco(por_bloco),
        "curva_de_ganho": graficos.curva_de_ganho(curva),
        "calibracao": graficos.calibracao(calibracao),
    }
    for nome, figura in figuras.items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)

    resultados = {
        "dados": {
            "leads": int(len(df)),
            "taxa_de_conversao": float(df[ALVO].mean()),
            "periodo": f"{df['periodo'].min()} a {df['periodo'].max()}",
        },
        "vazamento_auc": vazamento,
        "aleatoria_x_temporal": _para_json(promessa),
        "teto_regime_novo_auc": teto,
        "auc_por_bloco": {nome: tabela["auc"].to_dict() for nome, tabela in por_bloco.items()},
        "concorrentes": _para_json(resumo),
        "diferencas": _para_json(diferencas),
        "calibracao": _para_json(calibracao),
        "capacidade": _para_json(capacidade),
        "curva_de_ganho": json.loads(curva.reset_index().round(4).to_json(orient="records", force_ascii=False)),
        "grade_top10": json.loads(grade.head(10).to_json(orient="records", force_ascii=False)),
    }
    # O texto e o gráfico principal do site vêm de outro arquivo; daqui saem só os gráficos extras, calculados a
    # partir do próprio dicionário de resultados (passado por JSON para usar exatamente os números gravados).
    resultados["para_o_site"] = {"graficos_extras": site.graficos_extras(json.loads(json.dumps(resultados, ensure_ascii=False)))}
    (SAIDA / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    for titulo, tabela in [
        ("Aleatória x temporal", promessa),
        ("Concorrentes (AUC no tempo, IC 95%)", resumo),
        ("Diferenças pareadas", diferencas),
        ("Calibração por bloco", calibracao),
        ("Capacidade", capacidade),
        ("Grade de configurações (10 primeiras)", grade.head(10)),
    ]:
        print(f"\n== {titulo} ==")
        print(tabela.to_string())
    print(f"\nVazamento: {vazamento}")
    print(f"Teto do regime novo: {teto:.3f}")
    print(f"\nGravado em {SAIDA / 'resultados.json'} e {FIGURAS}")


if __name__ == "__main__":
    main()
