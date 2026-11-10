"""Orquestra a análise completa. Cada função devolve uma tabela usada no notebook e no README."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from lead_scoring import avaliacao, negocio
from lead_scoring.dados import ALVO
from lead_scoring.features import CONJUNTOS, MACRO, montar_features
from lead_scoring.modelos import TIPOS, criar_modelo

# Primeiro período de cada bloco de teste. O primeiro bloco começa quando já existem 11 meses
# de histórico para treinar; os demais dividem o restante em blocos de 900 a 1.700 leads.
INICIOS = ["2009-04", "2009-06", "2009-09", "2010-03", "2010-07"]
JANELA_MESES = 12

ENXUTO = ["contact", "campaign", "previous", "contatado_antes", "poutcome", "age"]
TUDO = CONJUNTOS["tudo (inclui macro)"]

LIGHTGBM = "LightGBM, todas as variáveis"
ESCOLHIDO = "Logística, 6 variáveis"
REGRA = "Regra de bolso (sem modelo)"
CONCORRENTES = {LIGHTGBM: ("lightgbm", TUDO), ESCOLHIDO: ("logistica", ENXUTO)}


def demonstrar_vazamento(bruto: pd.DataFrame) -> dict[str, float]:
    """AUC em validação aleatória com e sem `duration`, a variável que só existe após a ligação."""
    y = (bruto["y"] == "yes").astype(int)
    X = montar_features(bruto.rename(columns=lambda c: c.replace(".", "_")))[ENXUTO]
    X_com = X.assign(duration=bruto["duration"])
    colunas_com = [*ENXUTO, "duration"]
    return {
        "com duration": avaliacao.validacao_aleatoria(partial(criar_modelo, "logistica", colunas_com), X_com, y)["auc"],
        "sem duration": avaliacao.validacao_aleatoria(partial(criar_modelo, "logistica", ENXUTO), X, y)["auc"],
    }


def macro_dentro_do_mes(df: pd.DataFrame) -> pd.DataFrame:
    """Quantos valores distintos cada indicador macro assume dentro de um mesmo mês."""
    distintos = df.groupby("periodo")[MACRO].nunique()
    return pd.DataFrame({"mediana de valores distintos por mês": distintos.median(), "máximo": distintos.max()})


def aleatoria_x_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """A mesma dupla de modelos avaliada de dois jeitos: embaralhando o tempo e respeitando-o."""
    X = montar_features(df)
    y = df[ALVO]
    linhas = []
    for nome, (tipo, colunas) in CONCORRENTES.items():
        criar = partial(criar_modelo, tipo, colunas)
        aleatoria = avaliacao.validacao_aleatoria(criar, X[colunas], y)
        por_bloco, _ = avaliacao.validacao_walk_forward(criar, df, X[colunas], INICIOS, JANELA_MESES)
        linhas.append(
            {
                "modelo": nome,
                "AUC validação aleatória": aleatoria["auc"],
                "AUC no tempo (média dos blocos)": por_bloco["auc"].mean(),
                "pior bloco": por_bloco["auc"].min(),
                "melhor bloco": por_bloco["auc"].max(),
            }
        )
    return pd.DataFrame(linhas).set_index("modelo")


def teto_do_regime_novo(df: pd.DataFrame, inicio: str = "2009-06") -> float:
    """AUC do modelo completo em validação aleatória restrita ao regime de alta conversão.

    Serve de teto: dentro de um único regime o modelo não pode usar "quando" para acertar,
    então o número mostra quanto as variáveis explicam de fato sobre "quem".
    """
    novo = df[df["periodo"] >= pd.Period(inicio, freq="M")]
    tipo, colunas = CONCORRENTES[LIGHTGBM]
    X = montar_features(novo)[colunas]
    return avaliacao.validacao_aleatoria(partial(criar_modelo, tipo, colunas), X, novo[ALVO])["auc"]


def grade_de_configuracoes(df: pd.DataFrame) -> pd.DataFrame:
    """Todas as combinações testadas: 4 conjuntos de variáveis x 2 modelos x 3 janelas de treino.

    Está aqui por transparência. A configuração escolhida saiu desta grade, avaliada nos
    mesmos blocos usados para reportar o resultado, então a vantagem dela sobre as vizinhas
    está superestimada (maldição do vencedor).
    """
    X = montar_features(df)
    conjuntos = {**CONJUNTOS, "enxuto (6 variáveis)": ENXUTO}
    janelas = {"todo o passado": None, "12 meses": 12, "6 meses": 6}
    linhas = []
    for nome_conjunto, colunas in conjuntos.items():
        for tipo in TIPOS:
            for nome_janela, meses in janelas.items():
                por_bloco, _ = avaliacao.validacao_walk_forward(
                    partial(criar_modelo, tipo, colunas), df, X[colunas], INICIOS, meses
                )
                linhas.append(
                    {
                        "variáveis": nome_conjunto,
                        "modelo": tipo,
                        "janela de treino": nome_janela,
                        "AUC no tempo": por_bloco["auc"].mean(),
                        "pior bloco": por_bloco["auc"].min(),
                    }
                )
    return pd.DataFrame(linhas).sort_values("AUC no tempo", ascending=False).reset_index(drop=True)


def previsoes_no_tempo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Previsões fora da amostra de todos os concorrentes, mais a regra de bolso, alinhadas por lead."""
    X = montar_features(df)
    por_bloco = {}
    juntas = None
    for nome, (tipo, colunas) in CONCORRENTES.items():
        tabela, previsoes = avaliacao.validacao_walk_forward(
            partial(criar_modelo, tipo, colunas), df, X[colunas], INICIOS, JANELA_MESES
        )
        por_bloco[nome] = tabela
        if juntas is None:
            juntas = previsoes[["bloco", "y"]].copy()
        juntas[nome] = previsoes["p"]
    juntas[REGRA] = negocio.regra_de_bolso(df.loc[juntas.index])
    regra_blocos = [
        {"bloco": rotulo, **avaliacao.metricas(g["y"].to_numpy(), g[REGRA].to_numpy())}
        for rotulo, g in juntas.groupby("bloco", sort=False)
    ]
    por_bloco[REGRA] = pd.DataFrame(regra_blocos).set_index("bloco")
    return juntas, por_bloco


def comparar_concorrentes(juntas: pd.DataFrame, repeticoes: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUC média no tempo com intervalo de 95% por bootstrap, e diferenças pareadas contra o escolhido."""
    nomes = [ESCOLHIDO, LIGHTGBM, REGRA]
    reamostras = avaliacao.bootstrap_auc_media(
        juntas["bloco"].to_numpy(), juntas["y"].to_numpy(), {n: juntas[n].to_numpy() for n in nomes}, repeticoes
    )
    resumo = pd.DataFrame(
        {
            "AUC no tempo": [avaliacao.auc_media_por_bloco(juntas["bloco"], juntas["y"], juntas[n]) for n in nomes],
            "IC 95% inferior": reamostras.quantile(0.025).to_numpy(),
            "IC 95% superior": reamostras.quantile(0.975).to_numpy(),
        },
        index=nomes,
    )
    outros = [n for n in nomes if n != ESCOLHIDO]
    diferencas = pd.DataFrame(
        {
            "diferença de AUC": [resumo.loc[ESCOLHIDO, "AUC no tempo"] - resumo.loc[n, "AUC no tempo"] for n in outros],
            "IC 95% inferior": [(reamostras[ESCOLHIDO] - reamostras[n]).quantile(0.025) for n in outros],
            "IC 95% superior": [(reamostras[ESCOLHIDO] - reamostras[n]).quantile(0.975) for n in outros],
        },
        index=[f"{ESCOLHIDO} − {n}" for n in outros],
    )
    return resumo, diferencas


def calibracao_no_tempo(juntas: pd.DataFrame, por_bloco: pd.DataFrame) -> pd.DataFrame:
    """Conversões previstas x reais por bloco, com o modelo cru e com o nível corrigido.

    `por_bloco` é a tabela de métricas do modelo escolhido (traz taxa de treino e taxa recente).
    """
    linhas = []
    for rotulo, g in juntas.groupby("bloco", sort=False):
        info = por_bloco.loc[rotulo]
        cru = g[ESCOLHIDO].to_numpy()
        corrigido = avaliacao.corrigir_nivel(cru, info["taxa_treino"], info["taxa_recente"])
        reais = int(g["y"].sum())
        linhas.append(
            {
                "bloco": rotulo,
                "leads": len(g),
                "taxa no treino": info["taxa_treino"],
                "taxa recente": info["taxa_recente"],
                "taxa real": g["y"].mean(),
                "conversões reais": reais,
                "previstas (cru)": cru.sum(),
                "previstas (corrigido)": corrigido.sum(),
                "erro (cru)": cru.sum() / reais - 1,
                "erro (corrigido)": corrigido.sum() / reais - 1,
                "Brier (cru)": avaliacao.brier(g["y"], cru),
                "Brier (corrigido)": avaliacao.brier(g["y"], corrigido),
            }
        )
    return pd.DataFrame(linhas).set_index("bloco")


def ganho_com_capacidade(juntas: pd.DataFrame, fracoes: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Conversões obtidas quando a equipe só consegue ligar para parte dos leads de cada bloco."""
    colunas = {"regra de bolso": REGRA, "modelo": ESCOLHIDO}
    tabela = negocio.tabela_capacidade(juntas, colunas, list(fracoes))
    tabela["modelo vs sem priorizar"] = tabela["modelo"] / tabela["sem priorizar"] - 1
    tabela["modelo vs regra"] = tabela["modelo"] / tabela["regra de bolso"] - 1
    return tabela


def curva_de_ganho(juntas: pd.DataFrame, passos: int = 20) -> pd.DataFrame:
    """Fração das conversões capturada para cada fração de leads contatados (por bloco, somada)."""
    total = float(juntas["y"].sum())
    fracoes = np.linspace(0, 1, passos + 1)
    return pd.DataFrame(
        {
            "sem priorizar": fracoes,
            "regra de bolso": [negocio.conversoes_com_capacidade(juntas, REGRA, f) / total for f in fracoes],
            "modelo": [negocio.conversoes_com_capacidade(juntas, ESCOLHIDO, f) / total for f in fracoes],
        },
        index=pd.Index(fracoes, name="fração dos leads contatada"),
    )


def coeficientes_do_escolhido(df: pd.DataFrame) -> pd.DataFrame:
    """Razões de chances da logística treinada nos 12 meses mais recentes (leitura do modelo).

    Variáveis numéricas estão padronizadas: o coeficiente é o efeito de um desvio-padrão.
    """
    tipo, colunas = CONCORRENTES[ESCOLHIDO]
    recente = df[df["periodo"] > df["periodo"].max() - JANELA_MESES]
    modelo = criar_modelo(tipo, colunas).fit(montar_features(recente)[colunas], recente[ALVO])
    nomes = modelo.named_steps["pre"].get_feature_names_out()
    coeficientes = modelo.named_steps["modelo"].coef_[0]
    tabela = pd.DataFrame({"coeficiente": coeficientes, "razão de chances": np.exp(coeficientes)}, index=nomes)
    return tabela.sort_values("coeficiente", ascending=False)
