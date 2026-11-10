"""Métricas de ranqueamento e calibração, esquemas de validação e correção de nível."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from lead_scoring.dados import ALVO


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Área sob a curva ROC pela estatística de Mann-Whitney (empates recebem posto médio).

    Leitura prática: sorteando um lead que converteu e um que não converteu, é a chance de
    o modelo ter dado score maior ao que converteu. 0,5 = cara ou coroa.
    """
    y = np.asarray(y)
    positivos = int(y.sum())
    negativos = len(y) - positivos
    if positivos == 0 or negativos == 0:
        raise ValueError("AUC indefinida: é preciso ter pelo menos um positivo e um negativo.")
    postos = rankdata(np.asarray(p))
    return float((postos[y == 1].sum() - positivos * (positivos + 1) / 2) / (positivos * negativos))


def conversoes_no_topo(y: np.ndarray, score: np.ndarray, n_topo: int) -> float:
    """Conversões esperadas ligando para os `n_topo` leads de maior score.

    Leads empatados na linha de corte são sorteados entre si; a função devolve o valor
    esperado desse sorteio. Sem isso, um score com muitos empates (uma regra com poucos
    níveis, por exemplo) seria avaliado pela ordem do arquivo, que não é aleatória.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if n_topo <= 0:
        return 0.0
    n_topo = min(n_topo, len(y))
    corte = np.sort(score)[::-1][n_topo - 1]
    acima = score > corte
    empatados = score == corte
    vagas = n_topo - int(acima.sum())
    return float(y[acima].sum() + y[empatados].mean() * vagas)


def captura_no_topo(y: np.ndarray, score: np.ndarray, fracao: float) -> float:
    """Fração das conversões capturada ligando só para a `fracao` de leads com maior score."""
    y = np.asarray(y)
    n_topo = max(1, int(round(len(y) * fracao)))
    return conversoes_no_topo(y, score, n_topo) / float(y.sum())


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Resumo de desempenho de um vetor de scores ou probabilidades previstas."""
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "taxa_base": float(y.mean()),
        "auc": auc(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "lift_top10": captura_no_topo(y, p, 0.10) / 0.10,
        "captura_top30": captura_no_topo(y, p, 0.30),
    }


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Erro quadrático médio das probabilidades (menor é melhor; exige p entre 0 e 1)."""
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def validacao_aleatoria(
    criar: Callable[[], Pipeline], X: pd.DataFrame, y: pd.Series, dobras: int = 5, semente: int = 42
) -> dict[str, float]:
    """K-fold estratificado com embaralhamento: o esquema 'padrão', que ignora o tempo."""
    previsoes = np.zeros(len(y))
    divisor = StratifiedKFold(n_splits=dobras, shuffle=True, random_state=semente)
    for treino, teste in divisor.split(X, y):
        modelo = criar().fit(X.iloc[treino], y.iloc[treino])
        previsoes[teste] = modelo.predict_proba(X.iloc[teste])[:, 1]
    return metricas(y.to_numpy(), previsoes)


def blocos_walk_forward(
    periodos: pd.Series, inicios: list[str], janela_meses: int | None = None
) -> list[tuple[str, pd.Series, pd.Series]]:
    """Gera trios (rótulo, treino, teste) em que o treino é sempre anterior ao bloco de teste.

    `inicios` lista o primeiro período de cada bloco de teste; cada bloco vai até o início
    do seguinte (o último vai até o fim dos dados). Com `janela_meses`, o treino usa apenas
    os meses imediatamente anteriores ao bloco (janela deslizante); sem ela, todo o passado.
    """
    cortes = [pd.Period(i, freq="M") for i in inicios]
    trios = []
    for i, inicio in enumerate(cortes):
        teste = periodos >= inicio
        if i + 1 < len(cortes):
            teste &= periodos < cortes[i + 1]
        treino = periodos < inicio
        if janela_meses is not None:
            treino &= periodos >= inicio - janela_meses
        rotulo = f"{periodos[teste].min()} a {periodos[teste].max()}"
        trios.append((rotulo, treino, teste))
    return trios


def validacao_walk_forward(
    criar: Callable[[], Pipeline],
    df: pd.DataFrame,
    X: pd.DataFrame,
    inicios: list[str],
    janela_meses: int | None = None,
    leads_recentes: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina no passado e testa no bloco seguinte, como o modelo seria usado de verdade.

    Devolve (métricas por bloco, previsões fora da amostra com colunas bloco / y / p).
    `taxa_recente` é a conversão dos últimos `leads_recentes` leads anteriores ao bloco:
    a informação mais fresca que existiria no momento de pontuar.
    """
    y = df[ALVO]
    linhas, partes = [], []
    for rotulo, treino, teste in blocos_walk_forward(df["periodo"], inicios, janela_meses):
        modelo = criar().fit(X[treino], y[treino])
        p = modelo.predict_proba(X[teste])[:, 1]
        partes.append(pd.DataFrame({"bloco": rotulo, "y": y[teste], "p": p}, index=df.index[teste]))
        anteriores = y[df["periodo"] < df.loc[teste, "periodo"].min()]
        linha = metricas(y[teste].to_numpy(), p)
        linha.update(
            bloco=rotulo,
            n_treino=int(treino.sum()),
            taxa_treino=float(y[treino].mean()),
            taxa_recente=float(anteriores.tail(leads_recentes).mean()),
            prob_media_prevista=float(p.mean()),
        )
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("bloco"), pd.concat(partes)


def auc_media_por_bloco(blocos: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Média simples das AUCs calculadas dentro de cada bloco.

    Juntar todos os blocos numa AUC só infla o número quando a taxa de conversão muda de
    bloco para bloco: o modelo ganha crédito por acertar o período, não o lead.
    """
    blocos, y, p = np.asarray(blocos), np.asarray(y), np.asarray(p)
    return float(np.mean([auc(y[blocos == b], p[blocos == b]) for b in pd.unique(blocos)]))


def bootstrap_auc_media(
    blocos: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray], repeticoes: int = 1000, semente: int = 42
) -> pd.DataFrame:
    """Reamostra os leads dentro de cada bloco e recalcula a AUC média de cada concorrente.

    A mesma reamostra é usada para todos os concorrentes, o que permite montar o intervalo
    da *diferença* entre eles (comparação pareada). Devolve uma linha por repetição.
    """
    blocos, y = np.asarray(blocos), np.asarray(y)
    gerador = np.random.default_rng(semente)
    posicoes = [np.flatnonzero(blocos == b) for b in pd.unique(blocos)]
    resultado = {nome: np.zeros(repeticoes) for nome in scores}
    for r in range(repeticoes):
        amostras = [gerador.choice(pos, size=len(pos), replace=True) for pos in posicoes]
        for nome, s in scores.items():
            resultado[nome][r] = np.mean([auc(y[a], np.asarray(s)[a]) for a in amostras])
    return pd.DataFrame(resultado)


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def corrigir_nivel(p: np.ndarray, taxa_treino: float, taxa_recente: float) -> np.ndarray:
    """Desloca as probabilidades para o nível de conversão observado recentemente.

    Correção clássica de mudança de taxa-base: soma ao logit a diferença entre a taxa
    recente e a taxa em que o modelo foi treinado. É monótona, então não altera a ordem dos
    leads; só muda o nível das probabilidades. Usa apenas informação anterior ao bloco.
    """
    deslocamento = _logit(taxa_recente) - _logit(taxa_treino)
    return 1.0 / (1.0 + np.exp(-(_logit(p) + deslocamento)))
