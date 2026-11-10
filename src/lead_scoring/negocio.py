"""Tradução do score para decisão: regra de bolso concorrente e ganho sob capacidade limitada."""

from __future__ import annotations

import pandas as pd

from lead_scoring.avaliacao import conversoes_no_topo


def regra_de_bolso(df: pd.DataFrame) -> pd.Series:
    """Prioridade que um gestor montaria sem modelo nenhum, só com filtros de CRM.

    1º quem já comprou numa campanha anterior; 2º quem já foi contatado antes e não comprou;
    3º o restante. Em cada nível, celular na frente de telefone fixo.
    """
    score = 2.0 * (df["poutcome"] == "success") + 1.0 * (df["poutcome"] == "failure")
    return score + 0.5 * (df["contact"] == "cellular")


def conversoes_com_capacidade(previsoes: pd.DataFrame, coluna_score: str, fracao: float) -> float:
    """Conversões obtidas ligando, em cada bloco, só para a `fracao` de leads com maior score.

    A capacidade é aplicada bloco a bloco porque a equipe tem um teto de ligações por período,
    não um teto único para o histórico inteiro.
    """
    total = 0.0
    for _, bloco in previsoes.groupby("bloco", sort=False):
        n_ligacoes = int(round(len(bloco) * fracao))
        total += conversoes_no_topo(bloco["y"].to_numpy(), bloco[coluna_score].to_numpy(), n_ligacoes)
    return total


def tabela_capacidade(previsoes: pd.DataFrame, colunas_score: dict[str, str], fracoes: list[float]) -> pd.DataFrame:
    """Compara concorrentes em vários níveis de capacidade da equipe.

    A coluna "sem priorizar" é o valor esperado de ligar em ordem aleatória: a mesma fração
    das conversões de cada bloco.
    """
    blocos = [b for _, b in previsoes.groupby("bloco", sort=False)]
    linhas = []
    for fracao in fracoes:
        ligacoes = [int(round(len(b) * fracao)) for b in blocos]
        linha = {
            "capacidade": fracao,
            "ligações": sum(ligacoes),
            "sem priorizar": float(sum(b["y"].mean() * n for b, n in zip(blocos, ligacoes))),
        }
        for nome, coluna in colunas_score.items():
            linha[nome] = conversoes_com_capacidade(previsoes, coluna, fracao)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("capacidade")
