"""Grupos de variáveis e montagem da matriz de features."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Quem é o lead.
PERFIL = ["age", "job", "marital", "education", "default", "housing", "loan"]

# O que já sabemos do relacionamento com ele antes desta ligação.
CONTATO = ["contact", "campaign", "previous", "pdays", "contatado_antes", "poutcome"]

# Quando a ligação acontece.
CALENDARIO = ["month", "day_of_week"]

# Indicadores macroeconômicos do período. Variam pouco dentro de um mês, então funcionam
# como um "carimbo de data": ajudam muito em validação aleatória e pouco fora do período.
MACRO = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]

CATEGORICAS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "poutcome",
    "month",
    "day_of_week",
]

CONJUNTOS = {
    "perfil + contato": PERFIL + CONTATO,
    "perfil + contato + calendário": PERFIL + CONTATO + CALENDARIO,
    "tudo (inclui macro)": PERFIL + CONTATO + CALENDARIO + MACRO,
}

SEM_CONTATO_ANTERIOR = 999


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve as features prontas para os modelos.

    `pdays` usa 999 como código de "nunca foi contatado". Tratar 999 como número distorce
    qualquer modelo linear, então o código vira um indicador e o valor vira ausente.
    """
    X = df.copy()
    nunca = X["pdays"] == SEM_CONTATO_ANTERIOR
    X["contatado_antes"] = (~nunca).astype(int)
    X["pdays"] = X["pdays"].where(~nunca, np.nan).astype(float)
    return X[PERFIL + CONTATO + CALENDARIO + MACRO]
