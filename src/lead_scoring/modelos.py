"""Modelos de propensão: regressão logística (baseline interpretável) e LightGBM."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_scoring.features import CATEGORICAS

SEMENTE = 42
TIPOS = ("logistica", "lightgbm")


def criar_modelo(tipo: str, colunas: list[str]) -> Pipeline:
    """Cria o pipeline completo (pré-processamento + estimador) para as colunas dadas.

    Nenhum dos modelos usa peso de classe: reponderar melhora o recall mas destrói a
    calibração das probabilidades, e aqui a probabilidade é o produto (ela decide quem ligar).
    """
    categoricas = [c for c in colunas if c in CATEGORICAS]
    numericas = [c for c in colunas if c not in CATEGORICAS]
    # Variável com duas categorias vira uma coluna só: o coeficiente da logística passa a ser
    # lido direto como "uma categoria contra a outra".
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")

    if tipo == "logistica":
        numerico = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        estimador = LogisticRegression(C=0.5, max_iter=2000)
    elif tipo == "lightgbm":
        numerico = "passthrough"  # árvores lidam com ausentes e não precisam de escala
        estimador = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=SEMENTE,
            n_jobs=1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {tipo!r}. Use um de {TIPOS}.")

    pre = ColumnTransformer(
        [("cat", one_hot, categoricas), ("num", numerico, numericas)],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline([("pre", pre), ("modelo", estimador)])
    pipeline.set_output(transform="pandas")
    return pipeline
