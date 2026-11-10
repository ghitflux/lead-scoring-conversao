import numpy as np
import pandas as pd
import pytest

from lead_scoring import negocio
from lead_scoring.features import CONJUNTOS, SEM_CONTATO_ANTERIOR, montar_features
from lead_scoring.modelos import TIPOS, criar_modelo


def _leads_sinteticos(n: int = 400, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    return pd.DataFrame(
        {
            "age": gerador.integers(18, 80, n),
            "job": gerador.choice(["admin.", "technician", "retired"], n),
            "marital": gerador.choice(["married", "single"], n),
            "education": gerador.choice(["basic.9y", "university.degree"], n),
            "default": gerador.choice(["no", "unknown"], n),
            "housing": gerador.choice(["yes", "no"], n),
            "loan": gerador.choice(["yes", "no"], n),
            "contact": gerador.choice(["cellular", "telephone"], n),
            "month": gerador.choice(["may", "jun", "jul"], n),
            "day_of_week": gerador.choice(["mon", "tue", "wed"], n),
            "campaign": gerador.integers(1, 6, n),
            "pdays": gerador.choice([SEM_CONTATO_ANTERIOR, 3, 6], n, p=[0.8, 0.1, 0.1]),
            "previous": gerador.integers(0, 3, n),
            "poutcome": gerador.choice(["nonexistent", "failure", "success"], n, p=[0.8, 0.1, 0.1]),
            "emp_var_rate": gerador.normal(0, 1, n),
            "cons_price_idx": gerador.normal(93, 1, n),
            "cons_conf_idx": gerador.normal(-40, 4, n),
            "euribor3m": gerador.normal(3, 1, n),
            "nr_employed": gerador.normal(5100, 70, n),
        }
    )


def test_montar_features_trata_o_codigo_999_como_ausente():
    X = montar_features(_leads_sinteticos())
    nunca = X["contatado_antes"] == 0
    assert X.loc[nunca, "pdays"].isna().all()
    assert X.loc[~nunca, "pdays"].notna().all()
    assert X["pdays"].max() < SEM_CONTATO_ANTERIOR


@pytest.mark.parametrize("tipo", TIPOS)
def test_modelos_treinam_e_devolvem_probabilidades(tipo):
    leads = _leads_sinteticos()
    X = montar_features(leads)
    y = pd.Series((leads["poutcome"] == "success").astype(int))
    colunas = CONJUNTOS["tudo (inclui macro)"]
    p = criar_modelo(tipo, colunas).fit(X[colunas], y).predict_proba(X[colunas])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()


def test_modelo_aceita_categoria_que_nao_estava_no_treino():
    treino = _leads_sinteticos(semente=1)
    teste = _leads_sinteticos(50, semente=2).assign(job="astronauta")
    colunas = CONJUNTOS["perfil + contato"]
    y = pd.Series((treino["poutcome"] == "success").astype(int))
    modelo = criar_modelo("logistica", colunas).fit(montar_features(treino)[colunas], y)
    with pytest.warns(UserWarning, match="unknown categories"):
        p = modelo.predict_proba(montar_features(teste)[colunas])[:, 1]
    assert np.isfinite(p).all()


def test_tipo_de_modelo_desconhecido_gera_erro():
    with pytest.raises(ValueError):
        criar_modelo("rede_neural", ["age"])


def test_regra_de_bolso_ordena_como_descrito():
    leads = pd.DataFrame(
        {
            "poutcome": ["nonexistent", "nonexistent", "failure", "success"],
            "contact": ["telephone", "cellular", "telephone", "telephone"],
        }
    )
    assert negocio.regra_de_bolso(leads).rank().tolist() == [1, 2, 3, 4]


def test_tabela_capacidade_score_perfeito_e_score_sem_informacao():
    y = np.array([1] * 10 + [0] * 90)
    previsoes = pd.DataFrame({"bloco": "único", "y": y, "perfeito": y.astype(float), "nulo": 0.0})
    tabela = negocio.tabela_capacidade(previsoes, {"perfeito": "perfeito", "nulo": "nulo"}, [0.10, 0.50])
    assert tabela.loc[0.10, "ligações"] == 10
    assert tabela.loc[0.10, "perfeito"] == pytest.approx(10)
    assert tabela.loc[0.10, "nulo"] == pytest.approx(tabela.loc[0.10, "sem priorizar"]) == pytest.approx(1.0)
    assert tabela.loc[0.50, "perfeito"] == pytest.approx(10)
