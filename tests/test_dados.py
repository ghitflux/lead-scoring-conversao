import pandas as pd
import pytest

from lead_scoring.dados import ALVO, preparar, reconstruir_periodo


def test_reconstruir_periodo_vira_o_ano_quando_o_mes_volta():
    meses = pd.Series(["may", "jun", "nov", "dec", "mar", "apr", "nov", "mar"])
    periodos = reconstruir_periodo(meses, ano_inicial=2008)
    assert [str(p) for p in periodos] == [
        "2008-05",
        "2008-06",
        "2008-11",
        "2008-12",
        "2009-03",
        "2009-04",
        "2009-11",
        "2010-03",
    ]


def test_reconstruir_periodo_rejeita_mes_desconhecido():
    with pytest.raises(ValueError):
        reconstruir_periodo(pd.Series(["may", "maio"]))


def test_preparar_remove_vazamento_e_cria_alvo():
    bruto = pd.DataFrame(
        {
            "age": [30, 40],
            "month": ["may", "jun"],
            "duration": [100, 900],
            "emp.var.rate": [1.1, 1.1],
            "y": ["no", "yes"],
        }
    )
    df = preparar(bruto)
    assert "duration" not in df.columns
    assert "y" not in df.columns
    assert "emp_var_rate" in df.columns
    assert df[ALVO].tolist() == [0, 1]
