"""Os gráficos extras do site: forma, limites, três idiomas e valores vindos dos resultados gravados."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from lead_scoring import site

RESULTADOS = Path(__file__).resolve().parents[1] / "reports" / "resultados.json"
IDIOMAS = {"pt", "en", "es"}
PAPEIS = {"principal", "base", "alternativa", "neutro"}
FORMATOS = {"numero", "percentual", "data"}


@pytest.fixture(scope="module")
def resultados() -> dict:
    if not RESULTADOS.exists():
        pytest.skip("reports/resultados.json ainda não foi gerado")
    return json.loads(RESULTADOS.read_text(encoding="utf-8"))


def test_graficos_extras_tem_a_forma_que_o_site_espera(resultados):
    extras = site.graficos_extras(resultados)
    assert 3 <= len(extras) <= 5
    for g in extras:
        assert g["tipo"] in {"linhas", "barras", "pontos"}
        for campo in ("titulo", "leitura", "eixo_x", "eixo_y"):
            assert set(g[campo]) == IDIOMAS
            assert all(isinstance(v, str) and v.strip() for v in g[campo].values())
        assert all(len(t) <= 90 for t in g["titulo"].values()), g["titulo"]
        assert g["formato_x"] in FORMATOS and g["formato_y"] in FORMATOS
        assert 1 <= len(g["series"]) <= 4
        for serie in g["series"]:
            assert set(serie["nome"]) == IDIOMAS
            assert serie["papel"] in PAPEIS
            limite = 300 if g["tipo"] == "pontos" else 80
            assert 1 <= len(serie["pontos"]) <= limite
            for x, y in serie["pontos"]:
                assert isinstance(y, (int, float)) and math.isfinite(y)
                assert isinstance(x, str) if g["tipo"] == "barras" else isinstance(x, (int, float))
            if g["tipo"] == "barras":
                assert set(serie["rotulos"]) == IDIOMAS
                assert all(len(v) == len(serie["pontos"]) for v in serie["rotulos"].values())
                assert [p[0] for p in serie["pontos"]] == serie["rotulos"]["pt"]
        assert "faixa" not in g or g["tipo"] == "linhas"


def test_o_json_gravado_traz_os_extras_e_so_eles_no_bloco_do_site(resultados):
    assert set(resultados["para_o_site"]) == {"graficos_extras"}
    sem_bloco = {k: v for k, v in resultados.items() if k != "para_o_site"}
    assert resultados["para_o_site"]["graficos_extras"] == site.graficos_extras(sem_bloco)


def test_os_valores_dos_extras_saem_dos_resultados(resultados):
    promessa, por_bloco, capacidade, calibracao = site.graficos_extras(resultados)
    aleatoria = resultados["aleatoria_x_temporal"][site.LGBM]["AUC validação aleatória"]
    assert f"{aleatoria:.2f}".replace(".", ",") in promessa["titulo"]["pt"]
    assert f"{aleatoria:.2f}" in promessa["titulo"]["en"]
    assert len(por_bloco["series"]) == 3
    assert all(len(s["pontos"]) == len(resultados["auc_por_bloco"][site.LOGISTICA]) for s in por_bloco["series"])
    modelo = dict(capacidade["series"][2]["pontos"])
    assert modelo["30% dos leads"] == round(resultados["capacidade"]["0.3"]["modelo"], 1)
    erros = [p[1] for p in calibracao["series"][1]["pontos"]]
    assert erros == [round(b["erro (corrigido)"], 4) for b in resultados["calibracao"].values()]

    alterado = copy.deepcopy(resultados)
    for bloco in alterado["calibracao"].values():
        bloco["erro (corrigido)"] = 0.2
    assert "20% acima" in site.calibracao(alterado)["titulo"]["pt"]


def test_numeros_nos_tres_idiomas():
    assert site._n(-0.0626, 2) == {"pt": "−0,06", "en": "−0.06", "es": "−0,06"}
    assert site._p(0.4280) == {"pt": "43%", "en": "43%", "es": "43%"}
    assert site._milhar(13216) == {"pt": "13.216", "en": "13,216", "es": "13.216"}
    assert site._bloco("2009-04 a 2009-05")["en"] == "2009-04 to 2009-05"
