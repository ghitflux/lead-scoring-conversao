import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from lead_scoring import avaliacao


def test_auc_bate_com_sklearn_inclusive_com_empates():
    gerador = np.random.default_rng(0)
    y = gerador.integers(0, 2, 500)
    p = np.round(gerador.random(500), 1)  # arredondar cria muitos empates
    assert avaliacao.auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auc_exige_as_duas_classes():
    with pytest.raises(ValueError):
        avaliacao.auc(np.ones(5), np.arange(5))


def test_conversoes_no_topo_sem_empates():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert avaliacao.conversoes_no_topo(y, score, 3) == 2.0


def test_conversoes_no_topo_usa_valor_esperado_nos_empates():
    # Quatro leads empatados, dois convertem. Ligando para dois deles, espera-se uma conversão,
    # qualquer que seja a ordem em que aparecem no arquivo.
    score = np.ones(4)
    assert avaliacao.conversoes_no_topo(np.array([1, 1, 0, 0]), score, 2) == pytest.approx(1.0)
    assert avaliacao.conversoes_no_topo(np.array([0, 0, 1, 1]), score, 2) == pytest.approx(1.0)


def test_captura_no_topo_de_score_sem_informacao_e_a_propria_fracao():
    y = np.array([1, 0] * 50)
    assert avaliacao.captura_no_topo(y, np.zeros(100), 0.30) == pytest.approx(0.30)


def test_blocos_walk_forward_nunca_treinam_no_futuro():
    periodos = pd.Series(pd.period_range("2008-01", "2009-12", freq="M"))
    trios = avaliacao.blocos_walk_forward(periodos, ["2009-01", "2009-07"])
    assert len(trios) == 2
    for _, treino, teste in trios:
        assert not (treino & teste).any()
        assert periodos[treino].max() < periodos[teste].min()
    # Os blocos de teste não se sobrepõem e cobrem tudo a partir do primeiro início.
    assert (trios[0][2] ^ trios[1][2]).sum() == 12


def test_blocos_walk_forward_respeitam_a_janela():
    periodos = pd.Series(pd.period_range("2008-01", "2009-12", freq="M"))
    _, treino, _ = avaliacao.blocos_walk_forward(periodos, ["2009-07"], janela_meses=6)[0]
    assert [str(p) for p in periodos[treino]] == ["2009-01", "2009-02", "2009-03", "2009-04", "2009-05", "2009-06"]


def test_corrigir_nivel_leva_o_lead_medio_exatamente_para_a_taxa_recente():
    # Um lead com probabilidade igual à taxa do treino passa a ter a taxa recente.
    assert avaliacao.corrigir_nivel(np.array([0.06]), taxa_treino=0.06, taxa_recente=0.40) == pytest.approx([0.40])


def test_corrigir_nivel_aproxima_a_media_da_taxa_recente_e_preserva_a_ordem():
    gerador = np.random.default_rng(1)
    p = gerador.beta(1, 15, 5000)  # modelo treinado num regime de conversão baixa
    corrigido = avaliacao.corrigir_nivel(p, taxa_treino=p.mean(), taxa_recente=0.40)
    # O deslocamento é no logit, então a média não cai exatamente em 0,40: ela se aproxima.
    assert p.mean() < corrigido.mean() <= 0.40
    assert abs(corrigido.mean() - 0.40) < abs(p.mean() - 0.40) / 3
    assert (np.argsort(p) == np.argsort(corrigido)).all()


def test_corrigir_nivel_sem_mudanca_de_taxa_nao_altera_nada():
    p = np.array([0.05, 0.2, 0.7])
    assert avaliacao.corrigir_nivel(p, 0.1, 0.1) == pytest.approx(p)


def test_auc_media_por_bloco_nao_premia_acertar_o_periodo():
    # Dentro de cada bloco o score é ruído; entre blocos, ele acompanha a taxa de conversão.
    gerador = np.random.default_rng(2)
    blocos = np.repeat(["a", "b"], 2000)
    y = np.concatenate([gerador.random(2000) < 0.05, gerador.random(2000) < 0.50]).astype(int)
    score = np.concatenate([gerador.random(2000), 1 + gerador.random(2000)])
    assert avaliacao.auc(y, score) > 0.70  # a AUC agregada parece boa...
    assert avaliacao.auc_media_por_bloco(blocos, y, score) == pytest.approx(0.5, abs=0.05)  # ...e não é
