"""Gráficos adicionais para o site, como dados, para ele desenhar nativamente (`para_o_site.graficos_extras`).

O texto e o gráfico principal deste estudo (a curva de ganho) vêm de outro arquivo do site; aqui saem só os
extras, cada um equivalente a uma figura ou tabela do README. Todo valor sai do dicionário de resultados gravado
por `scripts/rodar_analise.py`; nada é digitado à mão.
"""

from __future__ import annotations

IDIOMAS = ("pt", "en", "es")
LGBM = "LightGBM, todas as variáveis"
LOGISTICA = "Logística, 6 variáveis"
REGRA = "Regra de bolso (sem modelo)"
NOMES = {
    LGBM: ("LightGBM, todas as variáveis", "LightGBM, all features", "LightGBM, todas las variables"),
    LOGISTICA: ("Logística, 6 variáveis", "Logistic, 6 features", "Logística, 6 variables"),
    REGRA: ("Regra de bolso (sem modelo)", "Rule of thumb (no model)", "Regla práctica (sin modelo)"),
}
ESTRATEGIAS = {
    "sem priorizar": (("Sem priorizar", "No prioritization", "Sin priorizar"), "base"),
    "regra de bolso": (("Regra de bolso", "Rule of thumb", "Regla práctica"), "alternativa"),
    "modelo": (("Modelo", "Model", "Modelo"), "principal"),
}


def _n(valor: float, casas: int) -> dict[str, str]:
    """O mesmo número nos três idiomas (pt e es com vírgula decimal; sinal de menos tipográfico)."""
    texto = f"{valor:.{casas}f}".replace("-", "−")
    return {"pt": texto.replace(".", ","), "en": texto, "es": texto.replace(".", ",")}


def _p(valor: float) -> dict[str, str]:
    return {i: f"{t}%" for i, t in _n(valor * 100, 0).items()}


def _milhar(valor: float) -> dict[str, str]:
    us = f"{valor:,.0f}"
    return {"pt": us.replace(",", "."), "en": us, "es": us.replace(",", ".")}


def _nome(chave: str) -> dict[str, str]:
    return dict(zip(IDIOMAS, NOMES[chave]))


def _bloco(rotulo: str) -> dict[str, str]:
    """'2009-04 a 2009-05' nos três idiomas."""
    return {"pt": rotulo, "en": rotulo.replace(" a ", " to "), "es": rotulo}


def _rotulos_de_blocos(blocos: list[str]) -> dict[str, list[str]]:
    return {i: [_bloco(b)[i] for b in blocos] for i in IDIOMAS}


def promessa_x_realidade(r: dict) -> dict:
    """Barras: AUC em validação aleatória contra AUC no tempo, para os dois modelos."""
    tabela = r["aleatoria_x_temporal"]
    modelos = [m for m in (LGBM, LOGISTICA) if m in tabela]
    rotulos = {i: [_nome(m)[i] for m in modelos] for i in IDIOMAS}
    aleatoria, tempo = _n(tabela[LGBM]["AUC validação aleatória"], 2), _n(tabela[LGBM]["AUC no tempo (média dos blocos)"], 2)
    teto = _n(r["teto_regime_novo_auc"], 2)

    def serie(coluna: str) -> list[list]:
        return [[rotulos["pt"][k], round(tabela[m][coluna], 4)] for k, m in enumerate(modelos)]

    return {
        "tipo": "barras",
        "titulo": {
            "pt": f"O mesmo modelo marca AUC {aleatoria['pt']} na validação aleatória e {tempo['pt']} no tempo",
            "en": f"The same model scores AUC {aleatoria['en']} on random validation and {tempo['en']} over time",
            "es": f"El mismo modelo marca AUC {aleatoria['es']} en validación aleatoria y {tempo['es']} en el tiempo",
        },
        "leitura": {
            "pt": f"Os indicadores macroeconômicos funcionam como carimbo de data: com as linhas embaralhadas, o modelo reconhece o período e parece ótimo. Treinado no passado e testado no futuro, ele cai. O teto no regime recente é {teto['pt']}.",
            "en": f"The macroeconomic indicators act as a timestamp: with shuffled rows, the model recognizes the period and looks great. Trained on the past and tested on the future, it drops. The ceiling within the recent regime is {teto['en']}.",
            "es": f"Los indicadores macroeconómicos funcionan como sello de fecha: con las filas mezcladas, el modelo reconoce el período y parece excelente. Entrenado en el pasado y probado en el futuro, cae. El techo en el régimen reciente es {teto['es']}.",
        },
        "eixo_x": {"pt": "Modelo", "en": "Model", "es": "Modelo"},
        "eixo_y": {"pt": "AUC", "en": "AUC", "es": "AUC"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            {
                "nome": {"pt": "Validação aleatória", "en": "Random validation", "es": "Validación aleatoria"},
                "papel": "base",
                "pontos": serie("AUC validação aleatória"),
                "rotulos": rotulos,
            },
            {
                "nome": {"pt": "No tempo (média dos blocos)", "en": "Over time (mean of blocks)", "es": "En el tiempo (media de los bloques)"},
                "papel": "principal",
                "pontos": serie("AUC no tempo (média dos blocos)"),
                "rotulos": rotulos,
            },
        ],
    }


def auc_por_bloco(r: dict) -> dict:
    """Barras: AUC de cada concorrente em cada bloco de teste do walk-forward."""
    por_bloco = r["auc_por_bloco"]
    blocos = list(por_bloco[LOGISTICA])
    rotulos = _rotulos_de_blocos(blocos)
    c = r["concorrentes"]
    log, lgb, regra = (_n(c[m]["AUC no tempo"], 2) for m in (LOGISTICA, LGBM, REGRA))
    d = r["diferencas"][f"{LOGISTICA} − {REGRA}"]
    dif, inf, sup = _n(d["diferença de AUC"], 2), _n(d["IC 95% inferior"], 3), _n(d["IC 95% superior"], 3)
    papeis = {LOGISTICA: "principal", LGBM: "alternativa", REGRA: "base"}
    return {
        "tipo": "barras",
        "titulo": {
            "pt": f"Simples não perde para complexo: {log['pt']} contra {lgb['pt']}; a regra de bolso fica em {regra['pt']}",
            "en": f"Simple does not lose to complex: {log['en']} vs {lgb['en']}; the rule of thumb gets {regra['en']}",
            "es": f"Lo simple no pierde ante lo complejo: {log['es']} frente a {lgb['es']}; la regla práctica, {regra['es']}",
        },
        "leitura": {
            "pt": f"AUC em cada bloco de teste, pontuado por um modelo treinado só nos 12 meses anteriores. A logística supera a regra de bolso em {dif['pt']} de AUC (IC 95% de {inf['pt']} a {sup['pt']}); a leitura segura é que o modelo complexo não é melhor.",
            "en": f"AUC in each test block, scored by a model trained only on the previous 12 months. The logistic model beats the rule of thumb by {dif['en']} AUC (95% CI {inf['en']} to {sup['en']}); the safe reading is that the complex model is not better.",
            "es": f"AUC en cada bloque de prueba, puntuado por un modelo entrenado solo con los 12 meses anteriores. La logística supera a la regla práctica en {dif['es']} de AUC (IC 95% de {inf['es']} a {sup['es']}); la lectura segura es que el modelo complejo no es mejor.",
        },
        "eixo_x": {"pt": "Bloco de teste", "en": "Test block", "es": "Bloque de prueba"},
        "eixo_y": {"pt": "AUC", "en": "AUC", "es": "AUC"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            {
                "nome": _nome(m),
                "papel": papeis[m],
                "pontos": [[rotulos["pt"][k], round(por_bloco[m][b], 4)] for k, b in enumerate(blocos)],
                "rotulos": rotulos,
            }
            for m in (LOGISTICA, LGBM, REGRA)
            if m in por_bloco
        ],
    }


def capacidade(r: dict) -> dict:
    """Barras: conversões captadas por estratégia conforme a fração dos leads que a equipe consegue contatar."""
    cap = r["capacidade"]
    fracoes = sorted(cap, key=float)
    rotulos_fracao = [_p(float(f)) for f in fracoes]
    rotulos = {
        "pt": [f"{t['pt']} dos leads" for t in rotulos_fracao],
        "en": [f"{t['en']} of leads" for t in rotulos_fracao],
        "es": [f"{t['es']} de los leads" for t in rotulos_fracao],
    }
    ref = "0.3" if "0.3" in cap else fracoes[len(fracoes) // 2]
    fr, vs_sem, vs_regra = _p(float(ref)), _p(cap[ref]["modelo vs sem priorizar"]), _p(cap[ref]["modelo vs regra"])
    leads = _milhar(sum(b["leads"] for b in r["calibracao"].values()))
    return {
        "tipo": "barras",
        "titulo": {
            "pt": f"Contatando {fr['pt']} dos leads, o modelo traz {vs_sem['pt']} mais conversões que sem priorizar",
            "en": f"Reaching {fr['en']} of leads, the model brings {vs_sem['en']} more conversions than no prioritization",
            "es": f"Contactando el {fr['es']} de los leads, el modelo logra {vs_sem['es']} más conversiones que sin priorizar",
        },
        "leitura": {
            "pt": f"Conversões somadas nos cinco blocos de teste ({leads['pt']} leads). Contra a regra de bolso, o ganho é de {vs_regra['pt']}: moderado, coerente com a AUC, e vale porque ordenar a fila custa quase nada.",
            "en": f"Conversions summed over the five test blocks ({leads['en']} leads). Against the rule of thumb the gain is {vs_regra['en']}: moderate, in line with the AUC, and worth it because ordering the queue costs almost nothing.",
            "es": f"Conversiones sumadas en los cinco bloques de prueba ({leads['es']} leads). Frente a la regla práctica, la ganancia es del {vs_regra['es']}: moderada, coherente con el AUC, y vale porque ordenar la cola casi no cuesta.",
        },
        "eixo_x": {"pt": "Capacidade da equipe", "en": "Team capacity", "es": "Capacidad del equipo"},
        "eixo_y": {"pt": "Conversões captadas", "en": "Conversions captured", "es": "Conversiones captadas"},
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            {
                "nome": dict(zip(IDIOMAS, nomes)),
                "papel": papel,
                "pontos": [[rotulos["pt"][k], round(cap[f][chave], 1)] for k, f in enumerate(fracoes)],
                "rotulos": rotulos,
            }
            for chave, (nomes, papel) in ESTRATEGIAS.items()
        ],
    }


def calibracao(r: dict) -> dict:
    """Barras: erro do total previsto de conversões por bloco, com o modelo cru e com o nível corrigido."""
    cal = r["calibracao"]
    blocos = list(cal)
    rotulos = _rotulos_de_blocos(blocos)
    pior = max(blocos, key=lambda b: abs(cal[b]["erro (corrigido)"]))
    erro = cal[pior]["erro (corrigido)"]
    tamanho = _p(abs(erro))
    lado = ("abaixo", "below", "por debajo") if erro < 0 else ("acima", "above", "por encima")
    cru = _p(max(abs(cal[b]["erro (cru)"]) for b in blocos))
    ok = sum(abs(cal[b]["erro (corrigido)"]) <= 0.1 for b in blocos)
    pb = _bloco(pior)

    def serie(coluna: str) -> list[list]:
        return [[rotulos["pt"][k], round(cal[b][coluna], 4)] for k, b in enumerate(blocos)]

    return {
        "tipo": "barras",
        "titulo": {
            "pt": f"O score ordena, mas o volume erra: mesmo corrigido, o total fica {tamanho['pt']} {lado[0]} do real",
            "en": f"The score ranks, but volume misses: even corrected, the total is {tamanho['en']} {lado[1]} actual",
            "es": f"El score ordena, pero el volumen falla: aun corregido, queda {tamanho['es']} {lado[2]} del real",
        },
        "leitura": {
            "pt": f"Erro do total previsto de conversões em cada bloco. Treinado em meses de conversão baixa, o modelo cru erra até {cru['pt']}; deslocar as probabilidades para a taxa recente deixa {ok} de {len(blocos)} blocos com erro de até 10%, mas não o de {pb['pt']}.",
            "en": f"Error in the forecast total of conversions per block. Trained on low-conversion months, the raw model misses by up to {cru['en']}; shifting probabilities to the recent rate brings {ok} of {len(blocos)} blocks within 10%, but not {pb['en']}.",
            "es": f"Error del total previsto de conversiones en cada bloque. Entrenado en meses de conversión baja, el modelo crudo falla hasta un {cru['es']}; desplazar las probabilidades a la tasa reciente deja {ok} de {len(blocos)} bloques con error de hasta 10%, pero no el de {pb['es']}.",
        },
        "eixo_x": {"pt": "Bloco de teste", "en": "Test block", "es": "Bloque de prueba"},
        "eixo_y": {"pt": "Erro do total previsto", "en": "Error in the forecast total", "es": "Error del total previsto"},
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [
            {"nome": {"pt": "Modelo cru", "en": "Raw model", "es": "Modelo crudo"}, "papel": "base", "pontos": serie("erro (cru)"), "rotulos": rotulos},
            {"nome": {"pt": "Nível corrigido", "en": "Level corrected", "es": "Nivel corregido"}, "papel": "principal", "pontos": serie("erro (corrigido)"), "rotulos": rotulos},
        ],
    }


def graficos_extras(r: dict) -> list[dict]:
    """As quatro especificações, do achado mais importante para o menos."""
    return [promessa_x_realidade(r), auc_por_bloco(r), capacidade(r), calibracao(r)]
