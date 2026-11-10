"""Figuras do projeto. Um estilo só: marcas finas, grade discreta, rótulos diretos e seletivos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from lead_scoring.analise import ESCOLHIDO, LIGHTGBM, REGRA

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
APAGADO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

# A cor segue a entidade em todas as figuras: azul é sempre o modelo escolhido.
AZUL = "#2a78d6"
AZUL_CLARO = "#86b6ef"
LARANJA = "#eb6834"
AQUA = "#1baf7a"
COR = {ESCOLHIDO: AZUL, REGRA: LARANJA, LIGHTGBM: AQUA}

ESTILO = {
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.labelcolor": TINTA_SECUNDARIA,
    "xtick.color": APAGADO,
    "ytick.color": APAGADO,
    "xtick.labelcolor": TINTA_SECUNDARIA,
    "ytick.labelcolor": TINTA_SECUNDARIA,
    "ytick.major.size": 0,
    "xtick.major.size": 3,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 8,
    "lines.markeredgewidth": 2,
    "lines.markeredgecolor": SUPERFICIE,
    "legend.frameon": False,
    "legend.labelcolor": TINTA_SECUNDARIA,
}


def _nova_figura(titulo: str, subtitulo: str, altura: float = 4.4) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.2, altura), dpi=160)
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.14)
    fig.text(0.02, 0.955, titulo, fontsize=13, fontweight="bold", color=TINTA, va="top")
    fig.text(0.02, 0.885, subtitulo, fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
    return fig, ax


def _numero(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _percentual(valor: float) -> str:
    return f"{valor * 100:.0f}%"


def _erro(valor: float) -> str:
    arredondado = round(valor * 100)
    return "0%" if arredondado == 0 else f"{arredondado:+d}%".replace("-", "−")


def _blocos_curtos(rotulos: pd.Index) -> list[str]:
    return [r.replace(" a ", "\na ") for r in rotulos]


def promessa_x_realidade(tabela: pd.DataFrame, teto: float) -> Figure:
    """Halter por modelo: AUC em validação aleatória x AUC respeitando o tempo."""
    with plt.rc_context(ESTILO):
        fig, ax = _nova_figura(
            "Validação aleatória promete mais do que o modelo entrega no tempo",
            "AUC de cada modelo medida de dois jeitos. A linha vertical é o teto estimado dentro do regime recente.",
            altura=3.4,
        )
        fig.subplots_adjust(left=0.30, bottom=0.26, top=0.74)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        posicoes = range(len(tabela))
        for y, (_, linha) in zip(posicoes, tabela.iterrows()):
            aleatoria, tempo = linha["AUC validação aleatória"], linha["AUC no tempo (média dos blocos)"]
            ax.plot([tempo, aleatoria], [y, y], color=EIXO, linewidth=2, zorder=1)
            ax.plot(aleatoria, y, "o", color=APAGADO, zorder=2, label="validação aleatória" if y == 0 else None)
            ax.plot(tempo, y, "o", color=AZUL, zorder=3, label="no tempo (treina no passado, testa no futuro)" if y == 0 else None)
            ax.annotate(_numero(aleatoria), (aleatoria, y), xytext=(0, 9), textcoords="offset points", ha="center", color=TINTA_SECUNDARIA, fontsize=9)
            ax.annotate(_numero(tempo), (tempo, y), xytext=(0, 9), textcoords="offset points", ha="center", color=TINTA, fontsize=9, fontweight="bold")
        ax.axvline(teto, color=EIXO, linewidth=1)
        ax.annotate(f"teto do regime recente: {_numero(teto)}", (teto, -0.62), xytext=(4, 0), textcoords="offset points", color=APAGADO, fontsize=8.5, va="center")
        ax.set_yticks(list(posicoes), list(tabela.index))
        ax.set_ylim(-0.8, len(tabela) - 0.45)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 0.85)
        ax.set_xlabel("AUC (0,5 = acaso)")
        ax.xaxis.set_major_formatter(lambda v, _: _numero(v))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=9)
        return fig


def auc_por_bloco(por_bloco: dict[str, pd.DataFrame]) -> Figure:
    """AUC de cada concorrente em cada bloco de teste, em ordem cronológica."""
    with plt.rc_context(ESTILO):
        fig, ax = _nova_figura(
            "No tempo, o modelo simples acompanha o complexo e supera a regra de bolso",
            "AUC por bloco de teste. Cada bloco é pontuado por um modelo treinado só com os 12 meses anteriores.",
        )
        fig.subplots_adjust(right=0.80, bottom=0.20)
        rotulos = por_bloco[ESCOLHIDO].index
        x = range(len(rotulos))
        finais = {}
        for nome in (REGRA, LIGHTGBM, ESCOLHIDO):
            valores = por_bloco[nome]["auc"].to_numpy()
            ax.plot(x, valores, "-o", color=COR[nome], label=nome, zorder=3 if nome == ESCOLHIDO else 2)
            finais[nome] = valores[-1]
        ax.axhline(0.5, color=EIXO, linewidth=1)
        ax.annotate("acaso", (len(rotulos) - 1, 0.5), xytext=(0, 4), textcoords="offset points", ha="right", color=APAGADO, fontsize=8.5)
        curtos = {ESCOLHIDO: "Logística", LIGHTGBM: "LightGBM", REGRA: "Regra"}
        for nome, valor in finais.items():
            ax.annotate(f"{curtos[nome]}  {_numero(valor)}", (len(rotulos) - 1, valor), xytext=(9, 0), textcoords="offset points", va="center", color=TINTA_SECUNDARIA, fontsize=9)
        ax.set_xticks(list(x), _blocos_curtos(rotulos))
        ax.set_xlim(-0.25, len(rotulos) - 1 + 0.08)
        ax.set_ylim(0.45, 0.80)
        ax.set_ylabel("AUC")
        ax.yaxis.set_major_formatter(lambda v, _: _numero(v))
        ax.legend(loc="upper left", fontsize=9)
        return fig


def curva_de_ganho(curva: pd.DataFrame, destaque: float = 0.30) -> Figure:
    """Conversões capturadas em função da fração de leads que a equipe consegue contatar."""
    with plt.rc_context(ESTILO):
        fig, ax = _nova_figura(
            "Com a mesma equipe, priorizar pelo modelo captura mais conversões",
            "Fração das conversões capturada conforme a fração dos leads que a equipe consegue contatar em cada bloco.",
        )
        cores = {"sem priorizar": APAGADO, "regra de bolso": LARANJA, "modelo": AZUL}
        for nome in ("sem priorizar", "regra de bolso", "modelo"):
            ax.plot(curva.index, curva[nome], color=cores[nome], label=nome, zorder=3 if nome == "modelo" else 2)
        ponto = curva.loc[curva.index[(curva.index - destaque).__abs__().argmin()]]
        ax.axvline(destaque, color=EIXO, linewidth=1)
        for nome in ("modelo", "regra de bolso", "sem priorizar"):
            ax.plot(destaque, ponto[nome], "o", color=cores[nome], zorder=4)
        nota = (
            f"Ligando para {_percentual(destaque)} dos leads, a equipe captura:\n"
            f"{_percentual(ponto['modelo'])} das conversões com o modelo\n"
            f"{_percentual(ponto['regra de bolso'])} com a regra de bolso\n"
            f"{_percentual(ponto['sem priorizar'])} sem priorizar"
        )
        ax.text(destaque + 0.03, 0.05, nota, color=TINTA_SECUNDARIA, fontsize=9, va="bottom", linespacing=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("fração dos leads contatada")
        ax.set_ylabel("fração das conversões capturada")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(loc="upper left", fontsize=9)
        return fig


def calibracao(tabela: pd.DataFrame) -> Figure:
    """Erro da previsão do total de conversões por bloco, com o modelo cru e com o nível corrigido."""
    with plt.rc_context(ESTILO):
        fig, ax = _nova_figura(
            "O nível da probabilidade atrasa quando o regime de conversão muda",
            "Conversões previstas em relação às reais, por bloco. 0% seria prever exatamente o total de conversões.",
        )
        fig.subplots_adjust(right=0.80, bottom=0.20)
        x = range(len(tabela))
        series = {"modelo cru": ("erro (cru)", AZUL_CLARO), "nível corrigido pela taxa recente": ("erro (corrigido)", AZUL)}
        for nome, (coluna, cor) in series.items():
            ax.plot(x, tabela[coluna], "-o", color=cor, label=nome, zorder=3 if cor == AZUL else 2)
        ax.axhline(0, color=EIXO, linewidth=1)
        pior = int(tabela["erro (corrigido)"].abs().to_numpy().argmax())
        for coluna in ("erro (cru)", "erro (corrigido)"):
            valor = tabela[coluna].iloc[pior]
            ax.annotate(_erro(valor), (pior, valor), xytext=(10, -9), textcoords="offset points", ha="left", va="center", color=TINTA, fontsize=9.5)
        curtos = {"erro (cru)": "cru", "erro (corrigido)": "corrigido"}
        for coluna, nome in curtos.items():
            valor = tabela[coluna].iloc[-1]
            ax.annotate(f"{nome}  {_erro(valor)}", (len(tabela) - 1, valor), xytext=(14, 0), textcoords="offset points", va="center", color=TINTA_SECUNDARIA, fontsize=9)
        ax.set_xticks(list(x), _blocos_curtos(tabela.index))
        ax.set_xlim(-0.25, len(tabela) - 1 + 0.08)
        ax.set_ylim(-0.80, 0.20)
        ax.set_ylabel("erro na previsão do total")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(loc="lower right", fontsize=9)
        return fig
