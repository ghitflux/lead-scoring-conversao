"""Download e carga do dataset Bank Marketing (UCI), com reconstrução da linha do tempo.

Fonte: Moro, Cortez e Rita (2014), "A data-driven approach to predict the success of bank
telemarketing". UCI Machine Learning Repository, licença CC BY 4.0.
https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL_UCI = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_INTERNO = "bank-additional.zip"
CSV_INTERNO = "bank-additional/bank-additional-full.csv"
ARQUIVO_PADRAO = Path("data/raw/bank-additional-full.csv")

ALVO = "converteu"
ORDEM_MESES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# `duration` é a duração da ligação: só existe depois do contato e praticamente revela o
# resultado. Usá-la é vazamento de informação, então ela nunca entra no modelo.
COLUNAS_VAZAMENTO = ["duration"]


def baixar(destino: Path = ARQUIVO_PADRAO) -> Path:
    """Baixa o zip da UCI e grava apenas o CSV usado no projeto."""
    destino = Path(destino)
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_UCI, timeout=120) as resposta:
        externo = zipfile.ZipFile(io.BytesIO(resposta.read()))
    interno = zipfile.ZipFile(io.BytesIO(externo.read(ZIP_INTERNO)))
    destino.write_bytes(interno.read(CSV_INTERNO))
    return destino


def reconstruir_periodo(meses: pd.Series, ano_inicial: int = 2008) -> pd.Series:
    """Reconstrói o período (ano-mês) a partir da ordem cronológica das linhas.

    O arquivo não traz o ano, mas as linhas estão em ordem de data (maio/2008 a
    novembro/2010). Sempre que o mês "volta" em relação à linha anterior, virou o ano.
    """
    numero = meses.map({m: i + 1 for i, m in enumerate(ORDEM_MESES)})
    if numero.isna().any():
        raise ValueError("Há valores de mês fora do esperado (jan..dec).")
    virou_ano = (numero.diff() < 0).cumsum()
    ano = ano_inicial + virou_ano
    return pd.PeriodIndex.from_fields(year=ano.to_numpy(), month=numero.to_numpy(), freq="M").to_series(
        index=meses.index
    )


def carregar(caminho: Path = ARQUIVO_PADRAO) -> pd.DataFrame:
    """Lê o CSV, cria o alvo binário, o período e remove a coluna de vazamento."""
    bruto = pd.read_csv(caminho, sep=";")
    return preparar(bruto)


def preparar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Transforma o dataframe bruto no formato usado pelo restante do projeto."""
    df = bruto.copy()
    df.columns = [c.replace(".", "_") for c in df.columns]
    df[ALVO] = (df.pop("y") == "yes").astype(int)
    df["periodo"] = reconstruir_periodo(df["month"])
    df["ordem"] = range(len(df))
    return df.drop(columns=[c for c in COLUNAS_VAZAMENTO if c in df.columns])
