"""Baixa o dataset Bank Marketing (UCI) para data/raw/."""

from lead_scoring.dados import baixar

if __name__ == "__main__":
    caminho = baixar()
    print(f"Dados em {caminho} ({caminho.stat().st_size / 1e6:.1f} MB)")
