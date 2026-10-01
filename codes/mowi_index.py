"""
Indice combinado de alucinacao alinhado ao MOWI (Ho et al., 2026).

Alucinacao = "commitment to material information that is neither derivable
nor appropriately qualified". Por linha i com valores imputados:

    h_i        = agregacao (media ou max) de v_ij do HIMDI sobre as celulas
                 imputadas -> "nao derivavel" (nivel Model).
    delta_i^+  = max(delta_i, 0) do CHRMI -> compromisso material alem da
                 incerteza real (nivel World).
    H_i        = h_i * delta_i^+   (AND suave: zera se qualquer parte zera,
                 monotono nao-decrescente em cada argumento, em [0, 1]).

O nivel Input (esparsidade/OOD da linha) fica fora do indice de proposito:
e um fator causal, e entra como variavel explicativa na analise.
"""

import numpy as np
import pandas as pd


def mowi_hallucination_index(
    himdi_cells: pd.DataFrame,
    chrmi_rows: pd.DataFrame,
    tau: float = 0.5,
    eps: float = 0.05,
):
    """
    Parameters
    ----------
    himdi_cells : pd.DataFrame
        Saida per-cell de himdi_score(return_per_cell=True):
        ["row", "feature", "violation"].
    chrmi_rows : pd.DataFrame
        row_df de compute_chrmi: ["row", ..., "delta"].
    tau : float, default=0.5
        Limiar de h_i para a taxa binaria (maioria ponderada dos parceiros
        violando a banda).
    eps : float, default=0.05
        Limiar de delta_i para a taxa binaria (ganho minimo de P(y|x)).

    Returns
    -------
    summary : dict
        "mowi": media de H_i com h_i = media por linha;
        "mowi_max": idem com h_i = max por linha (mais sensivel);
        "mowi_rate": fracao de linhas com h_i > tau E delta_i > eps;
        "h_mean", "delta_plus_mean": componentes, para mostrar de onde vem
        o sinal; "n_rows": linhas avaliadas.
    row_df : pd.DataFrame
        ["row", "h_mean", "h_max", "delta", "H", "H_max"].
    """
    h = himdi_cells.groupby("row")["violation"].agg(h_mean="mean", h_max="max")

    # Inner join: linhas sem nenhuma celula avaliavel no HIMDI (h indefinido)
    # ou sem delta saem do indice em vez de virarem zero.
    row_df = h.join(chrmi_rows.set_index("row")["delta"], how="inner").dropna()
    delta_plus = row_df["delta"].clip(lower=0)
    row_df["H"] = row_df["h_mean"] * delta_plus
    row_df["H_max"] = row_df["h_max"] * delta_plus

    if row_df.empty:
        nan_keys = ["mowi", "mowi_max", "mowi_rate", "h_mean", "delta_plus_mean"]
        return {**dict.fromkeys(nan_keys, np.nan), "n_rows": 0}, row_df.reset_index()

    summary = {
        "mowi": float(row_df["H"].mean()),
        "mowi_max": float(row_df["H_max"].mean()),
        "mowi_rate": float(
            ((row_df["h_mean"] > tau) & (row_df["delta"] > eps)).mean()
        ),
        "h_mean": float(row_df["h_mean"].mean()),
        "delta_plus_mean": float(delta_plus.mean()),
        "n_rows": int(len(row_df)),
    }
    return summary, row_df.reset_index()
