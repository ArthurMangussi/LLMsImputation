"""
Suavizacao do HIMDI (Hallucination Index for Missing Data Imputation).

Substitui o corte binario |rho_jk| > 0.7 por uma agregacao continua
ponderada pela forca de correlacao, correspondendo as Eq. (1)-(3) da
formulacao suavizada. A inclusao de um parceiro k no calculo passa a
depender de suficiencia de dados (n_min linhas conjuntamente observadas
no treino), e nao mais da forca da correlacao em si.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def himdi_score(
    X_train: pd.DataFrame,
    X_hat: pd.DataFrame,
    missing_mask: pd.DataFrame,
    n_min: int = 10,
    epsilon: float = 0.01,
    delta_multiplier: float = 1.0,
    return_per_feature: bool = False,
):
    """
    Calcula o HIMDI suavizado (correlation-weighted).

    Parameters
    ----------
    X_train : pd.DataFrame
        Dados de treino (parcialmente observados), usados para estimar
        rho_jk, ajustar as regressoes x_j ~ x_k e calcular sigma_hat_{j|k}.
    X_hat : pd.DataFrame
        Dataset com os valores imputados (mesmo shape/index de X_train).
    missing_mask : pd.DataFrame
        Mascara booleana (mesmo shape/index), True onde o valor original
        era missing e foi imputado em X_hat.
    n_min : int, default=10
        Numero minimo de linhas conjuntamente observadas (j e k
        nao-nulos em X_train) exigido para que k entre em K_j (Eq. 2).
        Substitui o corte por forca de correlacao da formulacao
        original (|rho_jk| > 0.7); e um requisito de disponibilidade
        de dados, nao de forca estatistica.
    epsilon : float, default=0.01
        Constante de regularizacao no peso w_jk = |rho_jk| + epsilon.
        Garante que a media ponderada permaneca bem definida mesmo
        quando todos os parceiros tem correlacao proxima de zero
        (Eq. 2), sem precisar de uma convencao separada para esse caso.
    delta_multiplier : float, default=1.0
        Multiplicador sobre o desvio-padrao residual para definir a
        banda de tolerancia delta_{j|k} (mesma semantica da versao
        original).
    return_per_feature : bool, default=False
        Se True, retorna tambem o dict {feature: HIMDI_j} (Eq. 2),
        alem do HIMDI agregado (Eq. 3).

    Returns
    -------
    float
        HIMDI agregado do dataset (Eq. 3), media sobre as features com
        pelo menos um parceiro avaliavel. Features sem nenhum parceiro
        com dados suficientes (K_j vazio) sao excluidas dessa media,
        em vez de zeradas por convencao -- ver nota no corpo da funcao.
    dict[str, float], opcional
        HIMDI_j por feature, retornado apenas se return_per_feature=True.
    """
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    corr_matrix = X_train[numeric_cols].corr(method="spearman").abs()

    # X_hat pode trazer celulas nao numericas remanescentes (ex.: parsing
    # imperfeito da saida da LLM). Regressao e subtracao exigem dtype
    # numerico; qualquer valor nao conversivel vira NaN (comparacao com
    # NaN retorna False, entao a celula nao conta como violacao).
    X_hat = X_hat[numeric_cols].apply(pd.to_numeric, errors="coerce")

    himdi_per_feature = {}

    for j in numeric_cols:
        imputed_idx = missing_mask[j]
        if imputed_idx.sum() == 0:
            continue  # feature j nao possui celulas imputadas a avaliar

        weighted_viol_sum = 0.0
        weight_sum = 0.0

        for k in numeric_cols:
            if k == j:
                continue

            # --- Suficiencia de dados para ajustar a regressao (K_j) ---
            # Filtro por disponibilidade, nao por forca de correlacao.
            train_pair = X_train[[k, j]].dropna()
            if len(train_pair) < n_min:
                continue

            # --- Ajuste de x_j ~ x_k e banda de tolerancia delta_{j|k} ---
            reg = LinearRegression()
            reg.fit(train_pair[[k]], train_pair[j])
            residuals = train_pair[j] - reg.predict(train_pair[[k]])
            delta_jk = delta_multiplier * residuals.std()

            # --- Celulas avaliaveis: j imputado e k observado ---
            k_missing = missing_mask.get(
                k, pd.Series(False, index=missing_mask.index)
            )
            valid_idx = imputed_idx & ~k_missing
            if valid_idx.sum() == 0:
                continue  # nenhuma celula avaliavel para este parceiro

            x_partner = X_hat.loc[valid_idx, [k]]
            expected = reg.predict(x_partner)
            actual_imputed = X_hat.loc[valid_idx, j].values

            viol_jk = (np.abs(actual_imputed - expected) > delta_jk).mean()

            # --- Peso continuo por forca de correlacao (Eq. 2) ---
            rho_jk = corr_matrix.loc[j, k]
            w_jk = rho_jk + epsilon

            weighted_viol_sum += w_jk * viol_jk
            weight_sum += w_jk

        if weight_sum < 1e-12:
            # K_j vazio: nenhum parceiro com dados suficientes para j.
            # Restricao rara de disponibilidade de dados -- excluida da
            # media geral em vez de zerada, para nao confundir
            # "nao avaliavel" com "sem hallucination detectado".
            himdi_per_feature[j] = np.nan
            continue

        himdi_per_feature[j] = weighted_viol_sum / weight_sum

    valid_scores = [v for v in himdi_per_feature.values() if not np.isnan(v)]
    himdi_overall = float(np.mean(valid_scores)) if valid_scores else np.nan

    if return_per_feature:
        return himdi_overall, himdi_per_feature
    return himdi_overall
