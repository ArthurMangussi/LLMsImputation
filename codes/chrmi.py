"""
CHRMI (Consequential Hallucination Rate for Missing Data Imputation).

Reescreve a implementacao original em granularidade de CELULA, e nao de
linha: para cada celula imputada (i, j), isola o efeito marginal daquele
valor especifico sobre a decisao do oraculo, mantendo todas as demais
celulas da linha (inclusive outras imputadas) fixas em seus valores
imputados. Isso corresponde as Eq. (1)-(4) da formalizacao em LaTeX e
permite agregar CHRMI por feature, na mesma granularidade do HIMDI.

Diferenca em relacao a versao original: o codigo original comparava a
linha imputada completa contra a linha verdadeira completa, misturando
o efeito de todas as celulas imputadas daquela linha em um unico flip.
Em missingness multivariado (mais de uma celula ausente por linha),
isso impede atribuir o flip a uma celula especifica e impede comparar
CHRMI com HIMDI celula-a-celula.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


class _OracleWrapper:
    """
    Encapsula o XGBClassifier treinado em rotulos codificados (0..n_classes-1,
    exigencia do XGBoost) e o LabelEncoder usado, de forma que .predict()
    devolva as classes no espaco original de y -- o mesmo espaco de
    df_true[label_col] em compute_chrmi.
    """

    def __init__(self, model, encoder: LabelEncoder):
        self.model = model
        self.encoder = encoder

    @staticmethod
    def _to_numeric(X: pd.DataFrame) -> pd.DataFrame:
        # Alguns datasets trazem colunas object por sujeira de origem (ex.:
        # placeholders de string em campos numericos). O XGBoost exige dtype
        # numerico estrito, entao qualquer valor nao conversivel vira NaN
        # (que o XGBoost trata nativamente como ausente).
        return X.apply(pd.to_numeric, errors="coerce")

    def predict(self, X):
        return self.encoder.inverse_transform(
            self.model.predict(self._to_numeric(X))
        )


def train_oracle(df_train: pd.DataFrame, label_col: str):
    """
    Treina o oraculo (classificador de referencia) apenas nas linhas
    100% observadas do fold de treino.

    Returns
    -------
    model : _OracleWrapper
        Oraculo treinado (no conjunto de treino inteiro, para aproveitar o
        maximo de dados), com .predict() no espaco original dos rotulos.
    acc : float
        Acuracia do oraculo estimada por validacao cruzada (nao a acuracia
        no proprio conjunto de treino, que com XGBoost de 200 arvores tende
        a memorizar os dados e ficar artificialmente perto de 1.0, mascarando
        overfitting). Reporte isto por dataset/fold antes de confiar no
        oraculo -- ver "Validity precondition" na formalizacao (paragrafo
        antes da Eq. 1): datasets/folds onde acc nao esta claramente acima
        de um baseline ingenuo nao devem ser interpretados via CHRMI.
        NaN quando nao ha classe minoritaria suficiente para pelo menos 2
        folds estratificados.
    """
    fully_observed = df_train.dropna()
    X = _OracleWrapper._to_numeric(fully_observed.drop(columns=[label_col]))
    y = fully_observed[label_col]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    def _novo_modelo():
        return xgb.XGBClassifier(n_estimators=200, max_depth=4, eval_metric="mlogloss")

    n_splits = min(5, np.bincount(y_encoded).min())
    if n_splits >= 2:
        acc = cross_val_score(
            _novo_modelo(), X, y_encoded, cv=n_splits, scoring="accuracy"
        ).mean()
    else:
        acc = np.nan

    model = _novo_modelo()
    model.fit(X, y_encoded)
    return _OracleWrapper(model, encoder), acc


def compute_chrmi(
    df_imputed: pd.DataFrame,
    df_true: pd.DataFrame,
    missing_mask: pd.DataFrame,
    oracle,
    label_col: str,
    material: bool = True,
    return_per_feature: bool = False,
):
    """
    Calcula o CHRMI celula-a-celula, Eq. (1)-(4).

    Para cada celula imputada (i, j): compara a predicao do oraculo na
    linha totalmente imputada (baseline, X_hat_i) contra a predicao na
    mesma linha com APENAS a celula j trocada pelo valor verdadeiro
    (contrafactual X_hat_i^{(j->true)}), mantendo todas as demais
    celulas -- inclusive outras imputadas na mesma linha -- inalteradas
    (Eq. 1).

    Parameters
    ----------
    df_imputed : pd.DataFrame
        Dataset com todos os valores imputados (X_hat), inclui a coluna
        de rotulo (label_col).
    df_true : pd.DataFrame
        Dataset com os valores verdadeiros (ground truth), mesmo shape
        e index de df_imputed, inclui label_col.
    missing_mask : pd.DataFrame
        Mascara booleana (mesmo shape/index das features, sem
        label_col), True onde a celula original era missing e foi
        imputada.
    oracle : objeto com metodo .predict(X) -> array de classes
        Classificador treinado via train_oracle().
    label_col : str
        Nome da coluna de rotulo.
    material : bool, default=True
        Se True, aplica a condicao de materialidade (Eq. 2): so conta
        como hallucination quando a predicao contrafactual-verdadeira
        coincide com o rotulo real y_i. Se False, retorna delta_ij
        (Eq. 1) sem essa restricao.
    return_per_feature : bool, default=False
        Se True, retorna tambem CHRMI_j por feature (Eq. 3).

    Returns
    -------
    chrmi_overall : float
        CHRMI agregado (Eq. 4), media sobre todas as celulas avaliadas.
    chrmi_per_feature : dict[str, float], opcional
        CHRMI_j por feature, retornado apenas se return_per_feature=True.
    flip_df : pd.DataFrame
        Uma linha por celula avaliada, colunas ["row", "feature", "flip"],
        com o vetor delta_ij^mat (ou delta_ij, se material=False).
    """
    feature_cols = [c for c in df_imputed.columns if c != label_col]
    mask = missing_mask[feature_cols]

    rows_with_missing = mask.any(axis=1)
    idx_rows = df_imputed.index[rows_with_missing]

    if len(idx_rows) == 0:
        empty = pd.DataFrame(columns=["row", "feature", "flip"])
        if return_per_feature:
            return np.nan, {}, empty
        return np.nan, empty

    # --- Predicao baseline: linha totalmente imputada (X_hat_i), Eq. 1 ---
    X_baseline = df_imputed.loc[idx_rows, feature_cols]
    pred_baseline = pd.Series(oracle.predict(X_baseline), index=idx_rows)

    # --- Construcao vetorizada das linhas contrafactuais ---
    # Uma linha por celula imputada: X_hat_i com APENAS a coluna j
    # trocada pelo valor verdadeiro (Eq. 1, X_hat_i^{(j->true)}).
    records = []  # (row_idx, feature)
    cf_rows = []
    for i in idx_rows:
        missing_cols_i = mask.loc[i]
        missing_cols_i = missing_cols_i[missing_cols_i].index.tolist()
        base_row = X_baseline.loc[i]
        for j in missing_cols_i:
            cf_row = base_row.copy()
            cf_row[j] = df_true.loc[i, j]
            cf_rows.append(cf_row)
            records.append((i, j))

    X_cf = pd.DataFrame(cf_rows, columns=feature_cols)
    pred_cf = oracle.predict(X_cf)

    y_true = df_true[label_col]

    flip_records = []
    for (i, j), pred_swap in zip(records, pred_cf):
        pred_imp = pred_baseline.loc[i]
        flip = bool(pred_imp != pred_swap)  # delta_ij, Eq. 1
        if material:
            flip = flip and bool(pred_swap == y_true.loc[i])  # delta_ij^mat, Eq. 2
        flip_records.append((i, j, flip))

    flip_df = pd.DataFrame(flip_records, columns=["row", "feature", "flip"])

    chrmi_overall = float(flip_df["flip"].mean())

    if return_per_feature:
        chrmi_per_feature = flip_df.groupby("feature")["flip"].mean().to_dict()
        return chrmi_overall, chrmi_per_feature, flip_df

    return chrmi_overall, flip_df
