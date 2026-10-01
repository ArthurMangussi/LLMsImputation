"""
CHRMI (Consequential Hallucination Rate for Missing Data Imputation).

Mede, por observacao com pelo menos um valor ausente, o quanto a imputacao
altera a dificuldade de classificacao percebida por um oraculo treinado nos
dados de treino. A dificuldade de uma linha x com rotulo verdadeiro y e
D(x) = 1 - P(y | x). Para cada linha i com algum valor imputado:

    delta_i = D(x_i^true) - D(x_hat_i)
            = P(y_i | x_hat_i) - P(y_i | x_i^true)

delta_i > 0: a imputacao deixou a classificacao facil demais -- os valores
             imputados carregam mais sinal do rotulo do que os valores
             reais (sinal fabricado / alucinacao "confiante").
delta_i < 0: a imputacao deixou a classificacao dificil demais -- os valores
             imputados apagaram ou contradisseram o sinal real do rotulo.
delta_i ~ 0: a imputacao preserva a dificuldade original da observacao.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


class _OracleWrapper:
    """
    Encapsula o XGBClassifier treinado em rotulos codificados (0..n_classes-1,
    exigencia do XGBoost) e o LabelEncoder usado, de forma que as
    probabilidades possam ser consultadas no espaco original de y.
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

    def proba_true_class(self, X: pd.DataFrame, y) -> np.ndarray:
        """P(y_i | x_i) para cada linha; NaN se y_i nao foi visto no treino."""
        proba = self.model.predict_proba(self._to_numeric(X))
        y = np.asarray(y)
        known = np.isin(y, self.encoder.classes_)
        out = np.full(len(y), np.nan)
        out[known] = proba[np.flatnonzero(known), self.encoder.transform(y[known])]
        return out


def train_oracle(df_train: pd.DataFrame, label_col: str):
    """
    Treina o oraculo (classificador de referencia) apenas nas linhas
    100% observadas do fold de treino.

    Returns
    -------
    model : _OracleWrapper
        Oraculo treinado no conjunto de treino inteiro.
    acc : float
        Acuracia do oraculo estimada por validacao cruzada (nao a acuracia
        no proprio treino, que com XGBoost de 200 arvores tende a ficar
        artificialmente perto de 1.0). Datasets/folds onde acc nao esta
        claramente acima de um baseline ingenuo nao devem ser interpretados
        via CHRMI. NaN quando nao ha classe minoritaria suficiente para pelo
        menos 2 folds estratificados.
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
    oracle: _OracleWrapper,
    label_col: str,
):
    """
    Calcula o CHRMI por observacao com pelo menos um valor imputado.

    Parameters
    ----------
    df_imputed : pd.DataFrame
        Dataset imputado (X_hat), inclui label_col.
    df_true : pd.DataFrame
        Dataset original completo (ground truth), mesmo shape e index de
        df_imputed, inclui label_col.
    missing_mask : pd.DataFrame
        Mascara booleana das features, True onde a celula foi imputada.
    oracle : _OracleWrapper
        Classificador treinado via train_oracle().
    label_col : str
        Nome da coluna de rotulo.

    Returns
    -------
    summary : dict
        "chrmi": media de delta_i (com sinal); "chrmi_abs": media de
        |delta_i|; "chrmi_plus": media de max(delta_i, 0), apenas o lado
        "facil demais" -- reducao de incerteza alem do que os dados reais
        sustentam (nivel World do MOWI). delta_i < 0 e perda de
        informacao, nao alucinacao, e por isso fica fora do chrmi_plus.
    row_df : pd.DataFrame
        Uma linha por observacao avaliada, colunas
        ["row", "difficulty_true", "difficulty_imputed", "delta"].
    """
    feature_cols = [c for c in df_imputed.columns if c != label_col]
    idx_rows = df_imputed.index[missing_mask[feature_cols].any(axis=1)]

    if len(idx_rows) == 0:
        empty = pd.DataFrame(
            columns=["row", "difficulty_true", "difficulty_imputed", "delta"]
        )
        return {"chrmi": np.nan, "chrmi_abs": np.nan, "chrmi_plus": np.nan}, empty

    y_true = df_true.loc[idx_rows, label_col]

    difficulty_true = 1 - oracle.proba_true_class(
        df_true.loc[idx_rows, feature_cols], y_true
    )
    difficulty_imputed = 1 - oracle.proba_true_class(
        df_imputed.loc[idx_rows, feature_cols], y_true
    )
    delta = difficulty_true - difficulty_imputed

    row_df = pd.DataFrame(
        {
            "row": idx_rows,
            "difficulty_true": difficulty_true,
            "difficulty_imputed": difficulty_imputed,
            "delta": delta,
        }
    )

    summary = {
        "chrmi": float(np.nanmean(delta)),
        "chrmi_abs": float(np.nanmean(np.abs(delta))),
        "chrmi_plus": float(np.nanmean(np.maximum(delta, 0))),
    }
    return summary, row_df
