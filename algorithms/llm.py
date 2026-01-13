import pandas as pd
from google import genai
from google.genai import types
import StringIO
import numpy as np

from utils.MeLogSingle import MeLogger

_logger = MeLogger()

DATASET_NAMES = {
    "pima": "Pima Indians Diabetes",
    "cleveland": "Heart Disease (Cleveland)",
    "wiscosin": "Breast Cancer Wisconsin (Diagnostic)",
    "cervical": "Cervical Cancer (Risk Factors)",
    "parkinsons": "Parkinson's Disease (Voice)",
    "hepatitis": "Hepatitis",
    "chronic": "Chronic Kidney Disease (CKD)",
    "stalog": "Statlog (Heart)",
    "mathernal_risk": "Maternal Health Risk",
    "stroke": "Stroke Prediction Dataset",
}


def adjust_prompt(dataset_name: str, missing_data: pd.DataFrame):
    prompt = f"""
    You are an expert data analyst. I am providing a subset of the {dataset_name} Dataset.
    Use your knowledge of this specific dataset's statistical properties (feature ranges, class distributions, and correlations) to perform data imputation.
    You are not allowed to execute any Python code.

    The matrix below contains missing values. Impute them to be as consistent as possible with the original dataset.

    Matrix:
    {missing_data}

    Return ONLY fully imputed matrix in the original shape. No prose
    Ensure every single row provided in the input is present in the output
    Do NOT include explanations, descriptions, markdown, labels, or extra text of any kind.
    """
    return prompt


def gemini_impute(
    dataset_name: str,
    X_teste_norm_md: pd.DataFrame,
    model_name: str = "gemini-3-flash-preview",
) -> str:

    try:
        client = genai.Client(api_key="AIzaSyDYwDWa96EwtzW9PBYtt-k21qHnxO00SQM")
    except Exception as e:
        _logger.error(f"Error initializing the client: {e}")

    output = X_teste_norm_md.copy()

    batch_row = 50
    batch_col = 10

    n_rows, n_cols = X_teste_norm_md.shape

    for row_start in range(0, n_rows, batch_row):
        row_end = min(row_start + batch_row, n_rows)

        for col_start in range(0, n_cols, batch_col):
            col_end = min(col_start + batch_col, n_cols)

            batch = X_teste_norm_md.iloc[row_start:row_end, col_start:col_end]

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=adjust_prompt(
                        dataset_name=dataset_name, missing_data=batch
                    ),
                    config=types.GenerateContentConfig(temperature=0.1),
                )

                imputed_value_str = response.text.strip()

                # Converte CSV retornado pela LLM em DataFrame
                try:
                    # Tenta ler como CSV com vírgula
                    df_imputed = pd.read_csv(
                        StringIO(imputed_value_str),
                        sep=",",
                        engine="python",
                        header=None,
                    )
                    if df_imputed.shape[1] == 1:
                        raise ValueError("Not a real CSV")

                except Exception:
                    # Fallback: separação por espaço
                    df_imputed = pd.read_csv(
                        StringIO(imputed_value_str),
                        sep=r"\s+",
                        engine="python",
                        header=None,
                    )

                # Validação de shape
                if df_imputed.shape != batch.shape:
                    raise ValueError(
                        f"Shape inválido retornado pela LLM. "
                        f"Esperado {batch.shape}, recebido {df_imputed.shape}"
                    )

                # Escreve no output
                output.iloc[row_start:row_end, col_start:col_end] = df_imputed.values

            except Exception as e:
                _logger.error(
                    f"Erro no batch [{row_start}:{row_end}, {col_start}:{col_end}]: {e}"
                )

    return output
