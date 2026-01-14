import pandas as pd
from google import genai
from google.genai import types
from io import StringIO

import re

from openai import OpenAI
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



def clean_and_parse_llm_data(response_text, expected_shape):
    # 1. Extração via Regex (Limpa as "conversas" da LLM)
    match = re.search(r'```(?:csv)?\s*(.*?)\s*```', response_text, re.DOTALL)
    content = match.group(1).strip() if match else response_text.strip()

    # 2. Estratégia de tentativa e erro (CSV vs Espaços)
    # Tentamos primeiro o separador que você definiu no novo prompt (Vírgula)
    for separator in [",", r"\s+"]:
        try:
            # Usamos header=0 se você espera que a LLM repita os nomes das colunas
            # Se a LLM NÃO retornar cabeçalhos, use header=None
            df_imputed = pd.read_csv(
                StringIO(content), 
                sep=separator, 
                engine="python"
            )

            # Se o shape bater (ou se for compatível ignorando o index), paramos aqui
            if df_imputed.shape == expected_shape:
                return df_imputed
            
            # Caso a LLM tenha retornado uma coluna de índice extra (comum)
            if df_imputed.shape[1] == expected_shape[1] + 1:
                return df_imputed.iloc[:, 1:] # Remove a primeira coluna (índice)

        except Exception:
            continue

    raise ValueError(f"Não foi possível parsear os dados. Esperado {expected_shape}.")

def adjust_prompt(dataset_name: str, missing_data: pd.DataFrame):
    prompt = f"""
    You are an expert data analyst. I am providing a subset of the {dataset_name} Dataset.
    Task: Use your knowledge of this specific dataset's statistical properties (feature ranges, class distributions, and correlations) to perform data imputation.
    Constraint: DO NOT execute Python code. DO NOT provide any conversational text.

    The matrix below contains missing values. Impute them to be as consistent as possible with the original dataset.
    Matrix:
    {missing_data}

    Output Format:
    Return the complete imputed matrix inside a single Markdown code block. 
    Use CSV format (comma-separated values) with the original headers.

    Strict Rules:
    1. Start directly with the code block: ```csv
    2. End exactly with: ```
    3. Ensure the exact same number of rows as the input.
    4. No explanations, no introductory text, no "Here is the matrix".
    """
    return prompt


def llm_impute(
    dataset_name: str,
    X_teste_norm_md: pd.DataFrame,
    model_name: str,
    api: str = "open_router",
) -> str:
    """
    Método para realizar a imputação com Large Language Models (LLMs),
    utilizando a API do Gemini ou OpenRouter

    Args:
        - dataset_name (str)
        - X_teste_norm_md (pd.DataFrame)
        - model_name (str)
        - api (str)
    """
    try:
        output = X_teste_norm_md.copy()

        batch_row = 50
        batch_col = 10

        n_rows, n_cols = X_teste_norm_md.shape

        for row_start in range(0, n_rows, batch_row):
            row_end = min(row_start + batch_row, n_rows)

            for col_start in range(0, n_cols, batch_col):
                col_end = min(col_start + batch_col, n_cols)

                batch = X_teste_norm_md.iloc[row_start:row_end, col_start:col_end]

                match api:
                    case "open_router":
                        client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key="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",
                        )
                        response = client.responses.create(
                            model=model_name,
                            temperature=0.1,
                            input=adjust_prompt(dataset_name=dataset_name, missing_data=batch),
                        )
                        imputed_value_str = response.output[0].content[0].text

                    case "gemini":
                        client = genai.Client(api_key="AIzaSyDYwDWa96EwtzW9PBYtt-k21qHnxO00SQM")
                        response = client.models.generate_content(
                            model=model_name,
                            contents=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch
                            ),
                            config=types.GenerateContentConfig(temperature=0.1),
                        )

                        imputed_value_str = response.text.strip()

                # Converte CSV retornado pela LLM em DataFrame
                df_imputed = clean_and_parse_llm_data(response_text=imputed_value_str,
                                                      expected_shape=batch.shape)
                    

                # Escreve no output
                output.iloc[row_start:row_end, col_start:col_end] = df_imputed.values

    except Exception as e:
        _logger.error(
            f"Erro no batch [{row_start}:{row_end}, {col_start}:{col_end}]: {e}"
        )
    return output
