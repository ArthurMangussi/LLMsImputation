import sys

sys.path.append("./")
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from utils.MyMain import BenchmarkPipeline
from utils.MyUtils import MyPipeline
from utils.MeLogSingle import MeLogger

from algorithms.llm import MAPPED_LLMS

from himdi import himdi_score
from chrmi import train_oracle, compute_chrmi


def pipeline_hallucination_metrics(
    model_impt: str, mecanismo: str, tabela_resultados: dict
):
    """
    Calcula HIMDI e CHRMI para os datasets ja imputados por um modelo, sem
    realizar nova imputacao. Reproduz o mesmo StratifiedKFold usado na
    geracao dos dados e le os arquivos de mascara e de dataset imputado
    ja salvos em disco.
    """
    _logger = MeLogger()
    try:
        tag = MAPPED_LLMS[model_impt]
    except Exception:
        tag = model_impt

    os.makedirs(
        f"./results/{tag}/Resultados/{mecanismo}_Hallucination", exist_ok=True
    )

    linhas_resultado = []

    for md in tabela_resultados["missing_rate"]:
        for dados, nome in zip(
            tabela_resultados["datasets"], tabela_resultados["nome_datasets"]
        ):
            df = dados.copy()
            df.columns = df.columns.str.strip()
            X = df.drop(columns="target")
            y = df["target"].values

            _logger.info(
                f"Hallucination metrics = {nome} com MD = {md} no {model_impt}\n"
            )

            fold = 0
            cv = StratifiedKFold(n_splits=5)
            x_cv = X.values

            for train_index, test_index in cv.split(x_cv, y):
                x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                y_treino, y_teste = y[train_index], y[test_index]

                X_treino = pd.DataFrame(x_treino, columns=X.columns)
                X_teste = pd.DataFrame(x_teste, columns=X.columns)

                mask_path = (
                    f"./results/masks/Datasets/{mecanismo}_Mask/"
                    f"{nome}_mask_fold{fold}_md{md}.csv"
                )
                imputed_path = (
                    f"./results/{tag}/Datasets/{mecanismo}_Multivariado/"
                    f"{nome}_{tag}_fold{fold}_md{md}.csv"
                )

                if not os.path.exists(mask_path) or not os.path.exists(imputed_path):
                    _logger.warning(
                        f"Arquivos ausentes para {nome}/{model_impt}/{mecanismo}/"
                        f"md{md}/fold{fold}, pulando."
                    )
                    fold += 1
                    continue

                missing_mask = pd.read_csv(mask_path).astype(bool)
                missing_mask.columns = missing_mask.columns.str.strip()

                df_imputed = pd.read_csv(imputed_path)
                df_imputed.columns = df_imputed.columns.str.strip()
                X_hat = df_imputed[X.columns]

                df_treino_oraculo = X_treino.copy()
                df_treino_oraculo["target"] = y_treino

                df_true = X_teste.copy()
                df_true["target"] = y_teste

                himdi = himdi_score(
                    X_train=X_treino,
                    X_hat=X_hat,
                    missing_mask=missing_mask,
                )

                oracle, oracle_acc = train_oracle(
                    df_treino_oraculo, label_col="target"
                )

                chrmi, _ = compute_chrmi(
                    df_imputed=df_imputed,
                    df_true=df_true,
                    missing_mask=missing_mask,
                    oracle=oracle,
                    label_col="target",
                )

                linhas_resultado.append(
                    {
                        "modelo": tag,
                        "mecanismo": mecanismo,
                        "dataset": nome,
                        "missing_rate": md,
                        "fold": fold,
                        "himdi": himdi,
                        "chrmi": chrmi,
                        "oracle_acc": oracle_acc,
                    }
                )

                fold += 1

    resultados_df = pd.DataFrame(linhas_resultado)
    resultados_df.to_csv(
        f"./results/{tag}/Resultados/{mecanismo}_Hallucination/"
        f"{tag}_{mecanismo}_hallucination_metrics.csv",
        index=False,
    )

    return _logger.info(f"Hallucination_metrics_{model_impt}_{mecanismo}_done!")


if __name__ == "__main__":

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    pipeline = BenchmarkPipeline(datasets)
    tabela_resultados = pipeline.cria_tabela()

    for mecanismo in ["MAR", "MNAR"]:
        # for model_impt in MAPPED_LLMS:
        #     pipeline_hallucination_metrics(model_impt, mecanismo, tabela_resultados)
        for model_impt in ["knn", "mice", "missForest", "saei", "tabpfn"]:
            pipeline_hallucination_metrics(model_impt, mecanismo, tabela_resultados)
