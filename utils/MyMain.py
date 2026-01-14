from utils.MyPreprocessing import PreprocessingDatasets
from utils.MeLogSingle import MeLogger

import numpy as np
import pandas as pd


class BenchmarkPipeline:
    def __init__(self, datasets: dict):
        self._logger = MeLogger()
        self._prep = PreprocessingDatasets()
        self.datasets = datasets
        self.pima = self.pre_processing_pima()
        self.cleveland = self.pre_processing_cleveland()
        self.wiscosin = self.pre_processing_wiscosin()
        self.parkinsons = self.pre_processing_parkinsons()
        self.hepatitis = self.pre_processing_hepatitis()
        self.mathernal_risk = self.pre_processing_mathernal_rick()
        self.cervical = self.pre_processing_cervical()
        self.chronic = self.pre_processing_chronic()
        self.stalog = self.pre_processing_stalog_heart()
        self.stroke = self.pre_processing_stroke()

    # ------------------------------------------------------------------------
    def pre_processing_cervical(self):
        cervical = self.datasets["risk_factors_cervical_cancer"].copy()
        cervical = cervical.drop(
            columns=[
                "STDs: Time since last diagnosis",
                "STDs: Time since first diagnosis",
            ]
        ).replace("?", np.nan)
        return cervical.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_chronic(self):
        chronic = self.datasets["kidney_disease"].copy()
        chronic.drop("id", axis=1, inplace=True)
        chronic.columns = [
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "aanemia",
            "target",
        ]

        chronic = self._prep.ordinal_encoder(
            chronic,
            [
                "aanemia",
                "peda_edema",
                "appetite",
                "coronary_artery_disease",
                "diabetes_mellitus",
                "hypertension",
                "bacteria",
                "pus_cell",
                "pus_cell_clumps",
                "red_blood_cells",
                "target",
            ],
        )

        return chronic.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_stalog_heart(self):
        stalog = self.datasets["heart"].copy()
        stalog = self._prep.label_encoder(stalog, ["target"])

        return stalog

    # ------------------------------------------------------------------------
    def pre_processing_stroke(self):
        stroke = self.datasets["healthcare-dataset-stroke-data"].copy()
        stroke.drop("id", axis=1, inplace=True)
        stroke = self._prep.ordinal_encoder(
            stroke, ["gender", "ever_married", "smoking_status"]
        )
        stroke = self._prep.one_hot_encode(
            stroke,
            [
                "Residence_type",
                "work_type",
            ],
        )
        return stroke.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_hepatitis(self):
        hepatitis = self.datasets["hepatitis"].copy()
        hepatitis = hepatitis.replace("?", np.nan).dropna()
        hepatitis = self._prep.label_encoder(hepatitis, ["target"])
        return hepatitis.astype("float64")

    # ------------------------------------------------------------------------
    def pre_processing_cleveland(self):
        heart_cleveland = self.datasets["cleveland"].copy()
        heart_cleveland = self._prep.label_encoder(heart_cleveland, ["target"])
        return heart_cleveland

    # ------------------------------------------------------------------------
    def pre_processing_mathernal_rick(self):
        maternal_health_risk_df = self.datasets["Maternal Health Risk Data Set"].copy()
        maternal_health_risk_df = self._prep.label_encoder(
            maternal_health_risk_df, ["target"]
        )
        return maternal_health_risk_df

    # ------------------------------------------------------------------------
    def pre_processing_parkinsons(self):
        parkinsons_df = self.datasets["parkinsons"].copy().drop(columns="name")
        return parkinsons_df

    # ------------------------------------------------------------------------
    def pre_processing_wiscosin(self):
        breast_cancer_wisconsin_df = self.datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = self._prep.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )
        return breast_cancer_wisconsin_df

    # ------------------------------------------------------------------------
    def pre_processing_pima(self):
        pima_diabetes_df = self.datasets["pima_diabetes"].copy()
        return pima_diabetes_df

    # ------------------------------------------------------------------------
    def pre_processing_covid(self):
        """
        Method to preprocess the covid dataset

        Returns:
            pd.DataFrame: COVID data
        """
        df = self.datasets["covid"].copy()
        df = df.drop(columns="id_notificacao")
        return df

    # ------------------------------------------------------------------------
    def cria_tabela_sintetico(self):
        tabela_resultados = {}

        syn_cat = pd.read_csv("./data/synthetic/synthetic-cat.csv")
        syn_cont = pd.read_csv("./data/synthetic/synthetic-cont.csv")
        syn_cont_cat = pd.read_csv("./data/synthetic/synthetic-cont-cat.csv")

        tabela_resultados["datasets"] = [syn_cat, syn_cont, syn_cont_cat]

        tabela_resultados["nome_datasets"] = [
            "synthetic-cont-cat",
            "synthetic-cat",
            "synthetic-cont",
        ]
        tabela_resultados["missing_rate"] = [5, 10, 20]

        return tabela_resultados

    # ------------------------------------------------------------------------
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [
            self.pima,
            #self.cleveland,
            #self.wiscosin,
            #self.parkinsons,
            #self.hepatitis,
            #self.mathernal_risk,
            #self.cervical,
            #self.chronic,
            #self.stalog,
            #self.stroke,
        ]

        tabela_resultados["nome_datasets"] = [
            "pima",
            #"cleveland",
            #"wiscosin",
            #"parkinsons",
            #"hepatitis",
            #"mathernal_risk",
            #"cervical",
            #"chronic",
            #"stalog",
            #"stroke",
        ]

        tabela_resultados["missing_rate"] = [5, 
                                             #10, 
                                             # 20
                                             ]

        return tabela_resultados
