from utils.MyPreprocessing import PreprocessingDatasets
from utils.MeLogSingle import MeLogger

import numpy as np
import pandas as pd


class BenchmarkPipeline:
    def __init__(self, datasets: dict):
        self._logger = MeLogger()
        self._prep = PreprocessingDatasets()
        self.datasets = datasets
        self.bc_coimbra = self.pre_processing_bcCoimbra()
        self.wiscosin = self.pre_processing_wiscosin()
        self.covid = self.pre_processing_covid()
        self.diabetes = self.pre_processing_diabetes()
        self.echocardiogram = self.pre_processing_echocardiogram()
        self.hcv = self.pre_processing_HCV()
        self.cleveland = self.pre_processing_cleveland()
        self.mathernal_risk = self.pre_processing_mathernal_rick()
        self.parkinsons = self.pre_processing_parkinsons()
        self.pima = self.pre_processing_pima()
        self.ricci = self.pre_processing_ricci()
        self.thoracic_surgery = self.pre_processing_thoracic()
        self.liver = self.pre_processing_indianLiver()
        self.acute = self.pre_processing_acute()
        self.autism_teen = self.pre_processing_autism_teen()
        self.autism_adult = self.pre_processing_autism_adult()
        self.autism_child = self.pre_processing_autism_child()
        self.bank = self.pre_processing_bank()
        self.blood_transfusion = self.pre_processing_blood()
        self.wine = self.pre_processing_wine()
        self.fertility = self.pre_processing_fertility()
        self.phoneme = self.pre_processing_phoneme()
        self.npha = self.pre_processing_npha()
        self.german_credit = self.pre_processing_german_credit()
        self.prob_football = self.pre_processing_football()
        self.diabetic = self.pre_processing_diabetic()
        self.thyroid = self.pre_processing_thyroid()
        self.haberman = self.pre_processing_haberman()
        self.hepatitis = self.pre_processing_hepatitis()
        self.sa_heart = self.pre_processing_saHeart()
        self.iris = self.pre_processing_iris()
        self.contraceptive = self.pre_processing_contraceptive()
        self.mammo_masses = self.pre_processing_mammo()
        self.adult = self.pre_processing_adult()
        self.law = self.pre_processing_law()
        self.dutch = self.pre_processing_dutch()
        self.student_math = self.pre_processing_student_math()
        self.student_port = self.pre_processing_student_port()
        self.compass_4k = self.pre_processing_compass_4k()
        self.compass_7k = self.pre_processing_compass_7k()
        self.cirohis = self.pre_processing_cirohis()
        self.obesity = self.pre_processing_obesity()
        self.lymphography = self.pre_processing_lymphography()
        self.mushroom = self.pre_processing_mushroom()
        self.nursery = self.pre_processing_nursery()
        self.heart_failure = self.pre_processing_heart_failure()
        self.breast_tissue = self.pre_processing_breast_tissue()
        self.monk_um = self.pre_processing_monk_um()
        self.monk_dois = self.pre_processing_monk_dois()
        self.monk_tres = self.pre_processing_monk_tres()
        self.glioma = self.pre_processing_glioma

    # ------------------------------------------------------------------------
    def pre_processing_monk_um(self):
        MONK_problem1_train = self.datasets["monks-1-train"].copy()
        MONK_problem1_test = self.datasets["monks-1-test"].copy()
        MONK_problem1 = pd.concat([MONK_problem1_train, MONK_problem1_test]).drop(
            columns="Id"
        )
        MONK_problem1 = MONK_problem1.astype("float64")
        return MONK_problem1

    # ------------------------------------------------------------------------
    def pre_processing_monk_dois(self):
        MONK_problem2_train = self.datasets["monks-2-train"].copy()
        MONK_problem2_test = self.datasets["monks-2-test"].copy()
        MONK_problem2 = pd.concat([MONK_problem2_train, MONK_problem2_test]).drop(
            columns="Id"
        )
        MONK_problem2 = MONK_problem2.astype("float64")
        return MONK_problem2

    # ------------------------------------------------------------------------
    def pre_processing_monk_tres(self):
        MONK_problem3_train = self.datasets["monks-3-train"].copy()
        MONK_problem3_test = self.datasets["monks-3-test"].copy()
        MONK_problem3 = pd.concat([MONK_problem3_train, MONK_problem3_test]).drop(
            columns="Id"
        )
        MONK_problem3 = MONK_problem3.astype("float64")
        return MONK_problem3

    # ------------------------------------------------------------------------
    def pre_processing_breast_tissue(self):
        breast_tissue = self.datasets["BreastTissue"].copy()
        breast_tissue = self._prep.label_encoder(breast_tissue, ["target"])
        return breast_tissue

    # ------------------------------------------------------------------------
    def pre_processing_heart_failure(self):
        return self.datasets["heart_failure_clinical_records_dataset"].copy()

    # ------------------------------------------------------------------------
    def pre_processing_lymphography(self):
        lymphography_df = self.datasets["lymphography"].copy()
        lymphography_df = self._prep.label_encoder(
            lymphography_df,
            [
                "target",
                "block_of_affere",
                "bl_of_lymph_c",
                "bl_of_lymph_s",
                "by_pass",
                "extravasates",
                "regeneration_of",
                "early_uptake_in",
                "dislocation_of",
                "exclusion_of_no",
            ],
        )
        lymphography_df = self._prep.ordinal_encoder(lymphography_df, ["lymphatics"])
        lymphography_df = self._prep.one_hot_encode(
            lymphography_df,
            [
                "changes_in_lym",
                "defect_in_node",
                "changes_in_node",
                "changes_in_stru",
                "special_forms",
            ],
        )
        return lymphography_df

    # ------------------------------------------------------------------------
    def pre_processing_nursery(self):
        nursery = self.datasets["nursery"].copy()
        nursery = self._prep.label_encoder(nursery, ["target", "finance"])
        nursery = self._prep.ordinal_encoder(nursery, ["form", "children", "health"])
        nursery = self._prep.one_hot_encode(
            nursery, ["parents", "social", "has_nurs", "housing"]
        )
        return nursery

    # ------------------------------------------------------------------------
    def pre_processing_mushroom(self):
        mushroom = self.datasets["agaricus-lepiota"].copy()
        mushroom = mushroom.replace("?", np.nan).dropna()
        mushroom = self._prep.label_encoder(mushroom, ["target"])
        mushroom = self._prep.one_hot_encode(
            mushroom,
            [
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
                "a5",
                "a6",
                "a7",
                "a8",
                "a9",
                "a10",
                "a11",
                "a12",
                "a13",
                "a14",
                "a15",
                "a16",
                "a17",
                "a18",
                "a19",
                "a20",
                "a21",
            ],
        ).dropna()
        return mushroom

    # ------------------------------------------------------------------------
    def pre_processing_obesity(self):
        obesity_eating = self.datasets["ObesityDataSet_raw_and_data_sinthetic"].copy()
        obesity_eating = self._prep.label_encoder(
            obesity_eating,
            [
                "target",
                "Gender",
                "family_history_with_overweight",
                "FAVC",
                "SMOKE",
                "SCC",
                "CAEC",
            ],
        )
        obesity_eating = self._prep.ordinal_encoder(obesity_eating, ["FCVC", "CALC"])
        obesity_eating = self._prep.one_hot_encode(obesity_eating, ["MTRANS"])
        return obesity_eating

    # ------------------------------------------------------------------------
    def pre_processing_student_port(self):
        student_port_df = self.datasets["student-por"].copy()
        student_port_df.target = [
            1 if nota >= 10 else 0 for nota in student_port_df.target
        ]
        student_port_df.age = [1 if idade >= 18 else 0 for idade in student_port_df.age]

        student_port_df = self._prep.ordinal_encoder(
            student_port_df,
            [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ],
        )
        student_port_df = self._prep.one_hot_encode(
            student_port_df, ["Mjob", "Fjob", "reason", "guardian"]
        )
        return student_port_df

    # ------------------------------------------------------------------------
    def pre_processing_student_math(self):
        student_mat_df = self.datasets["student-mat"].copy()
        student_mat_df.target = [
            1.0 if nota >= 10 else 0.0 for nota in student_mat_df.target
        ]
        student_mat_df.age = [
            1.0 if idade >= 18 else 0.0 for idade in student_mat_df.age
        ]

        student_mat_df = self._prep.ordinal_encoder(
            student_mat_df,
            [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ],
        )
        student_mat_df = self._prep.one_hot_encode(
            student_mat_df, ["Mjob", "Fjob", "reason", "guardian"]
        )
        return student_mat_df

    # ------------------------------------------------------------------------
    def pre_processing_compass_7k(self):
        compass_7k_df = self.datasets["compas-scores-two-years_clean"].copy()
        clean_compass_7k = compass_7k_df.drop(
            columns=[
                "id",
                "name",
                "first",
                "last",
                "compas_screening_date",
                "dob",
                "days_b_screening_arrest",
                "c_jail_in",
                "c_jail_out",
                "c_case_number",
                "c_offense_date",
                "c_arrest_date",
                "age_cat",
                "vr_case_number",
                "vr_offense_date",
                "decile_score.1",
                "r_case_number",
                "r_offense_date",
                "screening_date",
                "v_screening_date",
                "in_custody",
                "out_custody",
                "priors_count.1",
                "r_jail_in",
                "r_jail_out",
                "vr_charge_degree",
                "vr_charge_desc",
                "v_type_of_assessment",
                "type_of_assessment",
                "violent_recid",
                "r_charge_degree",
                "r_days_from_arrest",
                "c_charge_desc",
                "r_charge_desc",
            ]
        )
        map_races_compass = {
            "African-American": 1,
            "Caucasian": 0,
            "Hispanic": 0,
            "Other": 0,
            "Asian": 0,
            "Native American": 0,
        }
        clean_compass_7k["race"] = clean_compass_7k["race"].map(map_races_compass)

        map_colum = {"two_year_recid": "target"}
        clean_compass_7k = clean_compass_7k.rename(columns=map_colum)
        clean_compass_7k = self._prep.label_encoder(clean_compass_7k, ["target"])
        clean_compass_7k = self._prep.ordinal_encoder(
            clean_compass_7k, ["sex", "c_charge_degree"]
        )
        clean_compass_7k = self._prep.one_hot_encode(
            clean_compass_7k,
            [
                "score_text",
                "v_score_text",
            ],
        )
        return compass_7k_df

    # ------------------------------------------------------------------------
    def pre_processing_compass_4k(self):
        map_colum = {"two_year_recid": "target"}
        map_races_compass = {
            "African-American": 1,
            "Caucasian": 0,
            "Hispanic": 0,
            "Other": 0,
            "Asian": 0,
            "Native American": 0,
        }
        compass_4k_df = self.datasets["compas-scores-two-years-violent_clean"].copy()
        clean_compass_4k = (
            compass_4k_df.drop(
                columns=[
                    "id",
                    "name",
                    "first",
                    "last",
                    "compas_screening_date",
                    "dob",
                    "days_b_screening_arrest",
                    "c_jail_in",
                    "c_jail_out",
                    "c_case_number",
                    "c_offense_date",
                    "c_arrest_date",
                    "age_cat",
                    "vr_case_number",
                    "vr_offense_date",
                    "decile_score.1",
                    "r_case_number",
                    "r_offense_date",
                    "screening_date",
                    "v_screening_date",
                    "in_custody",
                    "out_custody",
                    "priors_count.1",
                    "r_jail_in",
                    "r_jail_out",
                    "vr_charge_degree",
                    "vr_charge_desc",
                    "v_type_of_assessment",
                    "type_of_assessment",
                    "violent_recid",
                    "r_charge_degree",
                    "r_days_from_arrest",
                    "c_charge_desc",
                    "r_charge_desc",
                ]
            )
            .dropna()
            .reset_index(drop=True)
        )
        clean_compass_4k = clean_compass_4k.rename(columns=map_colum)
        clean_compass_4k["race"] = clean_compass_4k["race"].map(map_races_compass)
        clean_compass_4k = self._prep.label_encoder(clean_compass_4k, ["target"])
        clean_compass_4k = self._prep.ordinal_encoder(
            clean_compass_4k, ["sex", "c_charge_degree"]
        )
        clean_compass_4k = self._prep.one_hot_encode(
            clean_compass_4k, ["score_text", "v_score_text"]
        )
        return clean_compass_4k

    # ------------------------------------------------------------------------
    def pre_processing_dutch(self):
        dutch_df = self.datasets["dutch"].copy()
        map_dutch = {"occupation": "target"}
        dutch_df = dutch_df.rename(columns=map_dutch)
        dutch_df = self._prep.label_encoder(dutch_df, ["target"])
        dutch_df = self._prep.ordinal_encoder(
            dutch_df, ["sex", "household_size", "cur_eco_activity"]
        )
        dutch_df = self._prep.one_hot_encode(
            dutch_df,
            [
                "household_position",
                "country_birth",
                "edu_level",
                "economic_status",
                "marital_status",
            ],
        )
        return dutch_df

    # ------------------------------------------------------------------------
    def pre_processing_diabetes(self):
        diabetes_df = self.datasets["diabetes-clean"].copy()
        diabetes_df = diabetes_df.drop(
            columns=[
                "encounter_id",
                "patient_nbr",
                "metformin-rosiglitazone",
                "metformin-pioglitazone",
                "citoglipton",
                "examide",
                "A1Cresult",
                "max_glu_serum",
                "discharge_disposition_id",
                "admission_source_id",
            ]
        )
        diabetes_df = self._prep.label_encoder(diabetes_df, ["target"])
        diabetes_df = self._prep.ordinal_encoder(
            diabetes_df,
            [
                "age",
                "diabetesMed",
                "glimepiride-pioglitazone",
                "glipizide-metformin",
                "troglitazone",
                "tolbutamide",
                "acetohexamide",
                "diag_1",
                "diag_2",
                "diag_3",
                "acetohexamide",
                "tolbutamide",
                "gender",
                "change",
            ],
        )
        diabetes_df = self._prep.one_hot_encode(
            diabetes_df,
            [
                "race",
                "metformin",
                "chlorpropamide",
                "glipizide",
                "rosiglitazone",
                "acarbose",
                "miglitol",
                "repaglinide",
                "nateglinide",
                "glimepiride",
                "glyburide",
                "pioglitazone",
                "tolazamide",
                "insulin",
                "glyburide-metformin",
            ],
        )
        return diabetes_df

    # ------------------------------------------------------------------------
    def pre_processing_law(self):
        law_df = self.datasets["law_school_clean"].copy()
        law_df = self._prep.label_encoder(law_df, ["target"])
        law_df = self._prep.one_hot_encode(law_df, ["fam_inc", "race"])

        return law_df

    # ------------------------------------------------------------------------
    def pre_processing_adult(self):
        df = self.datasets["adult-clean"].copy()
        df = self._prep.ordinal_encoder(
            df, ["age", "education", "occupation", "gender"]
        )

        df = self._prep.one_hot_encode(
            df,
            ["workclass", "marital-status", "relationship", "native-country", "race"],
        )
        df = self._prep.label_encoder(df, ["target"])
        return df

    # ------------------------------------------------------------------------
    def pre_processing_mammo(self):
        mammographic_masses_df = self.datasets["mammographic_masses"].copy()
        mammographic_masses_df = (
            self.datasets["mammographic_masses"]
            .replace("?", np.nan)
            .dropna()
            .drop(columns="BI-RADS assessment")
            .reset_index(drop=True)
        )
        mammographic_masses_df["Age"] = mammographic_masses_df["Age"].astype("int64")
        mammographic_masses_df["Density"] = mammographic_masses_df["Density"].astype(
            "int64"
        )
        mammographic_masses_df = self._prep.one_hot_encode(
            mammographic_masses_df, ["Shape", "Margin"]
        )
        return mammographic_masses_df

    # ------------------------------------------------------------------------
    def pre_processing_german_credit(self):
        german_credit_df = self.datasets["german"].copy()

        map_gender = {
            "A91": "male",
            "A92": "female",
            "A93": "male",
            "A94": "male",
            "A95": "female",
        }

        german_credit_df["personal-status-and-sex"] = german_credit_df[
            "personal-status-and-sex"
        ].map(map_gender)

        german_credit_df = self._prep.ordinal_encoder(
            german_credit_df,
            [
                "age",
                "checking-account",
                "savings-account",
                "employment-since",
                "telephone",
                "foreign-worker",
                "personal-status-and-sex",
            ],
        )
        german_credit_df = self._prep.label_encoder(german_credit_df, ["target"])

        german_credit_df = self._prep.one_hot_encode(
            german_credit_df,
            [
                "credit-history",
                "purpose",
                "other-debtors",
                "property",
                "other-installment",
                "housing",
                "job",
            ],
        )

        return german_credit_df

    # ------------------------------------------------------------------------
    def pre_processing_diabetic(self):
        messidor_df = self.datasets["messidor_features"].copy()
        messidor_df["target"] = messidor_df["target"].astype("int64")
        return messidor_df

    # ------------------------------------------------------------------------
    def pre_processing_thyroid(self):
        thyroid_df = self.datasets["Thyroid_Diff"].copy()
        thyroid_df = self._prep.label_encoder(thyroid_df, ["target"])

        thyroid_df = self._prep.ordinal_encoder(
            thyroid_df,
            [
                "Physical Examination",
                "Gender",
                "Smoking",
                "Hx Smoking",
                "Hx Radiothreapy",
                "Focality",
                "Thyroid Function",
                "Pathology",
                "Risk",
                "T",
                "N",
                "M",
                "Stage",
                "Response",
            ],
        )

        thyroid_df = self._prep.one_hot_encode(thyroid_df, ["Adenopathy"])
        return thyroid_df

    # ------------------------------------------------------------------------
    def pre_processing_hepatitis(self):
        hepatitis = self.datasets["hepatitis"].copy()
        hepatitis = hepatitis.replace("?", np.nan).dropna()
        hepatitis = self._prep.label_encoder(hepatitis, ["target"])
        return hepatitis.astype("float64")

    # ------------------------------------------------------------------------
    def pre_processing_saHeart(self):
        sa_heart = self.datasets["sa-heart"].copy()
        sa_heart = self._prep.label_encoder(sa_heart, ["target"])
        return sa_heart

    # ------------------------------------------------------------------------
    def pre_processing_haberman(self):
        haberman_df = self.datasets["dataset_43_haberman"].copy()
        haberman_df = self._prep.label_encoder(haberman_df, ["target"])
        return haberman_df.astype("float64")

    # ------------------------------------------------------------------------
    def pre_processing_acute(self):
        acute_inflammations = self.datasets["diagnosis1"].copy()
        acute_inflammations = acute_inflammations.drop(columns="target.1")
        acute_inflammations = self._prep.label_encoder(
            acute_inflammations,
            ["Nausea", "Lumbar", "Urine", "Micturition", "Burning", "target"],
        )
        return acute_inflammations

    # ------------------------------------------------------------------------
    def pre_processing_npha(self):
        npha_doctor_visits_df = (
            self.datasets["NPHA-doctor-visits"].copy().drop(columns="Age")
        )
        npha_doctor_visits_df = self._prep.label_encoder(
            npha_doctor_visits_df, ["target"]
        )
        return npha_doctor_visits_df

    # ------------------------------------------------------------------------
    def pre_processing_bank(self):
        bank_marketing = self.datasets["bank_marketing"].copy()

        bank_marketing = self._prep.one_hot_encode(
            bank_marketing, ["V4", "V3", "V2", "V9", "V11", "V16"]
        )
        bank_marketing = self._prep.ordinal_encoder(bank_marketing, ["V5", "V7", "V8"])
        bank_marketing = self._prep.label_encoder(bank_marketing, ["target"])
        return bank_marketing

    # ------------------------------------------------------------------------
    def pre_processing_football(self):
        proba_football = self.datasets["prob_sfootball"].copy()
        proba_football = self._prep.label_encoder(proba_football, ["target"]).drop(
            columns=["Weekday"]
        )
        proba_football = self._prep.ordinal_encoder(proba_football, ["Overtime"])
        proba_football = self._prep.one_hot_encode(
            proba_football, ["Favorite_Name", "Underdog_name"]
        )

        return proba_football

    # ------------------------------------------------------------------------
    def pre_processing_fertility(self):
        fertility_df = self.datasets["fertility_Diagnosis"].copy()
        map_season = {-1.0: "winter", -0.33: "spring", 0.33: "summer", -1.0: "fall"}

        map_fever = {
            -1.0: "less than three months ago",
            0.0: "more than three months ago",
            1.0: "no",
        }

        map_smoking = {-1.0: "never", 0.0: "occasional", 1.0: "daily"}

        fertility_df["Season"] = fertility_df["Season"].map(map_season)
        fertility_df[" high fevers"] = fertility_df[" high fevers"].map(map_fever)
        fertility_df["smoking"] = fertility_df["smoking"].map(map_smoking)

        fertility_df = self._prep.label_encoder(fertility_df, ["target"])

        fertility_df = self._prep.one_hot_encode(fertility_df, ["Season"])

        fertility_df = self._prep.ordinal_encoder(
            fertility_df, [" high fevers", " alcohol consumption", "smoking"]
        )
        return fertility_df

    # ------------------------------------------------------------------------
    def pre_processing_contraceptive(self):
        contraceptive_method = self.datasets["cmc"].copy()
        contraceptive_method = self._prep.label_encoder(
            contraceptive_method, ["target"]
        )
        return contraceptive_method

    # ------------------------------------------------------------------------
    def pre_processing_phoneme(self):
        phoneme_df = self.datasets["phoneme"].copy()
        phoneme_df = self._prep.label_encoder(phoneme_df, ["target"])
        return phoneme_df

    # ------------------------------------------------------------------------
    def pre_processing_iris(self):
        iris_df = self.datasets["iris"].copy()
        iris_df = self._prep.label_encoder(iris_df, ["target"])
        return iris_df

    # ------------------------------------------------------------------------
    def pre_processing_wine(self):
        wine_df = self.datasets["wine"].copy()
        wine_df = self._prep.label_encoder(wine_df, ["target"])
        return wine_df

    # ------------------------------------------------------------------------
    def pre_processing_blood(self):
        blood_df = self.datasets["blood-transfusion-service-center"].copy()
        blood_df = self._prep.label_encoder(blood_df, ["target"])
        return blood_df

    # ------------------------------------------------------------------------
    def pre_processing_autism_teen(self):
        autism_adoles_df = self.datasets["Autism-Adolescent-Data"].copy()
        autism_adoles_df = (
            autism_adoles_df.drop(
                columns=["age_desc", "ethnicity", "contry_of_res", "relation"]
            )
            .dropna()
            .reset_index(drop=True)
        )
        autism_adoles_df = self._prep.label_encoder(
            autism_adoles_df,
            ["gender", "jundice", "austim", "used_app_before", "target"],
        )
        return autism_adoles_df.astype("int64")

    # ------------------------------------------------------------------------
    def pre_processing_autism_child(self):
        autism_child_df = self.datasets["Autism-Child-Data"].copy()
        autism_child_df = (
            autism_child_df.drop(
                columns=["age_desc", "ethnicity", "contry_of_res", "relation"]
            )
            .dropna()
            .reset_index(drop=True)
        )
        autism_child_df = self._prep.label_encoder(
            autism_child_df,
            ["gender", "jundice", "austim", "used_app_before", "target"],
        )
        return autism_child_df.astype("int64")

    # ------------------------------------------------------------------------
    def pre_processing_autism_adult(self):
        autism_df = self.datasets["Autism-Adult-Data"].copy()
        autism_df = (
            autism_df.drop(
                columns=["age_desc", "ethnicity", "contry_of_res", "relation"], index=52
            )
            .dropna()
            .reset_index(drop=True)
        )
        autism_df = self._prep.label_encoder(
            autism_df, ["gender", "jundice", "austim", "used_app_before", "target"]
        )
        return autism_df.astype("int64")

    # ------------------------------------------------------------------------
    def pre_processing_echocardiogram(self):
        echocardiogram_df = self.datasets["echocardiogram"].copy()
        echocardiogram_df = echocardiogram_df.replace("?", np.nan).dropna()
        echocardiogram_df = echocardiogram_df.drop(columns=["name", "group", "mult"])
        echocardiogram_df = self._prep.label_encoder(echocardiogram_df, ["target"])
        return echocardiogram_df.astype("float64")

    # ------------------------------------------------------------------------
    def pre_processing_HCV(self):
        hcv_egyptian_df = self.datasets["HCV-Egy-Data"].copy()
        hcv_egyptian_df = self._prep.label_encoder(hcv_egyptian_df, ["target"])
        return hcv_egyptian_df

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
    def pre_processing_indianLiver(self):
        indian_liver_df = self.datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = self._prep.label_encoder(
            indian_liver_df, ["Gender", "target"]
        )
        return indian_liver_df

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
    def pre_processing_ricci(self):
        ricci_df = self.datasets["ricci_processed"].copy()
        ricci_df = self._prep.label_encoder(ricci_df, ["Position", "target"])
        ricci_df = self._prep.one_hot_encode(ricci_df, ["Race"])
        return ricci_df

    # ------------------------------------------------------------------------
    def pre_processing_thoracic(self):
        thoracic_surgery_df = self.datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = self._prep.label_encoder(
            self._prep.one_hot_encode(thoracic_surgery_df, ["DGN"]),
            [
                "PRE7",
                "PRE8",
                "PRE9",
                "PRE10",
                "PRE11",
                "PRE17",
                "PRE19",
                "PRE25",
                "PRE30",
                "PRE32",
                "target",
            ],
        )

        thoracic_surgery_df = self._prep.ordinal_encoder(
            thoracic_surgery_df, ["PRE6", "PRE14"]
        )
        return thoracic_surgery_df

    # ------------------------------------------------------------------------
    def pre_processing_bcCoimbra(self):
        bc_coimbra = self.datasets["bc_coimbra"].copy()
        bc_coimbra = self._prep.label_encoder(bc_coimbra, ["target"])
        return bc_coimbra

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
    def pre_processing_cirohis(self):
        cirrhosis_df = self.datasets["cirrhosis"].copy()
        cirrhosis_df = cirrhosis_df.dropna().drop(columns="ID")
        cirrhosis_df = self._prep.label_encoder(
            cirrhosis_df,
            ["target", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"],
        )
        cirrhosis_df = self._prep.ordinal_encoder(cirrhosis_df, ["Stage"])
        cirrhosis_df = self._prep.one_hot_encode(cirrhosis_df, ["Drug"]).dropna()

        return cirrhosis_df

    # ------------------------------------------------------------------------
    def pre_processing_glioma(self):
        return self.datasets["TCGA_InfoWithGrade"].copy()

    # ------------------------------------------------------------------------
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [
            self.acute,
            self.adult,
            self.autism_teen,
            self.autism_adult,
            self.autism_child,
            self.bank,
            self.blood_transfusion,
            self.bc_coimbra,
            self.wiscosin,
            self.cirohis,
            self.compass_4k,
            self.compass_7k,
            self.contraceptive,
            self.diabetic,
            self.diabetes,
            self.dutch,
            self.echocardiogram,
            self.fertility,
            self.german_credit,
            self.haberman,
            self.hcv,
            self.cleveland,
            self.hepatitis,
            self.iris,
            self.law,
            self.liver,
            self.mathernal_risk,
            self.mammo_masses,
            self.npha,
            self.parkinsons,
            self.phoneme,
            self.pima,
            self.prob_football,
            self.ricci,
            self.sa_heart,
            self.student_math,
            self.student_port,
            self.thoracic_surgery,
            self.thyroid,
            self.wine,
            self.obesity,
            self.lymphography,
            self.mushroom,
            self.nursery,
            self.heart_failure,
            self.breast_tissue,
            self.monk_um,
            self.monk_dois,
            self.monk_tres,
            self.glioma,
        ]

        tabela_resultados["nome_datasets"] = [
            "acute",
            "adult",
            "autism_teen",
            "autism_adult",
            "autism_child",
            "bank",
            "blood_transfusion",
            "bc_coimbra",
            "bc_wiscosin",
            "cirohis",
            "compass_4k",
            "compass_7k",
            "contraceptive",
            "diabetic",
            "diabetes",
            "dutch",
            "echocardiogram",
            "fertility",
            "german_credit",
            "haberman",
            "hcv",
            "cleveland",
            "hepatitis",
            "iris",
            "law",
            "liver",
            "mathernal_risk",
            "mammo_masses",
            "npha",
            "parkinsons",
            "phoneme",
            "pima",
            "prob_football",
            "ricci",
            "sa_heart",
            "student_math",
            "student_port",
            "thoracic_surgery",
            "thyroid",
            "wine",
            "obesity",
            "lymphography",
            "mushroom",
            "nursery",
            "heart_failure",
            "breast_tissue",
            "monk_um",
            "monk_dois",
            "monk_tres",
            "glioma",
        ]

        tabela_resultados["missing_rate"] = [5, 10, 20, 30, 40, 50]

        return tabela_resultados
