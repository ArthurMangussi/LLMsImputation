# LLMsImputation
Pergunta Gemini: Por favor, me informe 10 datasets tabulares para classificação na área da saúde que possuem entre 1k e 15k de observações, variando o tipo de colunas (categóricas e contínuas).

| # | Nome do Dataset | Área da Saúde | Observações (aprox.) | Colunas (Mistura) | Link de Referência (Exemplo) |
|---|------------------|----------------|----------------------|------------------|-------------------------------|
| 1 | Doença Cardíaca (Heart Disease) | Cardiologia | ~300 - 1000* | Contínuas e Categóricas | [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease) |
| 2 | Diabetes Pima Indians | Endocrinologia | 768 | Contínuas | [UCI Pima Indians Diabetes](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) |
| 3 | Câncer de Mama de Wisconsin (Original/Prognóstico) | Oncologia | 699 | Contínuas | [UCI Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)) |
| 4 | Doença Renal Crônica (Chronic Kidney Disease - CKD) | Nefrologia | 400 | Contínuas e Categóricas | [UCI Chronic Kidney Disease](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) |
| 5 | Parkinson's Disease Classification | Neurologia | ~195 | Contínuas | [UCI Parkinson's](https://archive.ics.uci.edu/ml/datasets/parkinsons) |
| 6 | Cleveland Heart Disease | Cardiologia | 303 | Contínuas e Categóricas | [UCI Heart Disease - Cleveland Subset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) |
| 7 | Vertebral Column Data Set (Hernia/Spondylolisthesis) | Ortopedia | 310 | Contínuas | [UCI Vertebral Column](https://archive.ics.uci.edu/ml/datasets/vertebral+column) |
| 8 | Breast Cancer (Original) | Oncologia | 286 | Categóricas | [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer) |
| 9 | Wine Quality (Classificação) | Saúde/Nutrição | ~1600 | Contínuas | [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| 10 | Hepatite (Hepatitis) | Gastroenterologia | 155 | Contínuas e Categóricas | [UCI Hepatitis](https://archive.ics.uci.edu/ml/datasets/hepatitis) |

ChatGPT:
| # | Nome do Dataset | Área da Saúde | Observações (aprox.) | Colunas (Mistura) | Tipo de Tarefa | Link de Referência |
|---|------------------|----------------|----------------------|------------------|----------------|--------------------|
| 1 | Breast Cancer Coimbra Dataset | Oncologia | 1.160 | Contínuas | Binária (Maligno/Benigno) | [UCI Breast Cancer Coimbra](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra) |
| 2 | Stroke Prediction Dataset | Neurologia | 5.110 | Contínuas e Categóricas | Binária (AVC/Não AVC) | [Kaggle Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| 3 | Heart Failure Clinical Records | Cardiologia | 2.999 | Contínuas e Categóricas | Binária (Óbito/Não Óbito) | [UCI Heart Failure](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) |
| 4 | Cervical Cancer Risk (Risk Factors) | Oncologia | 858 - 1.000* | Contínuas e Categóricas | Binária (Câncer/Não) | [UCI Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29) |
| 5 | Diabetic Retinopathy Debrecen | Oftalmologia | 1.151 | Contínuas | Binária (Presença/Ausência) | [UCI Diabetic Retinopathy](https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set) |
| 6 | Parkinson’s Telemonitoring | Neurologia | 5.875 | Contínuas | Regressão/Multiclasse (gravidade) | [UCI Parkinson’s Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) |
| 7 | COVID-19 Clinical Data (Mexico) | Epidemiologia | ~5.600 | Contínuas e Categóricas | Binária (Sobreviveu/Não) | [Kaggle COVID-19 Mexico](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) |
| 8 | Diabetes 130-US Hospitals | Endocrinologia | 10.000 | Contínuas e Categóricas | Binária (Readmissão/Não) | [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) |
| 9 | Hepatitis C Virus (HCV) Data | Gastroenterologia | 615 - 1.600* | Contínuas e Categóricas | Multiclasse (Fase da doença) | [UCI Hepatitis C Virus Data Set](https://archive.ics.uci.edu/ml/datasets/HCV+data) |
| 10 | Blood Transfusion Service Center | Hematologia | 7.48k | Contínuas | Binária (Doou/Não doou) | [UCI Blood Transfusion Service Center](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center) |
