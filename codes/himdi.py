from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def semantic_inconsistency(X_train, X_hat, missing_mask, 
                            rho_threshold=0.7, delta_multiplier=1.0):
    si_per_feature = {}
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    
    # Calcula correlações de Spearman
    corr_matrix = X_train[numeric_cols].corr(method='spearman').abs()
    
    for col in numeric_cols:
        imputed_idx = missing_mask[col]
        if imputed_idx.sum() == 0:
            continue
        
        # Features fortemente correlacionadas
        strong_partners = corr_matrix[col][
            (corr_matrix[col] > rho_threshold) & 
            (corr_matrix[col].index != col)
        ].index.tolist()
        
        if not strong_partners:
            si_per_feature[col] = 0.0
            continue
        
        violations_per_partner = []
        
        for partner in strong_partners:
            # Treina regressão simples 
            train_data = X_train[[partner, col]].dropna()
            reg = LinearRegression()
            reg.fit(train_data[[partner]], train_data[col])
            
            # Resíduo padrão do treino
            residuals = train_data[col] - reg.predict(train_data[[partner]])
            delta = delta_multiplier * residuals.std()
            
            # Avalia nos imputados
            valid_idx = imputed_idx & ~missing_mask.get(partner, 
                                       pd.Series(False, index=missing_mask.index))
            
            if valid_idx.sum() == 0:
                continue
                
            x_partner = X_hat.loc[valid_idx, [partner]]
            expected = reg.predict(x_partner)
            actual_imputed = X_hat.loc[valid_idx, col].values
            
            violations = (np.abs(actual_imputed - expected) > delta).mean()
            violations_per_partner.append(violations)
        
        si_per_feature[col] = np.mean(violations_per_partner) if violations_per_partner else 0.0
    
    return np.mean(list(si_per_feature.values()))