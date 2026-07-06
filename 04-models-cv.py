"""
Stage 4 — Six classifiers, StratifiedGroupKFold (k=3) by State, plus
chronological robustness split. Reproduces Tables 9 and 10.

Input : data/dataset_modeling.csv
Output: outputs/cv_performance.csv, outputs/chronological_split.csv
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

SEED = 42

FIN = ["Log_TotalAssets","Leverage_pct","CR","C_TA_pct","EBITDA_TA_pct","WC_TA_pct",
       "GR_pct","TS_TA_pct","Reserves_and_Funds","Firm_Age","ICR_filled","Has_Debt"]
LOCAL = ["Temp_Mean","Temp_Max","Heatwave_Days","Precip_Total","Heavy_Rain_Days","Dry_Days",
         "Wind_Mean","High_Wind_Days","Humidity_Mean","Humidity_Min","Low_Humidity_Days"]
NATL = ["National_Temp_Mean","National_Precip_Mean","National_Heatwave_Days_Avg"]

def make_models():
    return {
      "Logistic Regression": LogisticRegression(max_iter=300, class_weight="balanced"),
      "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8,
                        class_weight="balanced", random_state=SEED, n_jobs=-1),
      "ERT": ExtraTreesClassifier(n_estimators=150, max_depth=8,
                        class_weight="balanced", random_state=SEED, n_jobs=-1),
      "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=4,
                        learning_rate=0.08, random_state=SEED),
      "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                        eval_metric="aucpr", random_state=SEED, n_jobs=-1),
      "Deep Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                        max_iter=500, random_state=SEED, early_stopping=True),
    }

def resample(Xtr, ytr):
    Xu, yu = RandomUnderSampler(sampling_strategy=0.3, random_state=SEED).fit_resample(Xtr, ytr)
    return SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xu, yu)

def main():
    df = pd.read_csv("data/dataset_modeling.csv")
    own = pd.get_dummies(df["Ownership_Simple"], prefix="Own", drop_first=True)
    df = pd.concat([df, own], axis=1)
    feats = FIN + LOCAL + NATL + list(own.columns)      # 29-feature primary spec
    X, y = df[feats].astype(float).values, df["Liquidation_Flag"].values
    groups = df["State"].fillna("Unknown").values

    # --- grouped 3-fold CV (Table 9) ---
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for name in make_models():
        aucs, prs, f1s = [], [], []
        for tr, te in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xs, ys = resample(sc.transform(X[tr]), y[tr])
            m = make_models()[name].fit(Xs, ys)
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            prs.append(average_precision_score(y[te], p))
            f1s.append(f1_score(y[te], (p > 0.5).astype(int)))
        rows.append({"Model": name,
                     "AUC_ROC": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                     "AUC_PR":  f"{np.mean(prs):.3f} ± {np.std(prs):.3f}",
                     "F1":      f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("outputs/cv_performance.csv", index=False)

    # --- chronological split (Table 10) ---
    tr, te = df.FY_Year <= 2022, df.FY_Year >= 2023
    sc = StandardScaler().fit(X[tr.values])
    Xs, ys = resample(sc.transform(X[tr.values]), y[tr.values])
    rows = []
    for name, m in make_models().items():
        m.fit(Xs, ys)
        p = m.predict_proba(sc.transform(X[te.values]))[:, 1]
        rows.append({"Model": name,
                     "AUC_ROC": roc_auc_score(y[te.values], p),
                     "AUC_PR": average_precision_score(y[te.values], p),
                     "F1": f1_score(y[te.values], (p > 0.5).astype(int))})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("outputs/chronological_split.csv", index=False)

if __name__ == "__main__":
    main()
