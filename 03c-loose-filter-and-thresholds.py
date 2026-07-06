"""
Stage 3c - Alternative healthy-firm filter thresholds.
Reproduces Table 8 (loose zombie-exclusion filter, 3 classifiers: Logistic
Regression, Random Forest, XGBoost - the initial exploratory pass) and
Table 13 (50% leverage/interest-to-sales/gearing threshold, all classifiers).

Input : data/dataset_full_unfiltered.csv (output of 02_ibbi_matching.py,
        BEFORE the primary 30% healthy-firm filter is applied)
Output: outputs/table8_loose_filter.csv, outputs/table13_50pct_threshold.csv
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import importlib.util, os
_spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(__file__), "04-models-cv.py"))
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

SEED = 42
# The full unfiltered panel (data/dataset_full_unfiltered.csv) predates the
# humidity-variable merge applied in 03_climate_merge.py, so Humidity_Mean,
# Humidity_Min, and Low_Humidity_Days are unavailable here. This script uses
# the local-climate feature set excluding those three variables; all other
# scripts in this package use the full feature set including humidity.
LOCAL_NO_HUMIDITY = [c for c in m4.LOCAL if c not in ("Humidity_Mean", "Humidity_Min", "Low_Humidity_Days")]

def cv_performance(df, feats, models_dict):
    X = df[feats].astype(float).fillna(0).values
    y = df["Liquidation_Flag"].values
    g = df["State"].fillna("Unknown").values
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for name, make_model in models_dict.items():
        aucs, prs, f1s = [], [], []
        for tr, te in cv.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            Xs, ys = m4.resample(sc.transform(X[tr]), y[tr])
            model = make_model()
            model.fit(Xs, ys)
            p = model.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            prs.append(average_precision_score(y[te], p))
            f1s.append(f1_score(y[te], (p > 0.5).astype(int)))
        rows.append({"Model": name,
                     "AUC_ROC": f"{np.mean(aucs):.3f} \u00b1 {np.std(aucs):.3f}",
                     "AUC_PR": f"{np.mean(prs):.3f} \u00b1 {np.std(prs):.3f}",
                     "F1": f"{np.mean(f1s):.3f} \u00b1 {np.std(f1s):.3f}"})
        print(rows[-1])
    return pd.DataFrame(rows)

def apply_healthy_filter(df, threshold_pct):
    """Multi-criterion filter: positive cash/EBITDA/assets/sales/equity/WC,
    and leverage, interest-to-sales, gearing each below threshold_pct."""
    healthy = (
        (df[["Cash", "EBITDA", "TotalAssets", "Sales", "Equity", "WC"]] > 0).all(axis=1)
        & (df["Leverage_pct"] < threshold_pct)
        & (df["I_S_pct"] < threshold_pct)
        & (df["GR_pct"] < threshold_pct)
    )
    return df[(df["Liquidation_Flag"] == 1) | healthy].copy()

def apply_loose_zombie_filter(df):
    """Loose filter: exclude only firms meeting the two-year ICR<1 zombie
    criterion; no ratio thresholds. This was the initial exploratory pass
    before the stricter multi-criterion filter (Table 9) was finalized."""
    return df[(df["Liquidation_Flag"] == 1) | (df["Zombie_Flag"] == 0)].copy()

def main():
    full = pd.read_csv("data/dataset_full_unfiltered.csv")
    own = pd.get_dummies(full["Ownership_Simple"], prefix="Own", drop_first=True)
    full = pd.concat([full, own], axis=1)
    print(f"Full unfiltered panel: {len(full)} firm-years, {full.Liquidation_Flag.sum()} liquidations\n")

    # --- Table 8: loose filter, 3 classifiers only (exploratory pass) ---
    loose = apply_loose_zombie_filter(full)
    print(f"Loose-filter sample: {len(loose)} firm-years")
    models_3 = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=300, class_weight="balanced"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=8,
                          class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                          eval_metric="aucpr", random_state=SEED, n_jobs=-1),
    }
    t8 = cv_performance(loose, m4.FIN + LOCAL_NO_HUMIDITY + m4.NATL, models_3)
    t8.to_csv("outputs/table8_loose_filter.csv", index=False)

    # --- Table 13: 50% threshold, all 5 non-DNN classifiers ---
    thresh50 = apply_healthy_filter(full, 50)
    print(f"\n50%-threshold sample: {len(thresh50)} firm-years")
    models_5 = {k: v for k, v in m4.make_models().items() if k != "Deep Neural Network"}
    models_5_factories = {name: (lambda n=name: m4.make_models()[n]) for name in models_5}
    t13 = cv_performance(thresh50, m4.FIN + LOCAL_NO_HUMIDITY + m4.NATL, models_5_factories)
    t13.to_csv("outputs/table13_50pct_threshold.csv", index=False)

if __name__ == "__main__":
    main()
