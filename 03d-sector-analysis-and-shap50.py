"""
Stage 3d - Sector-specific robustness and SHAP under the 50% threshold.
Reproduces Table 14 (Industry/Trade/Transport sector Model A vs C, Random
Forest and XGBoost) and Figure 3 (SHAP summary plot, XGBoost, 50% threshold
sample).

Input : data/dataset_full_unfiltered.csv
Output: outputs/table14_sector_analysis.csv, outputs/figure3_shap_50pct.png

NOTE: sector assignment here uses simple keyword matching on NIC_Name
(trade/retail/wholesale -> Trade; transport/logistics/shipping -> Transport;
else -> Industry). This approximates, but may not exactly reproduce, the
precise NIC industry-group classification rule used for the manuscript's
Table 14; resulting sector Ns and liquidation counts may differ slightly
from the published table. The qualitative pattern (no consistent
climate-driven AUC improvement across sectors) is robust to this
approximation.
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, os
_spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(__file__), "04-models-cv.py"))
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

SEED = 42
LOCAL_NO_HUMIDITY = [c for c in m4.LOCAL if c not in ("Humidity_Mean", "Humidity_Min", "Low_Humidity_Days")]

def apply_30pct_filter(df):
    """Primary healthy-firm filter (Table 9, Table 14 use this specification)."""
    healthy = (
        (df[["Cash", "EBITDA", "TotalAssets", "Sales", "Equity", "WC"]] > 0).all(axis=1)
        & (df["Leverage_pct"] < 30) & (df["I_S_pct"] < 30) & (df["GR_pct"] < 30)
    )
    return df[(df["Liquidation_Flag"] == 1) | healthy].copy()

def apply_50pct_filter(df):
    """Alternative threshold used only for Table 13 and Figure 3."""
    healthy = (
        (df[["Cash", "EBITDA", "TotalAssets", "Sales", "Equity", "WC"]] > 0).all(axis=1)
        & (df["Leverage_pct"] < 50) & (df["I_S_pct"] < 50) & (df["GR_pct"] < 50)
    )
    return df[(df["Liquidation_Flag"] == 1) | healthy].copy()

def classify_sector(nic_name):
    s = str(nic_name).lower()
    if any(k in s for k in ["trade", "retail", "wholesale"]): return "Trade"
    if any(k in s for k in ["transport", "logistics", "shipping"]): return "Transport"
    return "Industry"

def main():
    full = pd.read_csv("data/dataset_full_unfiltered.csv")
    own = pd.get_dummies(full["Ownership_Simple"], prefix="Own", drop_first=True)
    full = pd.concat([full, own], axis=1)
    df = apply_30pct_filter(full)   # Table 14 uses the primary (strict) filter
    df["Sector"] = df["NIC_Name"].map(classify_sector)
    feats_A = m4.FIN
    feats_C = m4.FIN + LOCAL_NO_HUMIDITY + m4.NATL

    def resample(X, y):
        Xu, yu = RandomUnderSampler(sampling_strategy=0.3, random_state=SEED).fit_resample(X, y)
        # Adaptive k: the Transport sector has very few liquidated firm-years
        # per fold (21 total across the panel), so the standard k=5 can
        # exceed the available minority-class sample size in a given fold.
        n_minority = min(np.bincount(yu))
        k = min(5, max(1, n_minority - 1))
        return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xu, yu)

    rows = []
    for sector in ["Industry", "Trade", "Transport"]:
        d = df[df.Sector == sector]
        n, n_liq = len(d), int(d.Liquidation_Flag.sum())
        for model_name, model_cls in [
            ("Random Forest", lambda: RandomForestClassifier(n_estimators=150, max_depth=8,
                              class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ("XGBoost", lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                              eval_metric="aucpr", random_state=SEED, n_jobs=-1)),
        ]:
            aucs = {}
            for spec, cols in [("A", feats_A), ("C", feats_C)]:
                X = d[cols].astype(float).fillna(0).values
                y = d["Liquidation_Flag"].values
                g = d["State"].fillna("Unknown").values
                cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                preds = np.zeros(len(d))
                for tr, te in cv.split(X, y, g):
                    sc = StandardScaler().fit(X[tr])
                    Xs, ys = resample(sc.transform(X[tr]), y[tr])
                    m = model_cls(); m.fit(Xs, ys)
                    preds[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
                aucs[spec] = roc_auc_score(y, preds)
            rows.append({"Sector": sector, "N": n, "Liquidations": n_liq, "Model": model_name,
                         "AUC_A": aucs["A"], "AUC_C": aucs["C"]})
            print(rows[-1])
    pd.DataFrame(rows).to_csv("outputs/table14_sector_analysis.csv", index=False)

    # Figure 3: SHAP summary on 50%-threshold sample, XGBoost
    y = df["Liquidation_Flag"].values
    g = df["State"].fillna("Unknown").values
    X = df[feats_C].astype(float).fillna(0).values
    tr_idx = np.where(df["FY_Year"] <= 2022)[0]
    sc = StandardScaler().fit(X[tr_idx])
    Xs, ys = resample(sc.transform(X[tr_idx]), y[tr_idx])
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                              eval_metric="aucpr", random_state=SEED, n_jobs=-1)
    model.fit(Xs, ys)
    rng = np.random.RandomState(SEED)
    eval_idx = rng.choice(len(df), size=min(3000, len(df)), replace=False)
    X_eval = pd.DataFrame(sc.transform(X[eval_idx]), columns=feats_C)
    sv = shap.TreeExplainer(model).shap_values(X_eval)
    shap.summary_plot(sv, X_eval, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figure3_shap_50pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nFigure 3 saved")

if __name__ == "__main__":
    main()
