"""
Stage 11 - Calibration curves (reliability diagrams) for Random Forest and
XGBoost on the chronological test partition. Reproduces Figure A1 in the
Appendix and the Matthews correlation coefficients (MCC) reported in
Section 4.6 (Random Forest 0.688, XGBoost 0.594).

Input : data/dataset_modeling.csv
Output: outputs/figureA1_calibration_curve.png
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, os
_spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(__file__), "04-models-cv.py"))
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

SEED = 42

def main():
    df = pd.read_csv("data/dataset_modeling.csv")
    own = pd.get_dummies(df["Ownership_Simple"], prefix="Own", drop_first=True)
    df = pd.concat([df, own], axis=1)
    feats = m4.FIN + m4.LOCAL + m4.NATL
    X = df[feats].astype(float).values
    y = df["Liquidation_Flag"].values
    tr, te = (df.FY_Year <= 2022).values, (df.FY_Year >= 2023).values

    sc = StandardScaler().fit(X[tr])
    Xs, ys = m4.resample(sc.transform(X[tr]), y[tr])

    plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20,
                          "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16})
    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8,
                          class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                          eval_metric="aucpr", random_state=SEED, n_jobs=-1),
    }
    colors = {"Random Forest": "#1f4e8c", "XGBoost": "#c0622a"}
    markers = {"Random Forest": "o", "XGBoost": "s"}

    for name, m in models.items():
        m.fit(Xs, ys)
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        mcc = matthews_corrcoef(y[te], (p > 0.5).astype(int))
        frac, mean_pred = calibration_curve(y[te], p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker=markers[name], markersize=11, linewidth=3,
                color=colors[name], label=f"{name} (MCC = {mcc:.3f})")
        print(f"{name}: MCC = {mcc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=2.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=20, labelpad=12)
    ax.set_ylabel("Observed liquidation fraction", fontsize=20, labelpad=12)
    ax.legend(fontsize=16, loc="upper left", frameon=True, framealpha=0.95)
    ax.grid(alpha=0.3, linewidth=1)
    plt.tight_layout(pad=1.5)
    plt.savefig("outputs/figureA1_calibration_curve.png", dpi=200, bbox_inches="tight")
    print("Figure A1 saved")

if __name__ == "__main__":
    main()
