"""
Stage 7 — SHAP explainability on the final XGBoost model (29-feature primary
spec) with 2,000-resample bootstrap for ranking stability.
Reproduces Table 15 and Figures 4-9.

Input : data/dataset_modeling.csv
Output: outputs/shap_table.csv, outputs/shap_bar.png, outputs/shap_beeswarm.png,
        outputs/waterfall_*.png
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, os
_spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(__file__), "04-models-cv.py"))
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

SEED, N_EVAL, N_BOOT = 42, 3000, 2000

def main():
    df = pd.read_csv("data/dataset_modeling.csv")
    own = pd.get_dummies(df["Ownership_Simple"], prefix="Own", drop_first=True)
    df = pd.concat([df, own], axis=1)
    feats = m4.FIN + m4.LOCAL + m4.NATL + list(own.columns)
    X, y = df[feats].astype(float), df["Liquidation_Flag"].values

    from sklearn.preprocessing import StandardScaler
    tr = (df.FY_Year <= 2022).values
    sc = StandardScaler().fit(X[tr])
    Xs, ys = m4.resample(sc.transform(X[tr]), y[tr])
    model = m4.make_models()["XGBoost"].fit(Xs, ys)

    rng = np.random.RandomState(SEED)
    ev_idx = rng.choice(np.where(~tr)[0], size=min(N_EVAL, (~tr).sum()), replace=False)
    X_eval = pd.DataFrame(sc.transform(X.iloc[ev_idx]), columns=feats)
    sv = shap.TreeExplainer(model).shap_values(X_eval)

    tab = (pd.DataFrame({"Feature": feats, "Mean_absSHAP": np.abs(sv).mean(axis=0)})
             .sort_values("Mean_absSHAP", ascending=False).reset_index(drop=True))
    tab.index += 1
    tab.to_csv("outputs/shap_table.csv", index_label="Rank")
    print(tab.head(10))

    # Bootstrap: ownership vs Temp_Mean ranking stability
    i_own, i_tmp = feats.index("Own_Private Indian") if "Own_Private Indian" in feats else feats.index("Own_Private_Indian"), feats.index("Temp_Mean")
    wins = sum(np.abs(sv[rng.choice(len(sv), len(sv))][:, i_own]).mean()
               > np.abs(sv[rng.choice(len(sv), len(sv))][:, i_tmp]).mean()
               for _ in range(N_BOOT))
    print(f"Ownership > Temp_Mean in {wins/N_BOOT:.1%} of {N_BOOT} bootstrap resamples")

    shap.summary_plot(sv, X_eval, plot_type="bar", show=False)
    plt.savefig("outputs/shap_bar.png", dpi=200, bbox_inches="tight"); plt.close()
    shap.summary_plot(sv, X_eval, show=False)
    plt.savefig("outputs/shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    main()
