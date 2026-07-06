"""
Stage 5 — Model A/B/C nested comparison with Hanley-McNeil paired AUC tests.
Reproduces Table 11.

Model A: financial + ownership | Model B: A + national climate (incl. ASTC_equiv)
Model C: B + local climate magnitudes.

Input : data/dataset_modeling.csv
Output: outputs/abc_hanley_mcneil.csv
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.metrics import roc_auc_score
# model/resampling helpers identical to 04_models_cv
import importlib.util, os
_spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(__file__), "04-models-cv.py"))
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

def hm_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    return np.sqrt((auc*(1-auc) + (n_pos-1)*(q1-auc**2) + (n_neg-1)*(q2-auc**2)) / (n_pos*n_neg))

def hm_test(auc1, auc2, n_pos, n_neg):
    z = (auc2 - auc1) / np.sqrt(hm_se(auc1, n_pos, n_neg)**2 + hm_se(auc2, n_pos, n_neg)**2)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

def main():
    df = pd.read_csv("data/dataset_modeling.csv")
    nt = df[["FY_Year", "National_Temp_Mean"]].drop_duplicates().sort_values("FY_Year")
    nt["ASTC_equiv"] = nt["National_Temp_Mean"].diff().fillna(0)
    df = df.merge(nt[["FY_Year", "ASTC_equiv"]], on="FY_Year")
    own = pd.get_dummies(df["Ownership_Simple"], prefix="Own", drop_first=True)
    df = pd.concat([df, own], axis=1)

    A = m4.FIN + list(own.columns)
    B = A + m4.NATL + ["ASTC_equiv"]
    C = B + m4.LOCAL
    y = df["Liquidation_Flag"].values
    tr, te = (df.FY_Year <= 2022).values, (df.FY_Year >= 2023).values
    n_pos, n_neg = int(y[te].sum()), int((1 - y[te]).sum())

    rows = []
    for name in m4.make_models():
        aucs = {}
        for spec, cols in {"A": A, "B": B, "C": C}.items():
            X = df[cols].astype(float).values
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[tr])
            Xs, ys = m4.resample(sc.transform(X[tr]), y[tr])
            mdl = m4.make_models()[name].fit(Xs, ys)
            aucs[spec] = roc_auc_score(y[te], mdl.predict_proba(sc.transform(X[te]))[:, 1])
        zba, pba = hm_test(aucs["A"], aucs["B"], n_pos, n_neg)
        zca, pca = hm_test(aucs["A"], aucs["C"], n_pos, n_neg)
        rows.append({"Model": name, **{f"AUC_{k}": v for k, v in aucs.items()},
                     "p_BA": pba, "p_CA": pca})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("outputs/abc_hanley_mcneil.csv", index=False)

if __name__ == "__main__":
    main()
