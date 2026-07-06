"""
Stage 3b - Descriptive statistics, VIF, and correlation analysis.
Reproduces Table 4 (VIF), Table 5 (correlations), Table 6a/6b, Table 7a/7b
(descriptive statistics), Figure 1 (correlogram), and Figure 2 (correlation
forest plot with 95% CIs).

Input : data/dataset_modeling.csv (output of 03_climate_merge.py)
Output: outputs/table4_vif.csv, outputs/table5_correlations.csv,
        outputs/table6_7_descriptives.csv, outputs/figure1_correlogram.png,
        outputs/figure2_forest_plot.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

FIN_VARS = ["WC_TA_pct", "Leverage_pct", "Log_TotalAssets", "Reserves_and_Funds",
            "C_TA_pct", "Firm_Age", "EBITDA_TA_pct", "TS_TA_pct", "ICR_filled", "CR", "GR_pct"]

ALL_PREDICTORS = FIN_VARS + ["National_Temp_Mean", "Temp_Mean", "Wind_Mean", "Precip_Total",
                             "Humidity_Mean", "National_Precip_Mean"]

def compute_vif(df):
    """Table 4. VIF MUST include a constant term or values are meaningless (classic gotcha)."""
    X = df[FIN_VARS].fillna(0)
    X_const = add_constant(X)
    vifs = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])]
    out = pd.DataFrame({"Variable": FIN_VARS, "VIF": vifs}).sort_values("VIF", ascending=False)
    out.to_csv("outputs/table4_vif.csv", index=False)
    print("Table 4 (VIF):\n", out.to_string(index=False))
    return out

def compute_correlations(df):
    """Table 5 + Figure 2. Pearson r and Fisher z-transformed 95% CI for each predictor vs Liquidation_Flag."""
    y = df["Liquidation_Flag"]
    rows = []
    for col in ALL_PREDICTORS:
        x = df[col]
        mask = x.notna()
        r, _ = stats.pearsonr(x[mask], y[mask])
        n = mask.sum()
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"Variable": col, "r": r, "CI_low": lo, "CI_high": hi, "n": n})
    out = pd.DataFrame(rows).sort_values("r", ascending=False)
    out.to_csv("outputs/table5_correlations.csv", index=False)
    print("\nTable 5 (correlations):\n", out.to_string(index=False))

    # Figure 2: forest plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    yy = np.arange(len(out))
    ax.errorbar(out["r"], yy, xerr=[out["r"] - out["CI_low"], out["CI_high"] - out["r"]],
                fmt="o", color="#1f4e8c", capsize=4)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(yy); ax.set_yticklabels(out["Variable"])
    ax.set_xlabel("Pearson correlation with Liquidation_Flag (95% CI)")
    plt.tight_layout()
    plt.savefig("outputs/figure2_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    return out

def correlogram(df):
    """Figure 1. Full correlation matrix heatmap of predictors + Liquidation_Flag."""
    cols = ALL_PREDICTORS + ["Liquidation_Flag"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("outputs/figure1_correlogram.png", dpi=150, bbox_inches="tight")
    plt.close()

def descriptive_stats(df):
    """Table 6a/6b (raw monetary vars) and Table 7a/7b (ratio vars), split by Liquidation_Flag."""
    monetary = ["TotalAssets", "Equity", "Cash", "EBITDA", "WC", "Sales"]
    ratios = FIN_VARS + ["Temp_Mean", "Humidity_Mean", "Precip_Total", "Wind_Mean",
                         "National_Temp_Mean", "National_Precip_Mean"]
    frames = []
    for flag, label in [(1, "Liquidated"), (0, "Healthy")]:
        sub = df[df.Liquidation_Flag == flag]
        for group_name, cols in [("monetary", monetary), ("ratio", ratios)]:
            desc = sub[cols].agg(["mean", "std", "median", "min", "max"]).T
            desc["Group"] = label; desc["Type"] = group_name
            frames.append(desc)
    out = pd.concat(frames)
    out.to_csv("outputs/table6_7_descriptives.csv")
    print(f"\nDescriptives saved: {out.shape[0]} rows")

def main():
    df = pd.read_csv("data/dataset_modeling.csv")
    print(f"Loaded {len(df)} firm-years ({df.Liquidation_Flag.sum()} liquidations)\n")
    compute_vif(df)
    compute_correlations(df)
    correlogram(df)
    descriptive_stats(df)
    print("\nAll descriptive/VIF/correlation outputs written to outputs/")

if __name__ == "__main__":
    main()
