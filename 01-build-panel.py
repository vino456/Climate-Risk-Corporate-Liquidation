"""
Stage 1 — Build the Prowess firm-year panel.

Prowess extraction settings (Query Builder):
  - Universe: all non-financial Indian companies
  - Template: Standardised Annual Financials (Ind AS), Advanced, standalone mode
  - Fields: total assets, total debt, current assets/liabilities, cash, EBITDA,
    working capital, equity, sales, interest expense, reserves & surplus,
    incorporation year, registered State, pincode, NIC code/name, ownership group
  - Period: FY2014-FY2026; output: AP (annual panel), long format

Input : data/prowess_raw_*.csv  (one or more Prowess export files)
Output: data/prowess_panel.csv  (104,427 firm-years x 14,299 firms expected)
"""
import glob
import numpy as np
import pandas as pd

SEED = 42

def main():
    frames = [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob("data/prowess_raw_*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Company", "FY_Year"])

    # Derived ratios (percentages)
    df["Leverage_pct"]   = 100 * df["TotalDebt"] / df["TotalAssets"]
    df["CR"]             = df["CurrentAssets"] / df["CurrentLiabilities"]
    df["C_TA_pct"]       = 100 * df["Cash"] / df["TotalAssets"]
    df["EBITDA_TA_pct"]  = 100 * df["EBITDA"] / df["TotalAssets"]
    df["WC_TA_pct"]      = 100 * df["WC"] / df["TotalAssets"]
    df["GR_pct"]         = 100 * (df["TotalDebt"] - df["Cash"]) / df["Equity"]
    df["TS_TA_pct"]      = 100 * df["Sales"] / df["TotalAssets"]
    df["I_S_pct"]        = 100 * df["Interest_Expense"] / df["Sales"]
    df["Log_TotalAssets"] = np.log(df["TotalAssets"].clip(lower=1e-2))
    df["Firm_Age"]       = df["FY_Year"] - df["Incorporation_Year"]

    # Interest coverage: fixed 999 for negligible interest expense + Has_Debt flag
    negligible = df["Interest_Expense"].fillna(0) < 0.01
    df["Has_Debt"]   = (~negligible).astype(int)
    df["ICR"]        = df["EBITDA"] / df["Interest_Expense"].where(~negligible)
    df["ICR_filled"] = df["ICR"].fillna(999.0)

    # Data-irregularity exclusion: negative leverage or negative interest-to-sales
    bad = (df["Leverage_pct"] < 0) | (df["I_S_pct"] < 0)
    df = df[~bad].copy()

    df.to_csv("data/prowess_panel.csv", index=False)
    print(f"Panel: {df.Company.nunique()} firms, {len(df)} firm-years")

if __name__ == "__main__":
    main()
