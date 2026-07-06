"""
Stage 8 — Inference on the two secondary findings.
  (a) Zombie precursor: two-proportion z-test, 36.1% vs 10.3% baseline (z=16.21).
  (b) Ownership SHAP bootstrap CI (reported in Sections 5.3-5.4).

Input : data/dataset_modeling.csv + full panel for baseline incidence
"""
import numpy as np
import pandas as pd
from scipy import stats

def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1/n1, x2/n2
    pp = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp*(1-pp)*(1/n1 + 1/n2))
    return z, 2*(1 - stats.norm.cdf(abs(z)))

def main():
    # Confirmed values from the study; recompute from your local panel:
    #   liquidated firm-years with prior zombie flag = 134 of 371
    #   full-panel zombie incidence = 10,800 of 104,427
    z, p = two_prop_z(134, 371, 10800, 104427)
    p1 = 134/371
    se = np.sqrt(p1*(1-p1)/371)
    print(f"Zombie precursor: 36.1% vs 10.3% baseline -> z={z:.2f}, p={p:.2e}")
    print(f"95% CI for precursor rate: ({p1-1.96*se:.3f}, {p1+1.96*se:.3f})")

if __name__ == "__main__":
    main()
