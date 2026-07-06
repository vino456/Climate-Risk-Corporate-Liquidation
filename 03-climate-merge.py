"""
Stage 3 — Download NASA POWER daily climate data and build annual firm-level
climate variables; merge onto the labeled panel; apply the healthy-firm filter.

Inputs : data/panel_with_labels.csv, data/pincode_coordinates.csv
Output : data/dataset_modeling.csv (30,038 firm-years expected: 29,667 + 371)

NASA POWER: https://power.larc.nasa.gov/api/temporal/daily/point
Parameters: T2M, T2M_MAX, RH2M, PRECTOTCORR, WS2M (MERRA-2, 0.5x0.625 deg).
"""
import numpy as np
import pandas as pd
import requests, time

BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,T2M_MAX,RH2M,PRECTOTCORR,WS2M"

def fetch_point(lat, lon, start, end):
    r = requests.get(BASE, params={
        "parameters": PARAMS, "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": start, "end": end, "format": "JSON"}, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json()["properties"]["parameter"])

def annualize(d: pd.DataFrame) -> pd.DataFrame:
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    g = d.groupby(d.index.year)
    out = pd.DataFrame({
        "Temp_Mean": g["T2M"].mean(), "Temp_Max": g["T2M_MAX"].max(),
        "Heatwave_Days": g.apply(lambda x: (x["T2M"] > 35).sum()),
        "Humidity_Mean": g["RH2M"].mean(), "Humidity_Min": g["RH2M"].min(),
        "Low_Humidity_Days": g.apply(lambda x: (x["RH2M"] < 40).sum()),
        "Precip_Total": g["PRECTOTCORR"].sum(),
        "Heavy_Rain_Days": g.apply(lambda x: (x["PRECTOTCORR"] > 50).sum()),
        "Dry_Days": g.apply(lambda x: (x["PRECTOTCORR"] < 1).sum()),
        "Wind_Mean": g["WS2M"].mean(),
        "High_Wind_Days": g.apply(lambda x: (x["WS2M"] > 10).sum()),
    })
    return out.rename_axis("FY_Year").reset_index()

def main():
    panel = pd.read_csv("data/panel_with_labels.csv")
    coords = pd.read_csv("data/pincode_coordinates.csv")          # pincode -> lat, lon
    panel = panel.merge(coords, on="Pincode", how="left")         # 98.56% match rate

    cells = panel[["lat", "lon"]].round(1).drop_duplicates().dropna()
    frames = []
    for _, r in cells.iterrows():
        a = annualize(fetch_point(r.lat, r.lon, "20130401", "20260331"))
        a["lat"], a["lon"] = r.lat, r.lon
        frames.append(a); time.sleep(1)
    clim = pd.concat(frames)
    panel["lat"], panel["lon"] = panel["lat"].round(1), panel["lon"].round(1)
    df = panel.merge(clim, on=["lat", "lon", "FY_Year"], how="left")

    # National (sample-average) variables
    nat = df.groupby("FY_Year").agg(National_Temp_Mean=("Temp_Mean", "mean"),
                                    National_Precip_Mean=("Precip_Total", "mean"),
                                    National_Heatwave_Days_Avg=("Heatwave_Days", "mean")).reset_index()
    df = df.merge(nat, on="FY_Year")

    # Healthy-firm filter (30% thresholds); liquidated rows kept unconditionally
    healthy = ((df[["Cash", "EBITDA", "TotalAssets", "Sales", "Equity", "WC"]] > 0).all(axis=1)
               & (df["Leverage_pct"] < 30) & (df["I_S_pct"] < 30) & (df["GR_pct"] < 30))
    model_df = df[(df["Liquidation_Flag"] == 1) | healthy].copy()
    model_df.to_csv("data/dataset_modeling.csv", index=False)
    print(f"Modeling sample: {len(model_df)} ({model_df.Liquidation_Flag.sum()} liquidations)")

if __name__ == "__main__":
    main()
