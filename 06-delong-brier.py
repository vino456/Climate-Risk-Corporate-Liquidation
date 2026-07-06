"""
Stage 6 — DeLong paired AUC tests + Brier calibration on pooled out-of-fold
predictions (grouped 3-fold CV). Reproduces Table 12. This is the exact script
run for the published values.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from scipy import stats

# ---------- Fast DeLong implementation ----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_roc_variance(y_true, preds):
    # preds: (k, n) predictions of k classifiers on same n samples
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = preds[:, order]
    m, n = label_1_count, preds.shape[1]-label_1_count
    k = preds.shape[0]
    tx, ty, tz = [np.empty([k, s]) for s in [m, n, m+n]]
    for r in range(k):
        tx[r] = compute_midrank(preds[r, :m]); ty[r] = compute_midrank(preds[r, m:]); tz[r] = compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1)/(m*n) - (m+1.0)/(2.0*n)
    v01 = (tz[:, :m]-tx)/n; v10 = 1.0-(tz[:, m:]-ty)/m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx/m + sy/n

def delong_test(y_true, p1, p2):
    aucs, cov = delong_roc_variance(np.array(y_true), np.vstack([p1, p2]))
    diff = aucs[0]-aucs[1]
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    z = diff/np.sqrt(var) if var > 0 else 0.0
    p = 2*(1-stats.norm.cdf(abs(z)))
    return aucs, z, p

# ---------- Data & feature sets ----------
df = pd.read_csv('data/dataset_modeling.csv')
nt = df[['FY_Year','National_Temp_Mean']].drop_duplicates().sort_values('FY_Year')
nt['ASTC_equiv'] = nt['National_Temp_Mean'].diff().fillna(0)
df = df.merge(nt[['FY_Year','ASTC_equiv']], on='FY_Year', how='left')
own = pd.get_dummies(df['Ownership_Simple'], prefix='Own', drop_first=True)
df = pd.concat([df, own], axis=1)

fin = ['Log_TotalAssets','Leverage_pct','CR','C_TA_pct','EBITDA_TA_pct','WC_TA_pct','GR_pct','TS_TA_pct','Reserves_and_Funds','Firm_Age','ICR_filled','Has_Debt'] + list(own.columns)
natl = ['National_Temp_Mean','National_Precip_Mean','National_Heatwave_Days_Avg','ASTC_equiv']
local = ['Temp_Mean','Temp_Max','Heatwave_Days','Precip_Total','Heavy_Rain_Days','Dry_Days','Wind_Mean','High_Wind_Days','Humidity_Mean','Humidity_Min','Low_Humidity_Days']
SPECS = {'A': fin, 'B': fin+natl, 'C': fin+natl+local}
y = df['Liquidation_Flag'].values
groups = df['State'].fillna('Unknown').values

def models():
    return {
      'Logistic Regression': LogisticRegression(max_iter=300, class_weight='balanced'),
      'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
      'ERT': ExtraTreesClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
      'GBM': GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
      'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, eval_metric='aucpr', random_state=42, n_jobs=-1),
      'Deep Neural Network': MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', max_iter=500, random_state=42, early_stopping=True),
    }

cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
oof = {s: {m: np.zeros(len(df)) for m in models()} for s in SPECS}
for spec, cols in SPECS.items():
    X = df[cols].astype(float).values
    for tr, te in cv.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        Xu, yu = RandomUnderSampler(sampling_strategy=0.3, random_state=42).fit_resample(Xtr, y[tr])
        Xs, ys = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xu, yu)
        for name, m in models().items():
            m.fit(Xs, ys)
            oof[spec][name][te] = m.predict_proba(Xte)[:,1]
    print(f"Spec {spec} done", flush=True)

print("\nModel | AUC_A | AUC_B | AUC_C | DeLong p(B-A) | DeLong p(C-A) | Brier(C)")
rows=[]
for name in models():
    aucs_ab, z_ab, p_ab = delong_test(y, oof['B'][name], oof['A'][name])
    aucs_ca, z_ca, p_ca = delong_test(y, oof['C'][name], oof['A'][name])
    aucA = roc_auc_score(y, oof['A'][name]); aucB = roc_auc_score(y, oof['B'][name]); aucC = roc_auc_score(y, oof['C'][name])
    brier = brier_score_loss(y, oof['C'][name])
    print(f"{name}: {aucA:.4f} | {aucB:.4f} | {aucC:.4f} | p={p_ab:.3f} | p={p_ca:.3f} | Brier={brier:.4f}")
    rows.append({'Model':name,'AUC_A':aucA,'AUC_B':aucB,'AUC_C':aucC,'p_BA':p_ab,'p_CA':p_ca,'Brier_C':brier})
pd.DataFrame(rows).to_csv('outputs/delong_brier.csv', index=False)
print("SAVED")
