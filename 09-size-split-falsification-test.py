import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from scipy import stats

SEED = 42
def compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong_test(y,p1,p2):
    y=np.array(y); order=np.argsort(-y); m=int(y.sum())
    preds=np.vstack([p1,p2])[:,order]; n=preds.shape[1]-m
    tx,ty,tz=[np.empty([2,s]) for s in [m,n,m+n]]
    for r in range(2):
        tx[r]=compute_midrank(preds[r,:m]); ty[r]=compute_midrank(preds[r,m:]); tz[r]=compute_midrank(preds[r,:])
    aucs=tz[:,:m].sum(axis=1)/(m*n)-(m+1.0)/(2.0*n)
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n
    diff=aucs[0]-aucs[1]; var=cov[0,0]+cov[1,1]-2*cov[0,1]
    z=diff/np.sqrt(var) if var>0 else 0
    return aucs, z, 2*(1-stats.norm.cdf(abs(z)))

df = pd.read_csv('dataset_c_engineered_with_humidity.csv')
own = pd.get_dummies(df['Ownership_Simple'], prefix='Own', drop_first=True)
df = pd.concat([df, own], axis=1)
fin = ['Log_TotalAssets','Leverage_pct','CR','C_TA_pct','EBITDA_TA_pct','WC_TA_pct','GR_pct','TS_TA_pct','Reserves_and_Funds','Firm_Age','ICR_filled','Has_Debt']+list(own.columns)
loc = ['Temp_Mean','Temp_Max','Heatwave_Days','Precip_Total','Heavy_Rain_Days','Dry_Days','Wind_Mean','High_Wind_Days','Humidity_Mean','Humidity_Min','Low_Humidity_Days']
nat = ['National_Temp_Mean','National_Precip_Mean','National_Heatwave_Days_Avg']

# TEST: if HQ/plant mismatch drives attenuation, LARGE firms (more likely multi-site)
# should show WEAKER climate contribution than SMALL firms (more likely single-site).
# If instead both halves show similarly weak/null climate effects, mismatch is not the driver.
med = df['Log_TotalAssets'].median()
large = df[df['Log_TotalAssets'] > med].copy()
small = df[df['Log_TotalAssets'] <= med].copy()

def resample(X,y):
    Xu,yu = RandomUnderSampler(sampling_strategy=0.3, random_state=SEED).fit_resample(X,y)
    return SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xu,yu)

for label, d in [('SMALL firms (single-site presumption)', small), ('LARGE firms (multi-site presumption)', large)]:
    y = d.Liquidation_Flag.values
    g = d['State'].fillna('U').values
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = {}
    for spec, cols in [('A', fin), ('C', fin+nat+loc)]:
        X = d[cols].astype(float).values
        p = np.zeros(len(d))
        for tr, te in cv.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            Xs, ys = resample(sc.transform(X[tr]), y[tr])
            m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, eval_metric='aucpr', random_state=SEED, n_jobs=-1)
            m.fit(Xs, ys)
            p[te] = m.predict_proba(sc.transform(X[te]))[:,1]
        oof[spec] = p
    aucs, z, pval = delong_test(y, oof['C'], oof['A'])
    print(f"{label}: n={len(d)}, liq={int(y.sum())}, AUC_A={aucs[1]:.4f}, AUC_C={aucs[0]:.4f}, diff={100*(aucs[0]-aucs[1]):+.2f}pp, DeLong p={pval:.3f}")
