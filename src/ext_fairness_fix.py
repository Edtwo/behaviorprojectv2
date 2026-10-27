#!/usr/bin/env python3
# FirstWords - fairness fix test: does dropping vocabulary diversity shrink the ethnic false-alarm gap? (run from repo root)
# Result 2026-10-24: consistent ~7-10 point reduction, intervals borderline (exploratory, not a headline).
# Bias-fix test. Models trained on ENNI only. Gillam cutoffs chosen by 5-fold CROSS-FITTING
# (cutoff picked on 4/5 of Gillam children, applied to the held-out 1/5), so every reported
# Gillam number comes from children not used to set it. Target sensitivity pre-set at 80%.
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0,'src')
import ext_fairness_ethnicity as fe, ext_crosscorpus as cc, stage3_delay_classifier as s3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
rng=np.random.default_rng(11)
lr=lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
files={os.path.basename(f):f for f in glob.glob('data/childes/Gillam/**/*.cha',recursive=True)}
g=cc.extract_corpus('data/childes/Gillam'); g=g[g.group.isin(['TD','SLI'])].dropna(subset=['mluw','mlum','mattr50','age_months']).copy()
g['eth']=g.file.map(lambda b: fe.ethnicity(files[b])); g['y']=(g.group=='SLI').astype(int); g=g.reset_index(drop=True)
enni=s3.load_enni()
strata=g.eth+'|'+g.y.astype(str)
def crossfit_flags(p, target):
    flag=np.zeros(len(g),bool)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(g,strata):
        fpr,tpr,thr=roc_curve(g.y.iloc[tr],p[tr]); cut=thr[np.argmax(tpr>=target)]
        flag[te]=p[te]>=cut
    return flag
def report(name, p, flag):
    row=[f"{name:42s} AUC {roc_auc_score(g.y,p):.2f}  sens {flag[g.y==1].mean():.0%}  FA {flag[g.y==0].mean():.0%}"]
    fa={}
    for e in ['European American','African American','Hispanic American']:
        m=(g.eth==e)&(g.y==0); fa[e]=flag[m]; row.append(f"{e.split()[0][:3]} FA {flag[m].mean():.0%}")
    d=[rng.choice(fa['African American'],len(fa['African American'])).mean()-rng.choice(fa['European American'],len(fa['European American'])).mean() for _ in range(4000)]
    lo,hi=np.percentile(d,[2.5,97.5]); row.append(f"Black-White gap {fa['African American'].mean()-fa['European American'].mean():+.0%} [{lo:+.0%},{hi:+.0%}]")
    print("  ".join(row))
for target in [0.80, 0.90]:
    print(f"\n--- cutoff cross-fitted on Gillam for {target:.0%} sensitivity ---")
    for name,feats in [('words-only (length + vocabulary + age)',['mluw','mattr50','age_months']),
                       ('sentence length + age',['mluw','age_months']),
                       ('full (morphemes too)',['mluw','mlum','mattr50','age_months'])]:
        p=lr().fit(enni[feats],enni.y).predict_proba(g[feats])[:,1]
        report(name,p,crossfit_flags(p,target))

print("\n--- paired test: does dropping vocabulary shrink the gap? (same children, 4000 bootstraps) ---")
for target in [0.80, 0.90]:
    fw=crossfit_flags(lr().fit(enni[['mluw','mattr50','age_months']],enni.y).predict_proba(g[['mluw','mattr50','age_months']])[:,1],target)
    fl=crossfit_flags(lr().fit(enni[['mluw','age_months']],enni.y).predict_proba(g[['mluw','age_months']])[:,1],target)
    for grp in ['African American','Hispanic American']:
        A=np.where((g.eth==grp)&(g.y==0))[0]; E=np.where((g.eth=='European American')&(g.y==0))[0]
        d=[]
        for _ in range(4000):
            a=rng.choice(A,len(A)); e=rng.choice(E,len(E))
            d.append((fw[a].mean()-fw[e].mean())-(fl[a].mean()-fl[e].mean()))
        lo,hi=np.percentile(d,[2.5,97.5]); obs=(fw[A].mean()-fw[E].mean())-(fl[A].mean()-fl[E].mean())
        print(f"  {target:.0%} sens  {grp:18s} gap reduction {obs:+.0%} [{lo:+.0%}, {hi:+.0%}]  (positive = sentence-length screen is fairer)")
