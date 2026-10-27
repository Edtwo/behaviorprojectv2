#!/usr/bin/env python3
# FirstWords - split-half reliability of the words-only screen (run from repo root: .venv/bin/python src/ext_reliability.py)
# Split-half reliability: score each held-out ENNI child on odd vs even utterances (and
# first vs second half) with a model trained on other children; compare the two scores.
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'src')
import export_web_model as ew, stage2_level_estimator as s2
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, cohen_kappa_score, roc_auc_score
df=pd.read_csv('results/features.csv'); en=df[(df.corpus=='ENNI')&df.age_months.notna()].copy()
en['utts']=[ew.cha_utterances(open(os.path.join('data/childes',f),encoding='utf-8',errors='ignore').read()) for f in en.file]
en=en[en.utts.map(len)>=20].reset_index(drop=True); y=(en.group=='SLI').astype(int).to_numpy(); grp=en.child_id.to_numpy()
def F(utts,a): ws=[w for u in utts for w in u]; return [np.mean([len(u) for u in utts]), s2.mattr(ws), a]
full=np.array([F(u,a) for u,a in zip(en.utts,en.age_months)])
lr=lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
for label,split in [("odd vs even utterances",lambda u:(u[0::2],u[1::2])),("first vs second half",lambda u:(u[:len(u)//2],u[len(u)//2:]))]:
    A=np.array([F(split(u)[0],a) for u,a in zip(en.utts,en.age_months)]); B=np.array([F(split(u)[1],a) for u,a in zip(en.utts,en.age_months)])
    pa,pb,pf=np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(full,y,grp):
        m=lr().fit(full[tr],y[tr]); pa[te]=m.predict_proba(A[te])[:,1]; pb[te]=m.predict_proba(B[te])[:,1]; pf[te]=m.predict_proba(full[te])[:,1]
    fpr,tpr,thr=roc_curve(y,pf); cut=thr[np.argmax(tpr>=.9)]
    r=spearmanr(pa,pb).correlation; sb=2*r/(1+r)
    fa,fb=pa>=cut,pb>=cut
    print(f"{label}: n={len(y)} children (>=20 utterances), median {int(np.median(en.utts.map(len)))} utterances")
    print(f"  risk-score agreement between halves: Spearman r = {r:.2f}; full-length reliability (Spearman-Brown) = {sb:.2f}")
    print(f"  same flag decision on both halves: {np.mean(fa==fb):.0%} of children, Cohen's kappa = {cohen_kappa_score(fa,fb):.2f}")
    print(f"  AUC of each half alone: {roc_auc_score(y,pa):.2f} / {roc_auc_score(y,pb):.2f}")
