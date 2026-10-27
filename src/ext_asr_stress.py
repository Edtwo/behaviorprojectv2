#!/usr/bin/env python3
# FirstWords - transcription-error stress test (run from repo root: .venv/bin/python src/ext_asr_stress.py)
# Stress test: how much speech-recognition-style error can the words-only screen absorb?
# ENNI, child-independent 5-fold: train on CLEAN transcripts, test on CORRUPTED held-out
# children. Corruptions: word error rate (half deletions, half substitutions with common
# words) and missed pauses (adjacent utterances merged).
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'src')
import export_web_model as ew, stage2_level_estimator as s2, stage3_delay_classifier as s3
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
df=pd.read_csv('results/features.csv'); en=df[(df.corpus=='ENNI')&df.age_months.notna()].copy()
en['utts']=[ew.cha_utterances(open(os.path.join('data/childes',f),encoding='utf-8',errors='ignore').read()) for f in en.file]
en=en[en.utts.map(len)>0].reset_index(drop=True); y=(en.group=='SLI').astype(int).to_numpy(); grp=en.child_id.to_numpy()
vocab=Counter(w for u in en.utts for x in u for w in x); words=np.array(list(vocab)); prob=np.array([vocab[w] for w in words],float); prob/=prob.sum()
def feats(utts): 
    ws=[w for u in utts for w in u]; return [np.mean([len(u) for u in utts]), s2.mattr(ws)]
def corrupt(utts,wer,merge,rng):
    out=[]
    for u in utts:
        v=[]
        for w in u:
            r=rng.random()
            if r<wer/2: continue
            v.append(rng.choice(words,p=prob) if r<wer else w)
        if not v: continue
        if out and rng.random()<merge: out[-1]=out[-1]+v
        else: out.append(v)
    return out or [['uh']]
clean=np.array([feats(u)+[a] for u,a in zip(en.utts,en.age_months)])
lr=lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(clean,y,grp))
print("AUC (words-only screen) by word error rate (rows) and share of pauses missed (columns), mean of 3 runs")
print("WER     " + "  ".join(f"merge {m:.0%}" for m in [0,.15,.30]))
rows=[]
for wer in [0,.10,.20,.30,.40]:
    row=[]
    for m in [0,.15,.30]:
        aucs=[]
        for seed in range(3):
            rng=np.random.default_rng(seed); X=np.array([feats(corrupt(u,wer,m,rng))+[a] for u,a in zip(en.utts,en.age_months)])
            p=np.zeros(len(y))
            for tr,te in folds: p[te]=lr().fit(clean[tr],y[tr]).predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y,p))
        rows.append({"wer":wer,"merge":m,"auc":float(np.mean(aucs))})
        row.append(f"{np.mean(aucs):.3f}   ")
    print(f"{wer:4.0%}    " + "  ".join(row))

pd.DataFrame(rows).to_csv("results/asr_stress.csv", index=False)
print("wrote results/asr_stress.csv")
