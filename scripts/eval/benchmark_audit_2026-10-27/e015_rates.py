# Threshold metrics for the E-015 sealed FakeAVCeleb predictions at the frozen thresholds (visual 0.6138830927259571, audio 0.6954489212423093),
# from data/_results/e015/predictions_test_x.csv and labels derived from the manifest category. No model run.
import csv, re, json, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
R=Path(__file__).resolve().parents[3].as_posix()+'/'  # repository root (an absolute D: path until 2026-10-28)
res=json.load(open(R+'data/_results/e015/e015_results.json'))
tv=res['thresholds']['visual']; ta=res['thresholds']['audio']
cat={}
for r in csv.DictReader(open(R+'data/_manifests/fakeavceleb.manifest.csv',newline='')):
    m=re.search(r'category=(\w+)',r['notes']); cat[r['sample_id']]=m.group(1)
V=[];A=[]
for r in csv.DictReader(open(R+'data/_results/e015/predictions_test_x.csv',newline='')):
    c=cat[r['sample_id']]
    vlab=1 if c in('FVRA','FVFA') else 0; alab=1 if c in ('RVFA','FVFA') else 0
    V.append((vlab,float(r['visual_score']))); A.append((alab,float(r['audio_score'])))
def rates(D,t,name):
    y=np.array([d[0] for d in D]); s=np.array([d[1] for d in D]); p=s>=t
    tp=int((p&(y==1)).sum()); fn=int(((~p)&(y==1)).sum()); fp=int((p&(y==0)).sum()); tn=int(((~p)&(y==0)).sum())
    print(name,'n=%d fake=%d real=%d'%(len(y),y.sum(),(y==0).sum()),'AUC=%.4f'%roc_auc_score(y,s),'TP',tp,'FN',fn,'FP',fp,'TN',tn,
      'TPR=%.4f FPR=%.4f precision=%.4f accuracy=%.4f'%(tp/(tp+fn),fp/(fp+tn),tp/(tp+fp),(tp+tn)/len(y)))
print('predictions sha256',hashlib.sha256(open(R+'data/_results/e015/predictions_test_x.csv','rb').read()).hexdigest())
rates(V,tv,'visual (real video 0 vs fake video 1)'); rates(A,ta,'audio (real audio 0 vs fake audio 1)')
