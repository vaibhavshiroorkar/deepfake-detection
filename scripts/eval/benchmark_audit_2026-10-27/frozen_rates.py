# Computes TPR, FPR, precision, balanced accuracy at the frozen (validation-selected, per-seed) thresholds
# from stored sealed score CSVs. Reads only data/_final/scores/<tag>.csv and data/_final/eval/<tag>.json. No model run.
import sys, json, hashlib, csv
import numpy as np
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[3].as_posix()+'/'  # repository root (an absolute D: path until 2026-10-28)
tag=sys.argv[1]; sets=sys.argv[2:]
ev=json.load(open(R+'data/_final/eval/%s.json'%tag))
thr={}
for c in ev['candidates']:
    for rid,f in c['fits'].items(): thr[rid]=f['threshold']
print(tag,'sha256(score csv)=',hashlib.sha256(open(R+'data/_final/scores/%s.csv'%tag,'rb').read()).hexdigest())
data=defaultdict(lambda:([],[]))  # (run_id,dataset,split)->(labels,scores)
with open(R+'data/_final/scores/%s.csv'%tag,newline='') as f:
    for r in csv.DictReader(f):
        if r['split']=='val': continue
        k=(r['run_id'],r['dataset'],r['split'])
        data[k][0].append(int(r['label'])); data[k][1].append(float(r['score']))
runs=sorted({k[0] for k in data})
def get(rid,ds,sp):
    if ds=='celebdf_family':
        a=data[(rid,'celebdf_v2',sp)];b=data[(rid,'celebdf_pp',sp)]
        return np.array(a[0]+b[0]),np.array(a[1]+b[1])
    if ds=='celebdf_pp_with_v2_real':
        a=data[(rid,'celebdf_v2',sp)];b=data[(rid,'celebdf_pp',sp)]
        ya=np.array(a[0]);sa=np.array(a[1])
        yb=np.array(b[0]);sb=np.array(b[1])
        return np.concatenate([ya[ya==0],yb[yb==1]]),np.concatenate([sa[ya==0],sb[yb==1]])
    a=data[(rid,ds,sp)]; return np.array(a[0]),np.array(a[1])
for s in sets:
    ds,sp=s.split(':'); out=[]
    for rid in runs:
        if rid not in thr: continue
        import os
        if os.environ.get('MOD') and ('-'+os.environ['MOD']+'-') not in rid: continue
        y,sc=get(rid,ds,sp)
        if len(y)==0: continue
        t=thr[rid]; p=sc>=t
        tp=(p&(y==1)).sum(); fn=((~p)&(y==1)).sum(); fp=(p&(y==0)).sum(); tn=((~p)&(y==0)).sum()
        tpr=tp/max(tp+fn,1); fpr=fp/max(fp+tn,1); prec=tp/max(tp+fp,1)
        out.append((rid,t,tpr,fpr,prec,(tpr+1-fpr)/2,len(y),int(y.sum()),int((y==0).sum()),(tp+tn)/len(y)))
    if not out: print(s,'no rows'); continue
    # use only runs of one modality: filter by run_id containing modality is implicit as dataset belongs to one modality
    a=np.array([list(o[2:6])+[o[9]] for o in out]).mean(0)
    print('%s n=%d fake=%d real=%d | seed-mean TPR=%.4f FPR=%.4f precision=%.4f BA=%.4f ACC=%.4f'%(s,out[0][6],out[0][7],out[0][8],*a), '| per-seed FPR', [round(float(o[3]),4) for o in out],'runs',[o[0] for o in out])
