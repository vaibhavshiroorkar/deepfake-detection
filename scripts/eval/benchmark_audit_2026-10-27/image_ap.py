# Average precision (and AUC) per generator folder from stored sealed score file data/_final/scores/I-4.csv (image modality).
# No model run. Per-seed metric, then mean over the 3 seeds. Mean of per-folder AP is the CNNDetection/UnivFD convention.
import csv, hashlib, collections, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path
R=Path(__file__).resolve().parents[3].as_posix()+'/'  # repository root (an absolute D: path until 2026-10-28)
p=R+'data/_final/scores/I-4.csv'
print('sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
D=collections.defaultdict(lambda:collections.defaultdict(lambda:([],[])))
with open(p,newline='') as f:
    for r in csv.DictReader(f):
        if r['modality']!='image' or r['split'] in ('val','dev_x'): continue
        ds=r['dataset']; rest=r['sample_id'].split(':',1)[1]
        folder=rest.split('/')[0]
        D[(ds,r['split'])][(folder,r['seed'])][0].append(int(r['label'])); D[(ds,r['split'])][(folder,r['seed'])][1].append(float(r['score']))
for key,dd in D.items():
    folders=sorted({k[0] for k in dd}); aps=[];aucs=[]
    out=[]
    for fo in folders:
        a=[];b=[]
        for sd in ('0','1','2'):
            y,s=dd[(fo,sd)]; y=np.array(y); s=np.array(s)
            a.append(average_precision_score(y,s)); b.append(roc_auc_score(y,s))
        aps.append(np.mean(a)); aucs.append(np.mean(b)); out.append((fo,len(y),int(y.sum()),np.mean(a),np.mean(b)))
    print(key,'folders',len(folders),'mean AP %.4f mean AUC %.4f'%(np.mean(aps),np.mean(aucs)))
    for o in out: print('   %-22s n=%5d fake=%5d AP=%.4f AUC=%.4f'%o)
