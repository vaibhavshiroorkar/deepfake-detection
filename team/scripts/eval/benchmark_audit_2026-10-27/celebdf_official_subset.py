# Post hoc, unregistered, descriptive: AUC of stored sealed scores restricted to the part of the official Celeb-DF v2
# test list (List_of_testing_videos.txt, 518 videos) that lies in our test_x. No model run; reads stored score CSVs only.
import csv, sys, hashlib, numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
R=Path(__file__).resolve().parents[3].as_posix()+'/'  # repository root (an absolute D: path until 2026-10-28)
lst={'celebdf_v2:'+l.split()[1] for l in open(R+'data/celebdf_v2/List_of_testing_videos.txt') if l.strip()}
print('official list size',len(lst))
def run(tag):
    path=R+'data/_final/scores/%s.csv'%tag
    print(tag,'sha256',hashlib.sha256(open(path,'rb').read()).hexdigest())
    rows={}   # seed -> list of (sid,group,label,score)
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            if r['dataset']=='celebdf_v2' and r['split']=='test_x' and r['modality']=='visual' and r['sample_id'] in lst:
                rows.setdefault(r['seed'],{})[r['sample_id']]=(r['group_id'],int(r['label']),float(r['score']))
    seeds=sorted(rows); sids=sorted(rows[seeds[0]])
    grp=np.array([rows[seeds[0]][s][0] for s in sids]); y=np.array([rows[seeds[0]][s][1] for s in sids])
    S=np.array([[rows[sd][s][2] for s in sids] for sd in seeds])
    print(' n=%d fake=%d real=%d groups=%d seeds=%s'%(len(y),y.sum(),(y==0).sum(),len(set(grp)),seeds))
    aucs=[roc_auc_score(y,S[i]) for i in range(len(seeds))]
    est=float(np.mean(aucs)); print(' per-seed AUC',[round(a,4) for a in aucs],'mean %.4f'%est)
    ug=np.array(sorted(set(grp))); idx={g:np.where(grp==g)[0] for g in ug}
    rng=np.random.default_rng(20261027); bs=[]
    for _ in range(2000):
        pick=rng.choice(len(ug),len(ug),replace=True)
        ii=np.concatenate([idx[ug[p]] for p in pick]); yy=y[ii]
        if yy.min()==yy.max(): continue
        bs.append(np.mean([roc_auc_score(yy,S[i][ii]) for i in range(len(seeds))]))
    print(' group bootstrap 95%% [%.4f, %.4f] (n_boot used %d)'%(np.percentile(bs,2.5),np.percentile(bs,97.5),len(bs)))
for t in sys.argv[1:]: run(t)
