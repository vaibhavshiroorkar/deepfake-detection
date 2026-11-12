# AP and AUC of each fake diffusion folder against the LAION reals (and against ImageNet reals, sensitivity),
# from stored sealed scores data/_final/scores/I-4.csv (image modality, diffusion_test test_x). No model run.
import csv, hashlib, collections, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path
R=Path(__file__).resolve().parents[3].as_posix()+'/'  # repository root (an absolute D: path until 2026-10-28); p=R+'data/_final/scores/I-4.csv'
print('sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
S=collections.defaultdict(list)
with open(p,newline='') as f:
    for r in csv.DictReader(f):
        if r['modality']=='image' and r['dataset']=='diffusion_test' and r['split']=='test_x':
            S[(r['sample_id'].split(':',1)[1].split('/')[0], r['seed'])].append(float(r['score']))
fakes=['dalle','glide_100_10','glide_100_27','glide_50_27','guided','ldm_100','ldm_200','ldm_200_cfg']
for real in ('laion','imagenet'):
    aps=[];aucs=[]
    for fo in fakes:
        a=[];b=[]
        for sd in '012':
            sf=np.array(S[(fo,sd)]); sr=np.array(S[(real,sd)]); y=np.r_[np.ones(len(sf)),np.zeros(len(sr))]; s=np.r_[sf,sr]
            a.append(average_precision_score(y,s)); b.append(roc_auc_score(y,s))
        aps.append(np.mean(a)); aucs.append(np.mean(b))
        print('  vs %-8s %-14s AP=%.4f AUC=%.4f'%(real,fo,aps[-1],aucs[-1]))
    print('MEAN over 8 fake sets vs',real,': AP %.4f  AUC %.4f'%(np.mean(aps),np.mean(aucs)))
