"""E-023 step 0: measure mpeg4 recipe parameters from dev FVRA sample clips and count h264 FVRA over the whole dev half (metadata only, no model)."""
import csv, json, subprocess, sys, collections, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e021"))
import e021
O = e021.OUT.parent / "e023"
def frames(p):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","frame=pict_type,key_frame","-of","csv=p=0",str(p)],capture_output=True,text=True)
    t=[l.split(",") for l in r.stdout.split() if l]
    return t
def gop(p):
    t=frames(p); pt="".join(x[1] if len(x)>1 else "?" for x in t)
    ks=[i for i,x in enumerate(t) if x[0]=="1"]
    d=[b-a for a,b in zip(ks,ks[1:])]
    return {"n":len(t),"n_I":pt.count("I"),"n_B":pt.count("B"),"n_P":pt.count("P"),"first_key_gaps":d[:3],"max_gap":max(d) if d else len(t)}
rows=e021.read_sample(); probe={r["sid"]:r for r in e021.read_csv(e021.OUT/"probe.csv")}
mp=[r for r in rows if r["cat"]=="FVRA" and probe[r["sid"]]["v_codec"]=="mpeg4"]
with ThreadPoolExecutor(6) as ex: g=list(ex.map(lambda r: gop(e021.DATA/r["path"]),mp))
out={"n_mpeg4_FVRA_sample":len(mp),
 "frac_clips_with_B":sum(1 for x in g if x["n_B"]>0)/len(g),
 "max_gap_median":st.median(x["max_gap"] for x in g),"max_gap_counter":collections.Counter(x["max_gap"] for x in g).most_common(5),
 "nI_median":st.median(x["n_I"] for x in g),"n_median":st.median(x["n"] for x in g)}
pr=[probe[r["sid"]] for r in mp]
bit=sorted(float(p["v_bitrate"]) for p in pr)
out["bitrate_q25_med_q75"]=[bit[len(bit)//4],st.median(bit),bit[3*len(bit)//4]]
out["profile"]=collections.Counter(p["v_profile"] for p in pr).most_common(3)
out["encoder_tag"]=collections.Counter(p["encoder"] for p in pr).most_common(3)
out["container"]=collections.Counter(p["container"] for p in pr).most_common(2)
out["pix_fmt"]=collections.Counter(p["pix_fmt"] for p in pr).most_common(2)
out["fps"]=collections.Counter(p["fps"] for p in pr).most_common(3)
# whole dev FVRA: codec counts
dev=e021.load_dev(); fv=[v for v in dev.values() if v["cat"]=="FVRA"]
with ThreadPoolExecutor(6) as ex: pf=list(ex.map(lambda v: e021.ffprobe(e021.DATA/v["path"]),fv))
cnt=collections.Counter((v["method"],p["v_codec"]) for v,p in zip(fv,pf))
out["dev_FVRA_total"]=len(fv); out["dev_FVRA_method_codec"]={f"{a}|{b}":n for (a,b),n in sorted(cnt.items())}
with open(O/"fvra_all_probe.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["sid","method","group","cell","v_codec"]); 
    for v,p in zip(fv,pf): w.writerow([v["sid"],v["method"],v["group"],v["cell"],p["v_codec"]])
(O/"measure.json").write_text(json.dumps(out,indent=1),encoding="utf-8"); print(json.dumps(out,indent=1))
