# -*- coding: utf-8 -*-
"""Audit Table 4 (five-fold group-wise CV) from the three seed-stability runs."""
import json, math, os, statistics
import numpy as np

ROOT = r"F:\project\beifen\DiffMark-main"
GRID = np.arange(1.0, 5.001, 0.05)
TARGET = 0.01
FOLDS = 5

EXPECTED = {
 ("VQ-mean",2): (2.80,(2.75,2.95),100.0,98.7,0.0,1.3),
 ("VQ-mean",3): (2.95,(1.85,3.10),93.7,88.0,1.7,10.3),
 ("VQ-mean",4): (3.07,(2.75,3.10),66.7,63.3,2.0,34.7),
 ("VQ-mean",5): (3.05,(2.85,3.20),62.7,51.0,2.0,47.0),
 ("SimSwap-mean",2): (2.65,(2.65,2.95),100.0,99.0,0.0,1.0),
 ("SimSwap-mean",3): (2.68,(2.55,2.80),96.3,94.0,2.0,4.0),
 ("SimSwap-mean",4): (2.75,(2.70,2.75),85.0,74.7,1.3,24.0),
 ("SimSwap-mean",5): (2.85,(2.40,2.90),82.7,67.3,2.7,30.0),
 ("SimSwap-median",2): (2.63,(2.60,2.95),100.0,99.0,0.0,1.0),
 ("SimSwap-median",3): (2.83,(2.60,2.85),98.0,93.3,2.0,4.7),
 ("SimSwap-median",4): (2.53,(2.15,2.60),96.0,92.7,3.0,4.3),
 ("SimSwap-median",5): (2.95,(2.25,3.00),85.3,67.0,1.7,31.3),
 ("SimSwap-mosaic",2): (3.22,(2.60,3.30),71.7,65.7,2.0,32.3),
 ("SimSwap-mosaic",3): (3.10,(3.05,3.10),41.0,34.3,2.0,63.7),
 ("SimSwap-mosaic",4): (3.43,(3.05,3.45),12.0,12.0,2.0,86.0),
 ("SimSwap-mosaic",5): (3.15,(3.15,3.15),27.0,25.3,0.0,74.7),
}

def load_codes(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return np.asarray([[int(b) for b in c] for c in data["codes"]], dtype=np.float64)

def z_scores(probs, codes):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    scores = ((2.0 * codes - 1.0) * llr[None, :]).sum(axis=1)
    return scores / np.sqrt((llr * llr).sum())

def outcome(record, key, codes):
    z = z_scores(record[key + "_probs"], codes)
    top = int(np.argmax(z))
    return float(z[top]), top in set(record["colluders"])

def select_threshold(z, hit):
    best = None
    for tau in GRID:
        accused = z >= tau
        fa = float(np.sum(accused & ~hit) / len(accused))
        if fa > TARGET:
            continue
        det = float(np.sum(accused & hit) / len(accused))
        cand = (det, float(tau), fa)
        if best is None or cand > best:
            best = cand
    if best is None:
        tau = float(GRID[-1]); accused = z >= tau
        return tau, float(np.sum(accused & ~hit) / len(accused))
    return best[1], best[2]

def rates(dec):
    acc = np.asarray([d[0] for d in dec], dtype=bool)
    hit = np.asarray([d[1] for d in dec], dtype=bool)
    return (100.0 * np.mean(acc & hit), 100.0 * np.mean(acc & ~hit), 100.0 * np.mean(~acc))

def run_one(path, k):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = load_codes(data["codebook_json"])
    records = [r for r in data["records"] if r["k"] == k]
    fold_taus = []
    clean_dec, attack_dec = [], []
    for fold in range(FOLDS):
        cal_z, cal_hit = [], []
        test_clean, test_attack = [], []
        for rec in records:
            is_test = (rec["group"] % FOLDS == fold)
            for key, bucket in (("clean", test_clean), ("vqgan", test_attack)):
                row = outcome(rec, key, codes)
                if is_test:
                    bucket.append(row)
                else:
                    cal_z.append(row[0]); cal_hit.append(row[1])
        tau, _ = select_threshold(np.asarray(cal_z), np.asarray(cal_hit, dtype=bool))
        fold_taus.append(float(tau))
        for bucket, out in ((test_clean, clean_dec), (test_attack, attack_dec)):
            for z, hit in bucket:
                out.append((z >= tau, hit))
    cdet, _, _ = rates(clean_dec)
    adet, afa, aabst = rates(attack_dec)
    return dict(fold_taus=fold_taus, clean_detect=cdet, attack_detect=adet, attack_fa=afa, attack_abstain=aabst)

def main():
    sources = {
        "VQ-mean": ("collusion_vqgan_mean.json", [2026,2027,2028]),
        "SimSwap-mean": ("collusion_simswap224_mean.json", [2026,2027,2028]),
        "SimSwap-median": ("collusion_simswap224_median.json", [2026,2027,2028]),
        "SimSwap-mosaic": ("collusion_simswap224_mosaic.json", [2026,2027,2028]),
    }
    report = {}
    print("condition        k  tau_mean [range]      clean  attack  FA    abst   | maxdiff")
    max_diff_all = 0.0
    for cond,(fn,seeds) in sources.items():
        for k in range(2,6):
            rows=[]
            for seed in seeds:
                path=os.path.join(ROOT,"results","seed_stability",f"seed{seed}",fn)
                rows.append(run_one(path,k))
            taus=[t for r in rows for t in r["fold_taus"]]
            tau_mean=statistics.mean(statistics.median(r["fold_taus"]) for r in rows)
            rec=dict(
                k=k, tau_mean=tau_mean, tau_range=(min(taus),max(taus)),
                clean_detect=statistics.mean(r["clean_detect"] for r in rows),
                attack_detect=statistics.mean(r["attack_detect"] for r in rows),
                attack_fa=statistics.mean(r["attack_fa"] for r in rows),
                attack_abstain=statistics.mean(r["attack_abstain"] for r in rows),
                seeds={str(s):r for s,r in zip(seeds,rows)},
            )
            report[f"{cond}|{k}"]=rec
            exp=EXPECTED[(cond,k)]
            diffs=[abs(rec["tau_mean"]-exp[0]),abs(rec["tau_range"][0]-exp[1][0]),abs(rec["tau_range"][1]-exp[1][1]),
                   abs(rec["clean_detect"]-exp[2]),abs(rec["attack_detect"]-exp[3]),abs(rec["attack_fa"]-exp[4]),abs(rec["attack_abstain"]-exp[5])]
            max_diff_all=max(max_diff_all,max(diffs))
            print(f"{cond:15s} {k}  {tau_mean:.2f} [{min(taus):.2f}-{max(taus):.2f}]  "
                  f"{rec['clean_detect']:.1f}  {rec['attack_detect']:.1f}  {rec['attack_fa']:.1f}  {rec['attack_abstain']:.1f}  | {max(diffs):.2f}")
    out=os.path.join(ROOT,"results","table4_cv_3seed_audit.json")
    with open(out,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print("saved",out,"max_diff",max_diff_all)

if __name__=="__main__":
    main()
