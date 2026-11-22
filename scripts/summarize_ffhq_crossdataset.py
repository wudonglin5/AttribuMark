import json
from pathlib import Path

ROOT = Path(r"F:\project\beifen\DiffMark-main")
RES = ROOT / "results"
DOCS = ROOT / "docs"

CONDITIONS = [
    ("CelebA-HQ", "VQGAN", RES / "user_fingerprint_eval_30k_min11_probs.json", RES / "multimage_llr_summary.json"),
    ("CelebA-HQ", "SimSwap-224", RES / "user_fingerprint_eval_30k_min11_simswap224_probs.json", RES / "simswap224_multimage_llr_summary.json"),
    ("CelebA-HQ", "InSwapper-128", RES / "user_fingerprint_eval_30k_min11_inswapper128_probs.json", RES / "inswapper128_multimage_llr_summary.json"),
    ("FFHQ", "Clean", RES / "ffhq_crossdataset_100x5_clean_probs.json", RES / "ffhq_crossdataset_100x5_clean_multimage_llr.json"),
    ("FFHQ", "VQGAN", RES / "ffhq_crossdataset_100x5_vqgan_probs.json", RES / "ffhq_crossdataset_100x5_vqgan_multimage_llr.json"),
    ("FFHQ", "SimSwap-224", RES / "ffhq_crossdataset_100x5_simswap224_probs.json", RES / "ffhq_crossdataset_100x5_simswap224_multimage_llr.json"),
    ("FFHQ", "InSwapper-128", RES / "ffhq_crossdataset_100x5_inswapper128_probs.json", RES / "ffhq_crossdataset_100x5_inswapper128_multimage_llr.json"),
]

rows = []
for dataset, attack, probs_path, summary_path in CONDITIONS:
    probs = json.loads(probs_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_image = probs["per_image"]
    m5 = summary["summary"]["5"]["llr"]
    z3 = m5["thresholds"]["3.0"]
    rows.append({
        "dataset": dataset,
        "attack": attack,
        "num_users": probs["num_users"],
        "images_per_user": probs["images_per_user"],
        "per_image_ber": per_image["ber"],
        "per_image_exact_match": per_image["exact_match"],
        "mean_hamming": per_image["mean_hamming"],
        "max_hamming": per_image["max_hamming"],
        "m5_mean_z": m5["mean_true_z"],
        "m5_top1": m5["top1"],
        "z3_detect": z3["detect"],
        "z3_false_accuse": z3["false_accuse"],
        "z3_abstain": z3["abstain"],
    })

out_json = DOCS / "ffhq_crossdataset_100x5_summary.json"
out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

header = "| 数据集 | 攻击 | 逐图 BER | 逐图全对 | 平均/最大 Hamming | m=5 真实用户 z | Top-1@100 | Z=3 检出/误指控/拒识 |"
sep = "|---|---|---:|---:|---:|---:|---:|---:|"
lines = [header, sep]
for r in rows:
    lines.append(
        f"| {r['dataset']} | {r['attack']} | {r['per_image_ber']*100:.2f}% | "
        f"{r['per_image_exact_match']*100:.1f}% | {r['mean_hamming']:.3f} / {r['max_hamming']} | "
        f"{r['m5_mean_z']:.3f} | {r['m5_top1']*100:.0f}% | "
        f"{r['z3_detect']*100:.0f}/{r['z3_false_accuse']*100:.0f}/{r['z3_abstain']*100:.0f}% |"
    )
print("\n".join(lines))
print(f"\nSaved: {out_json}")