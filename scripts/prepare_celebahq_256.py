import io, os, random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(r"F:\project\beifen\DiffMark-main")
SRC = ROOT / "CelebA-HQ" / "_downloads" / "celebahq256_train_00000.parquet"
OUT = ROOT / "CelebA-HQ"
TRAIN = OUT / "train_256"
TEST = OUT / "test_256"
VAL = OUT / "val_256"
for d in (TRAIN, TEST, VAL):
    d.mkdir(parents=True, exist_ok=True)

pf = pq.ParquetFile(SRC)
rows = pf.metadata.num_rows
indices = list(range(rows))
random.Random(20261120).shuffle(indices)
test_idx = set(indices[:500])
val_idx = set(indices[500:786])

counts = {"train": 0, "test": 0, "val": 0}
row_no = 0
for batch in pf.iter_batches(batch_size=128, columns=["image"]):
    for item in batch.column(0).to_pylist():
        raw = item["bytes"]
        split = "test" if row_no in test_idx else ("val" if row_no in val_idx else "train")
        dst_dir = {"train": TRAIN, "test": TEST, "val": VAL}[split]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BICUBIC)
        dst = dst_dir / f"{row_no:06d}.png"
        img.save(dst, format="PNG", optimize=False)
        counts[split] += 1
        row_no += 1
        if row_no % 500 == 0:
            print(f"processed {row_no}/{rows} {counts}", flush=True)
print("done", row_no, counts, flush=True)
