import os, re
import numpy as np
import pandas as pd

root = r"D:\3_modelling\MICCAI_Slide_XL\dataset"
idx_csv = os.path.join(root, "patient_embedding_index.csv")
emb_npy = os.path.join(root, "gemini_embeddings_all.npy")

A = np.load(emb_npy).astype(np.float32)  # (P, 768)
df = pd.read_csv(idx_csv)

pat_re = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")

rows = {}
for fn, ei in zip(df["patient_filename"].astype(str), df["embedding_index"].astype(int)):
    m = pat_re.search(fn)
    if not m:
        continue
    pid = m.group(1)  # e.g. TCGA-BP-5195
    rows.setdefault(pid, []).append(ei)

cases_dir = os.path.join(root, "cases")
splits_dir = os.path.join(root, "splits")
os.makedirs(cases_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

T = 32  # dummy report length
patient_ids = sorted(rows.keys())

for pid in patient_ids:
    vec = A[rows[pid]].mean(axis=0)            # (768,)
    patch = vec[None, :]                       # (1, 768)
    cdir = os.path.join(cases_dir, pid)
    os.makedirs(cdir, exist_ok=True)
    np.save(os.path.join(cdir, "patch_embeds.npy"), patch)
    np.save(os.path.join(cdir, "report_input_ids.npy"), np.zeros((T,), np.int64))
    np.save(os.path.join(cdir, "report_attention_mask.npy"), np.ones((T,), np.int64))

# dummy labels + splits (80/10/10)
rng = np.random.default_rng(42)
perm = rng.permutation(len(patient_ids))
n = len(patient_ids)
n_tr = int(0.8 * n)
n_va = int(0.1 * n)

splits = {
    "train.csv": [patient_ids[i] for i in perm[:n_tr]],
    "val.csv":   [patient_ids[i] for i in perm[n_tr:n_tr+n_va]],
    "test.csv":  [patient_ids[i] for i in perm[n_tr+n_va:]],
}

for name, ids in splits.items():
    pd.DataFrame({"case_id": ids, "label": 0}).to_csv(os.path.join(splits_dir, name), index=False)

print("Patients:", len(patient_ids))
print("Wrote cases to:", cases_dir)
print("Wrote splits to:", splits_dir)
