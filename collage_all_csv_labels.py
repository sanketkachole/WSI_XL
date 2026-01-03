##################################################################
#---------- Done --------------  build master label file ---------------
##################################################################

import os
import pandas as pd

root = r"D:\3_modelling\MICCAI_Slide_XL\dataset_master"

import pandas as pd

lbl1 = pd.read_csv(r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\labels.csv", sep=",")
slide = pd.read_csv(r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\slide_master.csv", sep=",")
lbl2 = pd.read_csv(r"D:\3_modelling\MICCAI_Slide_XL\dataset\labels.csv", sep=",")
clinical = pd.read_csv(r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\clinical_master.csv", sep=",")


def collapse_slides(df):
    cols = [c for c in df.columns if c != "h5_path"]
    agg = {c: "first" for c in cols}
    if "h5_path" in df.columns:
        agg["h5_path"] = lambda s: "|".join(pd.unique(s.dropna().astype(str)))
    out = df.groupby("patient_id", as_index=False).agg(agg)
    return out

# patient-level versions of slide/label tables
lbl1_p = collapse_slides(lbl1)
lbl2_p = collapse_slides(lbl2)
slide_p = collapse_slides(slide)

# merge (outer keeps everything; you can switch to inner later)
m = clinical.merge(lbl1_p, on="patient_id", how="outer", suffixes=("", "_lbl1"))
m = m.merge(slide_p, on="patient_id", how="outer", suffixes=("", "_slide"))
m = m.merge(lbl2_p, on="patient_id", how="outer", suffixes=("", "_lbl2"))

# unify "project_id" / survival fields if duplicated across sources
for base in ["project_id", "ajcc_pathologic_stage", "tumor_grade", "vital_status", "survival_event", "survival_time_days"]:
    alts = [c for c in m.columns if c == base or c.startswith(base + "_")]
    if len(alts) > 1:
        m[base] = m[alts].bfill(axis=1).iloc[:, 0]
        drop = [c for c in alts if c != base]
        m.drop(columns=drop, inplace=True)

# unify h5 paths
h5_cols = [c for c in m.columns if c == "h5_path" or c.startswith("h5_path_")]
if len(h5_cols) > 1:
    m["h5_path"] = m[h5_cols].apply(
        lambda r: "|".join(pd.unique([x for x in r.dropna().astype(str) if x])),
        axis=1
    )
    m.drop(columns=[c for c in h5_cols if c != "h5_path"], inplace=True)

out = os.path.join(root, "master_labels_patient.csv")
m.to_csv(out, index=False)
print("Wrote:", out, "| rows:", len(m), "| cols:", len(m.columns))


##################################################################
# ------ Done ----------------- Create WSI table --------------------------
##################################################################
import os, re
import pandas as pd

src_dir = r"X:\trident_tcga_output\coords\coords_20x_224px\features_virchow2"
dst = r"/dataset_master/wsi_virchow_features_filename_table.csv"

pat_re = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
uuid_re = re.compile(r"\.([0-9a-fA-F-]{36})\.h5$")

rows = []
for fn in os.listdir(src_dir):
    if not fn.endswith(".h5"):
        continue
    m1 = pat_re.search(fn)
    m2 = uuid_re.search(fn)
    if not m1 or not m2:
        continue
    rows.append({
        "h5_path": os.path.join(src_dir, fn),
        "file_uuid": m2.group(1).lower(),
        "patient_id": m1.group(1),
    })

df = pd.DataFrame(rows).drop_duplicates()
df.to_csv(dst, index=False)

print("WSI files:", len(df))
print("Wrote:", dst)






##################################################################
# --------Done --------------- Crate clinical Reports' patient id table ------------------------
##################################################################
import os
import pandas as pd

# must be the SAME file/order used to produce report_token_embeds_*.npy
rep_csv = r"Y:\2_feature_extractors\text_feature_extractors\gemini_text_embeddings\TCGA_Reports.csv"

df = pd.read_csv(rep_csv)
if "patient_filename" not in df.columns:
    raise ValueError("TCGA_Reports.csv must have a 'patient_id' column to map rows -> patients.")

out = df[["patient_filename"]].copy()
out["report_row_index"] = range(len(out))

dst = r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\report_table.csv"
out.to_csv(dst, index=False)
print("Wrote:", dst, "| rows:", len(out))



##################################################################
# ----------------- Merge all .npy files into one by type ------------------
##################################################################
import os
import re
import glob
import numpy as np

# ---- config (edit if needed) ----
IN_DIR = r"Y:\2_feature_extractors\text_feature_extractors\distilbert_text_token_level_embeddings"
OUT_DIR = IN_DIR  # save next to inputs

OUT_MASK = os.path.join(OUT_DIR, "text_report_distilbert_attention_mask.npy")
OUT_CLS = os.path.join(OUT_DIR, "text_report_distilbert_cls.npy")
OUT_POOLED = os.path.join(OUT_DIR, "text_report_distilbert_pooled.npy")
OUT_TOKENS = os.path.join(OUT_DIR, "text_report_distilbert_token_embeds.npy")
OUT_IDS = os.path.join(OUT_DIR, "text_report_distilbert_patient_ids.npy")

_PAT = re.compile(r"_(\d+)_(\d+)\.npy$")


def _sorted_chunks(prefix: str):
    files = glob.glob(os.path.join(IN_DIR, f"{prefix}_*_*.npy"))
    items = []
    for f in files:
        m = _PAT.search(f)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            items.append((s, e, f))
    items.sort(key=lambda x: x[0])
    if not items:
        raise FileNotFoundError(f"No chunk files found for prefix: {prefix}")
    return items


def _merge_numeric(prefix: str, out_path: str):
    chunks = _sorted_chunks(prefix)

    # get shape/dtype from first chunk
    a0 = np.load(chunks[0][2], mmap_mode="r")
    dtype = a0.dtype
    tail_shape = a0.shape[1:]  # everything except batch dim

    # compute total rows + validate shapes
    total = 0
    for _, _, f in chunks:
        a = np.load(f, mmap_mode="r")
        if a.dtype != dtype or a.shape[1:] != tail_shape:
            raise ValueError(f"Shape/dtype mismatch in {f}: got {a.dtype} {a.shape}, expected {dtype} (N,{tail_shape})")
        total += a.shape[0]

    # stream-write into a single .npy (memmap) to avoid huge RAM usage
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(total, *tail_shape))

    off = 0
    for _, _, f in chunks:
        a = np.load(f, mmap_mode="r")
        n = a.shape[0]
        out[off:off + n] = a
        off += n

    out.flush()
    print(f"Saved {out_path}  shape={(total, *tail_shape)} dtype={dtype}")


# ---- merge patient IDs (force fixed-width unicode) ----
id_chunks = _sorted_chunks("patient_ids")

ids = []
for _, _, f in id_chunks:
    a = np.load(f, allow_pickle=False)
    a = a.astype("U64")  # force uniform string dtype
    ids.append(a)

ids = np.concatenate(ids, axis=0)
np.save(OUT_IDS, ids)
print(f"Saved {OUT_IDS}  shape={ids.shape} dtype={ids.dtype}")


# ---- merge numeric arrays ----
_merge_numeric("report_attention_mask", OUT_MASK)
_merge_numeric("report_cls", OUT_CLS)
_merge_numeric("report_pooled", OUT_POOLED)
_merge_numeric("report_token_embeds", OUT_TOKENS)
