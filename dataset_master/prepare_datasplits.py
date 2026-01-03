############################################################
#--------------- build unique multimodal patient dataset --------
#############################################################
import os
import json
import numpy as np
import pandas as pd
import h5py


WSI_CSV = r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\wsi_virchow_features_filename_ID_table.csv"
LABELS_CSV = r"D:\3_modelling\MICCAI_Slide_XL\dataset_master\labels.csv"

TEXT_DIR = r"Y:\2_feature_extractors\text_feature_extractors\distilbert_text_token_level_embeddings"
TEXT_IDS_NPY = os.path.join(TEXT_DIR, "text_report_distilbert_patient_ids.npy")

OUT_DIR = r"meta"
OUT_PARQUET = os.path.join(OUT_DIR, "patients_breast.parquet")
OUT_CSV = os.path.join(OUT_DIR, "patients_breast.csv")  # fallback if parquet fails

BREAST_PROJECT_ID = "TCGA-BRCA"  # breast cancer cohort in TCGA


def _best_effort_patch_count(h5_path: str) -> int:
    """
    Returns number of patches in the slide feature file.
    Tries to find the 'most patch-like' dataset: 2D/3D with largest first dimension.
    """
    if not os.path.exists(h5_path):
        return -1

    best = -1

    def _visit(_, obj):
        nonlocal best
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            if shape is None or len(shape) < 2:
                return
            n = int(shape[0])
            if n > best:
                best = n

    try:
        with h5py.File(h5_path, "r") as f:
            f.visititems(_visit)
    except Exception:
        return -1

    return best


def _build_text_row_map(text_ids: np.ndarray, to_tcga3_fn) -> dict:
    # map normalized patient_id (TCGA-XX-YYYY) -> first row index
    m = {}
    for i, pid in enumerate(text_ids.astype(str)):
        pid3 = to_tcga3_fn(pid)
        if pid3 and pid3 not in m:
            m[pid3] = i
    return m



os.makedirs(OUT_DIR, exist_ok=True)

# ---- load sources ----
wsi = pd.read_csv(WSI_CSV)
labels = pd.read_csv(LABELS_CSV)

text_ids = np.load(TEXT_IDS_NPY, allow_pickle=False).astype(str)


import re
TCGA3 = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")

def to_tcga3(x):
    m = TCGA3.search(str(x))
    return m.group(1) if m else None

text_row_map = _build_text_row_map(text_ids, to_tcga3)

# normalize IDs to TCGA-XX-YYYY in BOTH tables
wsi["patient_id_3"] = wsi["patient_id"].map(to_tcga3)
labels["patient_id_3"] = labels["patient_id"].map(to_tcga3)

# ---- breast filter (step 3) ----
labels_breast = labels[labels["project_id"].astype(str) == BREAST_PROJECT_ID].copy()

# ---- merge using normalized ID (steps 1-2) ----
df = wsi.merge(
    labels_breast,
    how="inner",
    on="patient_id_3",
    suffixes=("", "_lbl"),
)

# keep a single canonical patient_id column
df["text_row"] = df["patient_id_3"].astype(str).map(text_row_map)
df["patient_id"] = df["patient_id_3"]

# --- pick one slide per patient (FAST heuristic) ---
# prefer DX1 if present, otherwise keep first
df["is_dx1"] = df["h5_path"].astype(str).str.contains(r"\.DX1\.", case=False, regex=True).astype(int)
df = (
    df.sort_values(["patient_id", "is_dx1"], ascending=[True, False])
      .drop_duplicates("patient_id", keep="first")
      .reset_index(drop=True)
)
df = df.drop(columns=["is_dx1"])


# ---- choose highest-patch slide per patient (step 4) ----
# compute patch counts (can take time; but simple and correct)
# patch_counts = []
# for p in df["h5_path"].astype(str).tolist():
#     patch_counts.append(_best_effort_patch_count(p))
# df["wsi_n_patches"] = patch_counts
#
# # drop missing/unreadable WSI files
# df = df[df["wsi_n_patches"] > 0].copy()

# # pick best slide per patient
# df = (
#     df.sort_values(["patient_id", "wsi_n_patches"], ascending=[True, False])
#       .drop_duplicates("patient_id", keep="first")
#       .reset_index(drop=True)
# )

# ---- join text rows (step 5) ----
#df["text_row"] = df["patient_id"].astype(str).map(text_row_map)

# ---- drop missing text + missing targets (steps 6-7) ----
# define "label available" = survival_time_days and survival_event present (edit if you want different target)
targets = ["survival_time_days", "survival_event"]
for c in targets:
    if c not in df.columns:
        raise ValueError(f"Missing required target column in labels.csv: {c}")

print("rows before:", len(df))
print("missing text_row:", df["text_row"].isna().sum())
print("missing survival_time_days:", df["survival_time_days"].isna().sum())
print("missing survival_event:", df["survival_event"].isna().sum())
print("project_id value counts head:\n", df["project_id"].value_counts().head(10))


print("example merged patient_id_3:", df["patient_id_3"].dropna().astype(str).head(5).tolist())
print("example text map keys:", list(text_row_map.keys())[:5])


df = df.dropna(subset=["text_row"] + targets).copy()
df["text_row"] = df["text_row"].astype(int)

# ---- final minimal meta table (what your dataloader should read) ----
meta_cols = [
    "patient_id",
    "project_id",
    "primary_site",
    "disease_type",
    "vital_status",
    "survival_event",
    "survival_time_days",
    "h5_path",
    "file_uuid",
    "wsi_n_patches",
    "text_row",
]
meta_cols = [c for c in meta_cols if c in df.columns]  # keep only existing
out = df[meta_cols].copy()

# ---- write outputs ----
try:
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved: {OUT_PARQUET}")
except Exception as e:
    out.to_csv(OUT_CSV, index=False)
    print(f"Parquet failed ({e}). Saved CSV instead: {OUT_CSV}")

print("Breast patients kept:", len(out))
print("Unique patient_ids:", out["patient_id"].nunique())

# optional: dump a quick summary json
summary = {
    "breast_project_id": BREAST_PROJECT_ID,
    "n_patients": int(out["patient_id"].nunique()),
    "text_rows_min": int(out["text_row"].min()) if len(out) else None,
    "text_rows_max": int(out["text_row"].max()) if len(out) else None,
}
with open(os.path.join(OUT_DIR, "patients_breast_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary JSON.")


print("sample text_ids:", text_ids[:10])





print("TEXT ids sample raw:", text_ids[:5])
print("TEXT ids sample normalized:", [to_tcga3(x) for x in text_ids[:10]])
print("TEXT map sample keys:", list(text_row_map.keys())[:10])
print("TEXT map size:", len(text_row_map))

print("WSI patient_id sample:", wsi["patient_id"].astype(str).head(5).tolist())
print("WSI patient_id_3 sample:", wsi["patient_id_3"].astype(str).head(5).tolist())

print("LABEL patient_id sample:", labels["patient_id"].astype(str).head(5).tolist())
print("LABEL patient_id_3 sample:", labels["patient_id_3"].astype(str).head(5).tolist())

print("BRCA label patients:", labels_breast["patient_id_3"].nunique())
print("WSI patients:", wsi["patient_id_3"].nunique())
print("TEXT patients:", len(text_row_map))


print("Merged rows:", len(df), "unique patients:", df["patient_id_3"].nunique())
print("Merged sample patient_id_3:", df["patient_id_3"].head(10).tolist())
print("Intersection BRCA∩TEXT:", len(set(labels_breast["patient_id_3"].dropna()) & set(text_row_map.keys())))


print("rows after 1-slide-per-patient:", len(df), "unique:", df["patient_id"].nunique())
print("missing text_row:", df["text_row"].isna().sum())
print("missing survival_time_days:", df["survival_time_days"].isna().sum())
print("missing survival_event:", df["survival_event"].isna().sum())



meta_cols = [
    "patient_id",
    "project_id",                   # cancer type
    "primary_site",
    "disease_type",
    "vital_status",
    "survival_event",               # Target Label
    "survival_time_days",           # Target Label
    "h5_path",                      # which slide file
    "file_uuid",
    "text_row",                     # which text row
]
out = df[meta_cols].copy()

out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print("Final patients:", len(out), "unique:", out["patient_id"].nunique())
print(out.head(3))
