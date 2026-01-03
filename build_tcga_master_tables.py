#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

PATIENT_ID_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
UUID_RE = re.compile(
    r"\.([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.h5$"
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def safe_first_non_null(values: Sequence[Any]) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def get_first_in_list(obj: Any) -> Any:
    if isinstance(obj, list) and obj:
        return obj[0]
    return obj


def flatten_join(values: Any, sep: str = "|", limit: int = 20) -> str:
    """
    For list fields (e.g., multiple treatments), store a compact joined string.
    """
    if values is None:
        return ""
    if not isinstance(values, list):
        return str(values)
    out = []
    for v in values[:limit]:
        if v is None:
            continue
        out.append(str(v))
    return sep.join(out)


def extract_patient_id(filename: str) -> Optional[str]:
    m = PATIENT_ID_RE.match(filename)
    return m.group(1).upper() if m else None


def extract_uuid(filename: str) -> Optional[str]:
    m = UUID_RE.search(filename)
    return m.group(1).lower() if m else None


def list_h5_files(h5_dir: Path) -> List[Path]:
    return sorted([p for p in h5_dir.rglob("*.h5") if p.is_file()])


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def post_gdc(endpoint: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GDC request failed: {endpoint} | HTTP {r.status_code} | {r.text[:800]}")
    return r.json()


# ---------------------------
# 1) CLINICAL MASTER (per patient)
# ---------------------------

CLINICAL_FIELDS = [
    # identifiers
    "submitter_id",
    "case_id",
    "project.project_id",
    "project.primary_site",
    "project.disease_type",
    # demographic
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    # diagnosis core
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.age_at_diagnosis",
    # stage / grade
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    # survival follow-up-ish fields commonly used
    "diagnoses.days_to_last_follow_up",
    # exposures (availability varies a lot by cancer)
    "exposures.smoking_status",
    "exposures.alcohol_history",
    "exposures.cigarettes_per_day",
    "exposures.years_smoked",
    # treatments (often sparse / inconsistent across projects)
    "treatments.treatment_type",
    "treatments.therapeutic_agents",
    "treatments.treatment_or_therapy",
    # follow_ups (can be multiple; keep compact)
    "follow_ups.days_to_follow_up",
    "follow_ups.progression_or_recurrence",
]


def fetch_clinical_master(
    patient_ids: List[str], batch_size: int, sleep_s: float, timeout_s: int
) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: patient_id -> flat dict row for clinical_master.csv
    """
    out: Dict[str, Dict[str, Any]] = {}

    for batch in chunked(patient_ids, batch_size):
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "project.program.name", "value": ["TCGA"]}},
                {"op": "in", "content": {"field": "submitter_id", "value": batch}},
            ],
        }
        payload = {
            "filters": filters,
            "fields": ",".join(CLINICAL_FIELDS),
            "format": "JSON",
            "size": len(batch),
        }
        logging.info("Clinical: querying %d patients...", len(batch))
        data = post_gdc(GDC_CASES_ENDPOINT, payload, timeout_s=timeout_s)
        hits = data.get("data", {}).get("hits", [])
        logging.info("Clinical: got %d hits.", len(hits))

        for h in hits:
            pid = (h.get("submitter_id") or "").upper()
            project = h.get("project") or {}
            demo = h.get("demographic") or {}

            diagnoses = h.get("diagnoses") or []
            # Take first diagnosis with most info:
            dx = get_first_in_list(diagnoses) or {}

            exposures = h.get("exposures") or []
            exp0 = get_first_in_list(exposures) or {}

            treatments = h.get("treatments") or []
            # Keep treatments compact (lists)
            trt_types = [t.get("treatment_type") for t in treatments if isinstance(t, dict)]
            trt_agents = [t.get("therapeutic_agents") for t in treatments if isinstance(t, dict)]
            trt_therapy = [t.get("treatment_or_therapy") for t in treatments if isinstance(t, dict)]

            follow_ups = h.get("follow_ups") or []
            fu_days = [fu.get("days_to_follow_up") for fu in follow_ups if isinstance(fu, dict)]
            fu_prog = [fu.get("progression_or_recurrence") for fu in follow_ups if isinstance(fu, dict)]

            row = {
                "patient_id": pid,
                "case_uuid": h.get("case_id"),
                "project_id": project.get("project_id"),
                "primary_site": project.get("primary_site"),
                "disease_type": project.get("disease_type"),
                # demographic
                "gender": demo.get("gender"),
                "race": demo.get("race"),
                "ethnicity": demo.get("ethnicity"),
                "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                # diagnosis
                "primary_diagnosis": dx.get("primary_diagnosis"),
                "morphology": dx.get("morphology"),
                "tissue_or_organ_of_origin": dx.get("tissue_or_organ_of_origin"),
                "age_at_diagnosis": dx.get("age_at_diagnosis"),
                # stage / grade
                "ajcc_pathologic_stage": dx.get("ajcc_pathologic_stage"),
                "ajcc_pathologic_t": dx.get("ajcc_pathologic_t"),
                "ajcc_pathologic_n": dx.get("ajcc_pathologic_n"),
                "ajcc_pathologic_m": dx.get("ajcc_pathologic_m"),
                "tumor_grade": dx.get("tumor_grade"),
                # follow-up
                "days_to_last_follow_up": dx.get("days_to_last_follow_up"),
                # exposures (first record)
                "smoking_status": exp0.get("smoking_status"),
                "alcohol_history": exp0.get("alcohol_history"),
                "cigarettes_per_day": exp0.get("cigarettes_per_day"),
                "years_smoked": exp0.get("years_smoked"),
                # treatments (compact)
                "treatment_type_list": flatten_join(trt_types),
                "therapeutic_agents_list": flatten_join(trt_agents),
                "treatment_or_therapy_list": flatten_join(trt_therapy),
                # follow_ups (compact)
                "followup_days_list": flatten_join(fu_days),
                "followup_progression_list": flatten_join(fu_prog),
            }

            # survival labels (canonical)
            vs = (row.get("vital_status") or "").strip().lower()
            if vs == "dead":
                row["survival_event"] = 1
                row["survival_time_days"] = row.get("days_to_death")
            elif vs == "alive":
                row["survival_event"] = 0
                row["survival_time_days"] = row.get("days_to_last_follow_up")
            else:
                row["survival_event"] = ""
                row["survival_time_days"] = ""

            out[pid] = row

        time.sleep(sleep_s)

    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------
# 2) SLIDE MASTER (per .h5 / file UUID)
# ---------------------------

FILE_FIELDS = [
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "experimental_strategy",
    # linkage
    "cases.submitter_id",
    "cases.case_id",
    "cases.project.project_id",
    "samples.sample_type",
    "samples.submitter_id",
]


def fetch_file_master_by_uuids(
    uuids: List[str], batch_size: int, sleep_s: float, timeout_s: int
) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: file_uuid -> row dict
    """
    out: Dict[str, Dict[str, Any]] = {}

    for batch in chunked(uuids, batch_size):
        filters = {"op": "in", "content": {"field": "file_id", "value": batch}}
        payload = {"filters": filters, "fields": ",".join(FILE_FIELDS), "format": "JSON", "size": len(batch)}
        logging.info("Files: querying %d file UUIDs...", len(batch))
        data = post_gdc(GDC_FILES_ENDPOINT, payload, timeout_s=timeout_s)
        hits = data.get("data", {}).get("hits", [])
        logging.info("Files: got %d hits.", len(hits))

        for h in hits:
            fid = (h.get("file_id") or "").lower()
            cases = h.get("cases") or []
            case0 = get_first_in_list(cases) or {}
            case_pid = (case0.get("submitter_id") or "").upper()
            case_uuid = case0.get("case_id")
            proj = (case0.get("project") or {})
            project_id = proj.get("project_id")

            samples = h.get("samples") or []
            sample_types = [s.get("sample_type") for s in samples if isinstance(s, dict)]
            sample_submitter_ids = [s.get("submitter_id") for s in samples if isinstance(s, dict)]

            out[fid] = {
                "file_uuid": fid,
                "file_name": h.get("file_name"),
                "data_category": h.get("data_category"),
                "data_type": h.get("data_type"),
                "experimental_strategy": h.get("experimental_strategy"),
                "patient_id": case_pid,
                "case_uuid": case_uuid,
                "project_id": project_id,
                "sample_type_list": flatten_join(sample_types),
                "sample_submitter_id_list": flatten_join(sample_submitter_ids),
            }

        time.sleep(sleep_s)

    return out


# ---------------------------
# MAIN
# ---------------------------

def main() -> int:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", type=str, required=True, help="Folder containing .h5 files (recursively searched).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV master tables.")
    p.add_argument("--batch_size", type=int, default=200, help="Batch size for GDC API calls.")
    p.add_argument("--timeout_s", type=int, default=90, help="HTTP timeout seconds.")
    p.add_argument("--sleep_s", type=float, default=0.15, help="Small delay between requests (be nice to GDC).")
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_dir.exists():
        logging.error("h5_dir does not exist: %s", h5_dir)
        return 2

    h5_files = list_h5_files(h5_dir)
    logging.info("Found %d .h5 files.", len(h5_files))

    # Extract patient IDs + UUIDs from filenames
    patient_ids: Set[str] = set()
    uuids: Set[str] = set()
    slide_rows: List[Dict[str, Any]] = []

    for f in h5_files:
        pid = extract_patient_id(f.name)
        uid = extract_uuid(f.name)
        if pid:
            patient_ids.add(pid)
        if uid:
            uuids.add(uid)

        slide_rows.append(
            {
                "h5_path": str(f),
                "patient_id_from_name": pid or "",
                "file_uuid_from_name": uid or "",
            }
        )

    patient_id_list = sorted(patient_ids)
    uuid_list = sorted(uuids)
    logging.info("Extracted %d unique patient IDs.", len(patient_id_list))
    logging.info("Extracted %d unique file UUIDs.", len(uuid_list))

    # 1) Clinical master (per patient)
    clinical = fetch_clinical_master(
        patient_id_list, batch_size=args.batch_size, sleep_s=args.sleep_s, timeout_s=args.timeout_s
    )

    clinical_rows = list(clinical.values())
    clinical_fields = list(clinical_rows[0].keys()) if clinical_rows else ["patient_id"]
    clinical_path = out_dir / "clinical_master.csv"
    write_csv(clinical_path, clinical_rows, clinical_fields)
    logging.info("Wrote clinical master: %s (%d patients)", clinical_path, len(clinical_rows))

    # 2) File/slide master by UUID (per slide)
    file_master = fetch_file_master_by_uuids(
        uuid_list, batch_size=args.batch_size, sleep_s=args.sleep_s, timeout_s=args.timeout_s
    )

    # Build slide_master rows by merging filename-derived mapping with file_master + clinical
    merged_slide_rows: List[Dict[str, Any]] = []
    for r in slide_rows:
        uid = (r["file_uuid_from_name"] or "").lower()
        pid_from_name = (r["patient_id_from_name"] or "").upper()

        fm = file_master.get(uid, {})
        pid = (fm.get("patient_id") or pid_from_name).upper() if (fm.get("patient_id") or pid_from_name) else ""
        cl = clinical.get(pid, {})

        merged_slide_rows.append(
            {
                "h5_path": r["h5_path"],
                "file_uuid": uid,
                "patient_id": pid,
                "project_id": fm.get("project_id") or cl.get("project_id") or "",
                "sample_type_list": fm.get("sample_type_list") or "",
                "sample_submitter_id_list": fm.get("sample_submitter_id_list") or "",
                "ajcc_pathologic_stage": cl.get("ajcc_pathologic_stage") or "",
                "tumor_grade": cl.get("tumor_grade") or "",
                "vital_status": cl.get("vital_status") or "",
                "survival_event": cl.get("survival_event") if cl else "",
                "survival_time_days": cl.get("survival_time_days") if cl else "",
            }
        )

    slide_master_path = out_dir / "slide_master.csv"
    slide_fieldnames = list(merged_slide_rows[0].keys()) if merged_slide_rows else ["h5_path"]
    write_csv(slide_master_path, merged_slide_rows, slide_fieldnames)
    logging.info("Wrote slide master: %s (%d slides)", slide_master_path, len(merged_slide_rows))

    # 3) Modeling labels (same idea as before, slide-level)
    labels_path = out_dir / "labels.csv"
    label_fieldnames = [
        "h5_path",
        "patient_id",
        "project_id",
        "ajcc_pathologic_stage",
        "tumor_grade",
        "vital_status",
        "survival_event",
        "survival_time_days",
    ]
    label_rows = [
        {
            "h5_path": r["h5_path"],
            "patient_id": r["patient_id"],
            "project_id": r["project_id"],
            "ajcc_pathologic_stage": r["ajcc_pathologic_stage"],
            "tumor_grade": r["tumor_grade"],
            "vital_status": r["vital_status"],
            "survival_event": r["survival_event"],
            "survival_time_days": r["survival_time_days"],
        }
        for r in merged_slide_rows
    ]
    write_csv(labels_path, label_rows, label_fieldnames)
    logging.info("Wrote labels table: %s (%d slides)", labels_path, len(label_rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
