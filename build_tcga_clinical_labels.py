#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


PATIENT_ID_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class ClinicalLabels:
    patient_id: str
    case_uuid: Optional[str]
    project_id: Optional[str]
    ajcc_stage: Optional[str]
    tumor_grade: Optional[str]
    vital_status: Optional[str]
    days_to_death: Optional[float]
    days_to_last_follow_up: Optional[float]

    @property
    def survival_event(self) -> Optional[int]:
        # event: 1=dead, 0=alive
        if self.vital_status is None:
            return None
        vs = self.vital_status.strip().lower()
        if vs == "dead":
            return 1
        if vs == "alive":
            return 0
        return None

    @property
    def survival_time_days(self) -> Optional[float]:
        # time: days_to_death if dead else days_to_last_follow_up
        ev = self.survival_event
        if ev is None:
            return None
        if ev == 1:
            return self.days_to_death
        return self.days_to_last_follow_up


def extract_patient_id_from_filename(name: str) -> Optional[str]:
    m = PATIENT_ID_RE.match(name)
    if not m:
        return None
    return m.group(1).upper()


def list_h5_files(h5_dir: Path) -> List[Path]:
    return sorted([p for p in h5_dir.rglob("*.h5") if p.is_file()])


def unique_patient_ids_from_h5(files: Iterable[Path]) -> Set[str]:
    ids: Set[str] = set()
    for f in files:
        pid = extract_patient_id_from_filename(f.name)
        if pid:
            ids.add(pid)
    return ids


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def gdc_fetch_labels_for_patients(
    patient_ids: List[str],
    batch_size: int = 200,
    sleep_s: float = 0.2,
    timeout_s: int = 60,
) -> Dict[str, ClinicalLabels]:
    """
    Returns dict keyed by patient submitter_id (TCGA-XX-YYYY).
    Uses POST requests (much easier than URL-encoding long filters).
    """
    fields = [
        "submitter_id",
        "case_id",
        "project.project_id",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_grade",
        "diagnoses.days_to_last_follow_up",
        "demographic.vital_status",
        "demographic.days_to_death",
    ]

    out: Dict[str, ClinicalLabels] = {}

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
            "fields": ",".join(fields),
            "format": "JSON",
            "size": len(batch),
        }

        logging.info("Querying GDC for %d patients...", len(batch))
        resp = requests.post(GDC_CASES_ENDPOINT, json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"GDC request failed: HTTP {resp.status_code} | {resp.text[:500]}")

        data = resp.json()
        hits = data.get("data", {}).get("hits", [])
        logging.info("Got %d hits back.", len(hits))

        for hit in hits:
            pid = (hit.get("submitter_id") or "").upper()
            case_uuid = hit.get("case_id")
            project_id = (hit.get("project", {}) or {}).get("project_id")

            demographic = hit.get("demographic") or {}
            vital_status = demographic.get("vital_status")
            days_to_death = demographic.get("days_to_death")

            # diagnoses is a list; take the first non-null stage/grade/time you find
            ajcc_stage = None
            tumor_grade = None
            days_to_last_follow_up = None
            for dx in (hit.get("diagnoses") or []):
                if ajcc_stage is None and dx.get("ajcc_pathologic_stage") is not None:
                    ajcc_stage = dx.get("ajcc_pathologic_stage")
                if tumor_grade is None and dx.get("tumor_grade") is not None:
                    tumor_grade = dx.get("tumor_grade")
                if days_to_last_follow_up is None and dx.get("days_to_last_follow_up") is not None:
                    days_to_last_follow_up = dx.get("days_to_last_follow_up")

            out[pid] = ClinicalLabels(
                patient_id=pid,
                case_uuid=case_uuid,
                project_id=project_id,
                ajcc_stage=ajcc_stage,
                tumor_grade=tumor_grade,
                vital_status=vital_status,
                days_to_death=days_to_death,
                days_to_last_follow_up=days_to_last_follow_up,
            )

        time.sleep(sleep_s)

    return out


def write_slide_level_csv(
    h5_files: List[Path],
    patient_to_labels: Dict[str, ClinicalLabels],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "h5_path",
        "patient_id",
        "case_uuid",
        "project_id",
        "ajcc_pathologic_stage",
        "tumor_grade",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
        "survival_event",
        "survival_time_days",
    ]

    missing_patients = 0
    rows_written = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for h5 in h5_files:
            pid = extract_patient_id_from_filename(h5.name)
            if not pid:
                continue
            pid = pid.upper()
            labels = patient_to_labels.get(pid)
            if labels is None:
                missing_patients += 1
                continue

            w.writerow(
                {
                    "h5_path": str(h5),
                    "patient_id": labels.patient_id,
                    "case_uuid": labels.case_uuid,
                    "project_id": labels.project_id,
                    "ajcc_pathologic_stage": labels.ajcc_stage,
                    "tumor_grade": labels.tumor_grade,
                    "vital_status": labels.vital_status,
                    "days_to_death": labels.days_to_death,
                    "days_to_last_follow_up": labels.days_to_last_follow_up,
                    "survival_event": labels.survival_event,
                    "survival_time_days": labels.survival_time_days,
                }
            )
            rows_written += 1

    logging.info("Wrote %d slide rows to %s", rows_written, out_csv)
    if missing_patients:
        logging.warning("Missing clinical data for %d slides (no patient match returned by GDC).", missing_patients)


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", type=str, default=r"X:\trident_tcga_output\coords\coords_20x_224px\features_virchow2", help="Folder containing .h5 files (recursively searched).")
    p.add_argument("--out_csv", type=str, default=r"D:\3_modelling\MICCAI_Slide_XL\dataset\labels.csv", help="Output CSV path.")
    p.add_argument("--batch_size", type=int, default=200, help="Patients per GDC API call.")
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    out_csv = Path(args.out_csv)

    if not h5_dir.exists():
        logging.error("h5_dir does not exist: %s", h5_dir)
        return 2

    h5_files = list_h5_files(h5_dir)
    logging.info("Found %d .h5 files.", len(h5_files))

    patient_ids = sorted(unique_patient_ids_from_h5(h5_files))
    logging.info("Extracted %d unique patient IDs.", len(patient_ids))

    if not patient_ids:
        logging.error("No TCGA patient IDs found in filenames. Example filename should start with TCGA-XX-YYYY...")
        return 2

    patient_to_labels = gdc_fetch_labels_for_patients(patient_ids, batch_size=args.batch_size)

    # Show a quick sample
    sample_pid = patient_ids[0]
    logging.info("Example patient %s labels: %s", sample_pid, patient_to_labels.get(sample_pid))

    write_slide_level_csv(h5_files, patient_to_labels, out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
