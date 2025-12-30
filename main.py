#!/usr/bin/env python3
"""
Slide-XL: KV-summarized long-context WSI + report modeling (late fusion)

Assumptions (adjust via CLI flags):
- You already extracted and saved:
  1) WSI patch embeddings per case:  (N_patches, d_v) float32
  2) (Optional) patch coordinates per case: (N_patches, 2) int/float
  3) Text report token ids per case: (T,) int64 and attention mask: (T,) int64/bool
- This script trains/evaluates a model that:
  - Streams patch embeddings in chunks
  - Inserts learnable Slide Summarization Tokens (SST) every r patches
  - Runs a transformer over each chunk while attending to cached SST K/V from previous chunks
  - Keeps only SST K/V across chunks; discards raw patch token K/V
  - Performs late fusion: report tokens query the final slide memory via cross-attention
  - Produces slide-level predictions (classification)

Data layout (recommended):
  data_root/
    cases/
      <case_id>/
        patch_embeds.npy          # (N, d_v)
        coords.npy                # (N, 2) OPTIONAL
        report_input_ids.npy      # (T,)
        report_attention_mask.npy # (T,)
    splits/
      train.csv
      val.csv
      test.csv

Each split CSV must contain at least:
  case_id,label
For multi-label, add multiple columns and use --label_cols.

Notes:
- This is an end-to-end training script, but you must adapt:
  - number of classes / label columns
  - loss type (binary vs multiclass)
  - evaluation metrics
  - exact feature filenames if different

"""


from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:
    wandb = None  # type: ignore


# ------------------------------
# Logging
# -----------------------------
def setup_logging(log_level: str) -> None:
    numeric = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


LOGGER = logging.getLogger("slide_xl")


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_cast_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    if np.issubdtype(mask.dtype, np.integer):
        return mask.astype(np.bool_)
    raise ValueError(f"Unsupported attention_mask dtype: {mask.dtype}")


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Failed to load numpy array: {path}")
    return arr


# -----------------------------
# Data
# -----------------------------
@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    y: np.ndarray  # shape: (num_labels,)


class WSIFeatureDataset(Dataset[CaseRecord]):
    def __init__(
        self,
        csv_path: Path,
        cases_dir: Path,
        label_cols: List[str],
        strict_files: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.cases_dir = cases_dir
        self.label_cols = label_cols
        self.strict_files = strict_files
        self.records: List[CaseRecord] = self._read_csv()

    def _read_csv(self) -> List[CaseRecord]:
        records: List[CaseRecord] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in (["case_id"] + self.label_cols) if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV {self.csv_path} missing columns: {missing}")

            for row in reader:
                cid = row["case_id"]
                y_vals: List[float] = []
                for c in self.label_cols:
                    if row[c] == "" or row[c] is None:
                        raise ValueError(f"Missing label in {self.csv_path} for case_id={cid}, col={c}")
                    y_vals.append(float(row[c]))
                y = np.asarray(y_vals, dtype=np.float32)
                if self.strict_files:
                    case_path = self.cases_dir / cid
                    required = [
                        case_path / "patch_embeds.npy",
                        case_path / "report_input_ids.npy",
                        case_path / "report_attention_mask.npy",
                    ]
                    for p in required:
                        if not p.exists():
                            raise FileNotFoundError(f"Missing required file: {p}")
                records.append(CaseRecord(case_id=cid, y=y))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> CaseRecord:
        return self.records[idx]


@dataclass
class Batch:
    case_ids: List[str]
    patch_embeds: List[torch.Tensor]  # list of (N, d_v)
    coords: List[Optional[torch.Tensor]]  # list of (N, 2) or None
    report_input_ids: torch.Tensor  # (B, T)
    report_attn_mask: torch.Tensor  # (B, T) bool
    labels: torch.Tensor  # (B, L)


def collate_fn(records: Sequence[CaseRecord], cases_dir: Path, max_report_len: int) -> Batch:
    case_ids: List[str] = []
    patch_embeds: List[torch.Tensor] = []
    coords_list: List[Optional[torch.Tensor]] = []
    report_ids_list: List[torch.Tensor] = []
    report_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for rec in records:
        cid = rec.case_id
        case_path = cases_dir / cid

        pe = torch.from_numpy(load_npy(case_path / "patch_embeds.npy")).float()  # (N, d_v)
        if pe.ndim != 2:
            raise ValueError(f"patch_embeds must be 2D (N,d_v). Got {pe.shape} for {cid}")

        coords_path = case_path / "coords.npy"
        if coords_path.exists():
            co = torch.from_numpy(load_npy(coords_path)).float()
            if co.ndim != 2 or co.shape[1] != 2:
                raise ValueError(f"coords must be (N,2). Got {co.shape} for {cid}")
        else:
            co = None

        ids = torch.from_numpy(load_npy(case_path / "report_input_ids.npy")).long()
        msk = load_npy(case_path / "report_attention_mask.npy")
        msk = safe_cast_bool_mask(msk)
        msk_t = torch.from_numpy(msk).bool()

        # truncate/pad report
        if ids.ndim != 1 or msk_t.ndim != 1:
            raise ValueError(f"report arrays must be 1D. Got ids={ids.shape} mask={msk_t.shape} for {cid}")

        T = min(ids.shape[0], max_report_len)
        ids = ids[:T]
        msk_t = msk_t[:T]

        case_ids.append(cid)
        patch_embeds.append(pe)
        coords_list.append(co)
        report_ids_list.append(ids)
        report_mask_list.append(msk_t)
        labels_list.append(torch.from_numpy(rec.y).float())

    # pad reports to max len in batch
    max_T = max(x.shape[0] for x in report_ids_list) if report_ids_list else 0
    max_T = min(max_T, max_report_len)

    B = len(records)
    report_input_ids = torch.zeros((B, max_T), dtype=torch.long)
    report_attn_mask = torch.zeros((B, max_T), dtype=torch.bool)

    for i in range(B):
        ids = report_ids_list[i]
        msk = report_mask_list[i]
        t = min(ids.shape[0], max_T)
        report_input_ids[i, :t] = ids[:t]
        report_attn_mask[i, :t] = msk[:t]

    labels = torch.stack(labels_list, dim=0)  # (B, L)

    return Batch(
        case_ids=case_ids,
        patch_embeds=patch_embeds,
        coords=coords_list,
        report_input_ids=report_input_ids,
        report_attn_mask=report_attn_mask,
        labels=labels,
    )


# -----------------------------
# Chunking
# -----------------------------
def spatial_order_indices(coords: torch.Tensor) -> torch.Tensor:
    """
    Produce a stable spatial ordering from coords (N,2).
    Default: lexicographic sort by (y, x) to approximate scanline.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N,2), got {coords.shape}")
    # sort by y then x
    y = coords[:, 1]
    x = coords[:, 0]
    # argsort lexicographically: use stable sorting
    idx = torch.argsort(y * 1e6 + x)
    return idx


def make_chunks(
    n_tokens: int,
    chunk_size: int,
) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        chunks.append((start, end))
        start = end
    return chunks


# -----------------------------
# Model components
# -----------------------------
class MLPProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MultiHeadSelfAttentionKV(nn.Module):
    """
    Causal self-attention variant that supports a cached KV (memory) from previous chunks.
    Here we run non-causal attention (WSI patch stream doesn't need strict causality), but we
    still preserve the "carry-over" behavior by concatenating cached keys/values.

    We implement attention manually so we can:
      - accept per-layer cached KV tensors
      - return per-token K/V so caller can keep only SST positions
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, D)
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return x

    def forward(
        self,
        x: torch.Tensor,  # (B, L, D)
        mem_k: Optional[torch.Tensor],  # (B, H, M, Dh)
        mem_v: Optional[torch.Tensor],  # (B, H, M, Dh)
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, L_total) or (B, 1, L, L_total)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          y: (B, L, D)
          k: (B, H, L, Dh)  keys for current tokens
          v: (B, H, L, Dh)  values for current tokens
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if mem_k is not None and mem_v is not None:
            k_cat = torch.cat([mem_k, k], dim=2)  # (B,H,M+L,Dh)
            v_cat = torch.cat([mem_v, v], dim=2)
        else:
            k_cat = k
            v_cat = v

        # scaled dot-product
        # q: (B,H,L,Dh) ; k_cat: (B,H,Ltot,Dh)
        scores = torch.matmul(q, k_cat.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,Ltot)

        if attn_mask is not None:
            # mask should be True for allowed, False for blocked
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v_cat)  # (B,H,L,Dh)
        y = self._merge_heads(y)
        y = self.o_proj(y)
        return y, k, v


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayerKV(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionKV(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem_k: Optional[torch.Tensor],
        mem_v: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        y, k, v = self.attn(h, mem_k=mem_k, mem_v=mem_v, attn_mask=attn_mask)
        x = x + self.dropout(y)
        h2 = self.norm2(x)
        x = x + self.dropout(self.ff(h2))
        return x, k, v


class CrossAttention(nn.Module):
    """Report queries attend over slide memory (late fusion)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        q_in: torch.Tensor,      # (B, T, D)
        mem: torch.Tensor,       # (B, M, D)
        mem_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, M) allowed mask
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(q_in))     # (B,H,T,Dh)
        k = self._split_heads(self.k_proj(mem))      # (B,H,M,Dh)
        v = self._split_heads(self.v_proj(mem))      # (B,H,M,Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,M)
        if mem_mask is not None:
            scores = scores.masked_fill(~mem_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        out = self._merge_heads(out)
        out = self.o_proj(out)
        return out


class SlideXL(nn.Module):
    """
    Slide-XL core model:
      - Projects patch embeddings to D
      - Streams slide in chunks
      - Inserts SST tokens every r patches inside each chunk
      - Runs L transformer layers; carries over only SST KV across chunks
      - Late fusion: report tokens query final slide memory via cross-attention
      - Classification head from pooled report outputs
    """

    def __init__(
        self,
        d_v: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        sst_every_r: int,
        sst_init_std: float,
        max_mem_sst: int,
        num_labels: int,
        task_type: str,  # "binary" | "multiclass" | "multilabel"
    ) -> None:
        super().__init__()
        if sst_every_r <= 0:
            raise ValueError("sst_every_r must be > 0")

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sst_every_r = sst_every_r
        self.max_mem_sst = max_mem_sst
        self.num_labels = num_labels
        self.task_type = task_type

        self.projector = MLPProjector(d_in=d_v, d_out=d_model, dropout=dropout)

        # A simple text embedding; replace with your own report encoder if you have embeddings already.
        self.text_emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [TransformerLayerKV(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )

        # Learnable SST token embeddings (we reuse a single parameter vector, expanded per insertion)
        self.sst_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.sst_token, mean=0.0, std=sst_init_std)

        # Late fusion: one cross-attention block + FFN
        self.report_norm1 = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.report_norm2 = RMSNorm(d_model)
        self.report_ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Linear(d_model, num_labels)

    @torch.no_grad()
    def _truncate_memory(
        self,
        mem_k: List[Optional[torch.Tensor]],
        mem_v: List[Optional[torch.Tensor]],
        keep_last: int,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Keep only the most recent keep_last SSTs in memory for each layer.
            keep the KV “memory” from getting too long.
            It keeps only the most recent keep_last SST entries per transformer layer."""
        if keep_last <= 0:
            return [None] * len(mem_k), [None] * len(mem_v)

        new_k: List[Optional[torch.Tensor]] = []
        new_v: List[Optional[torch.Tensor]] = []
        for k, v in zip(mem_k, mem_v):
            if k is None or v is None:
                new_k.append(None)
                new_v.append(None)
                continue
            # k,v: (B,H,M,Dh)
            if k.shape[2] <= keep_last:
                new_k.append(k)
                new_v.append(v)
            else:
                new_k.append(k[:, :, -keep_last:, :].contiguous())
                new_v.append(v[:, :, -keep_last:, :].contiguous())
        return new_k, new_v

    def _interleave_sst(
        self,
        patch_tokens: torch.Tensor,  # (B, L, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave an SST after every r patches.
        Returns:
          tokens: (B, L + K, D)
          sst_mask: (B, L + K) bool indicating SST positions
        """
        B, L, D = patch_tokens.shape
        r = self.sst_every_r
        # number of SSTs: ceil(L / r) but we add after each full segment; if last shorter, still add one
        K = (L + r - 1) // r  # Compute how many SST tokens to add
        out_len = L + K

        tokens = patch_tokens.new_empty((B, out_len, D))
        sst_mask = torch.zeros((B, out_len), device=patch_tokens.device, dtype=torch.bool)

        # Fill
        out_pos = 0
        patch_pos = 0
        for k in range(K):
            seg_end = min(patch_pos + r, L)
            seg_len = seg_end - patch_pos
            if seg_len > 0:
                tokens[:, out_pos : out_pos + seg_len, :] = patch_tokens[:, patch_pos:seg_end, :]
                out_pos += seg_len
                patch_pos = seg_end
            # insert SST
            tokens[:, out_pos : out_pos + 1, :] = self.sst_token.expand(B, 1, D)
            sst_mask[:, out_pos] = True
            out_pos += 1

        if out_pos != out_len:
            raise RuntimeError("Interleave length mismatch")
        return tokens, sst_mask

    def encode_slide_memory(
        self,
        patch_embeds: torch.Tensor,            # (N, d_v)
        coords: Optional[torch.Tensor],        # (N,2) or None
        chunk_size: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Stream the slide and build final memory representation.
        Returns:
          mem_tokens_per_layer: list of (1, M, D) reconstructed token representations for memory (for cross-attn)
          mem_mask: (1, M) bool
        """
        # Order patches : If you have (x,y) coords for patches, it sorts them into a spatial scan order (so the stream is consistent).
        if coords is not None:
            order = spatial_order_indices(coords.to(device))
            patch_embeds = patch_embeds.to(device)[order]
        else:
            patch_embeds = patch_embeds.to(device)

        # project to model space
        patch_tokens_all = self.projector(patch_embeds.unsqueeze(0))  # (1, N, D)
        _, N, _ = patch_tokens_all.shape

        # Split into chunks : Instead of processing all N patches at once (too big), it processes ranges (s,e).
        chunks = make_chunks(n_tokens=N, chunk_size=chunk_size)

        # per-layer cached KV for SSTs : These hold the cached attention keys/values from SST tokens across chunks.
        mem_k: List[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]
        mem_v: List[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]

        # We also maintain a token-level memory representation for cross-attention.
        # We'll store the final-layer SST hidden states per chunk and concatenate.
        mem_tokens: List[torch.Tensor] = []

        for (s, e) in chunks:
            # take chunk patches and insert SST tokens This produces a sequence like: P P P P SST P P P P SST
            # sst_mask marks where SST tokens are.
            cur_patch_tokens = patch_tokens_all[:, s:e, :]  # (1, Lc, D)
            tokens, sst_mask = self._interleave_sst(cur_patch_tokens)  # (1, Lc+Kc, D) ******** patch token into SST ****************

            x = tokens
            # No attention mask: we allow full attention over (memory + current chunk tokens).
            for li, layer in enumerate(self.layers):
                # run transformer layers with past SST memory. Key idea: current tokens attend to (past SST memory + current tokens).
                x, k, v = layer(x, mem_k=mem_k[li], mem_v=mem_v[li], attn_mask=None)

                # Keep only SST K/V from current chunk  ************** IMP step *****************
                # k,v: (B=1, H, Lcur, Dh) where Lcur = tokens_len
                # sst_mask: (B=1, Lcur)
                sst_idx = torch.nonzero(sst_mask[0], as_tuple=False).squeeze(-1)  # (Kc,)
                k_sst = k[:, :, sst_idx, :].contiguous()
                v_sst = v[:, :, sst_idx, :].contiguous()

                # append the above SST KV from the current chunk to layer memory: ************** IMP step *****************
                if mem_k[li] is None:
                    mem_k[li], mem_v[li] = k_sst, v_sst
                else:
                    mem_k[li] = torch.cat([mem_k[li], k_sst], dim=2)
                    mem_v[li] = torch.cat([mem_v[li], v_sst], dim=2)

            # Save SST hidden states (final layer) as memory tokens for cross-attention
            sst_h = x[:, sst_mask[0], :].contiguous()  # (1, Kc, D)
            mem_tokens.append(sst_h)

            # Truncate memory if needed
            if self.max_mem_sst > 0:
                mem_k, mem_v = self._truncate_memory(mem_k, mem_v, keep_last=self.max_mem_sst)

                # Also truncate token memory to match
                cat = torch.cat(mem_tokens, dim=1)  # (1, M, D)
                if cat.shape[1] > self.max_mem_sst:
                    cat = cat[:, -self.max_mem_sst :, :].contiguous()
                mem_tokens = [cat]  # collapse after truncation

        mem = torch.cat(mem_tokens, dim=1) if len(mem_tokens) > 1 else mem_tokens[0]  # (1, M, D)
        mem_mask = torch.ones((1, mem.shape[1]), device=device, dtype=torch.bool)
        return [mem], mem_mask  # keep list to match signature expansion if desired

    def forward(
        self,
        batch_patch_embeds: List[torch.Tensor],
        batch_coords: List[Optional[torch.Tensor]],
        report_input_ids: torch.Tensor,   # (B, T)
        report_attn_mask: torch.Tensor,   # (B, T) bool
        chunk_size: int,
    ) -> torch.Tensor:
        device = report_input_ids.device
        B, T = report_input_ids.shape

        # Encode slide memory per case (streaming; per-sample)
        # For speed, this is per-sample loop; for production, consider batching slides by chunk length.
        mem_list: List[torch.Tensor] = []
        mem_mask_list: List[torch.Tensor] = []


        # Each sample has its own slide (different #patches), so it loops over batch.
        # encode_slide_memory streams the slide in chunks and returns SST memory tokens (shape (1, M, D)).
        # mem_mask marks which memory positions are valid.
        # Result: mem_list[i] is a variable-length memory sequence for slide i.
        for i in range(B):
            mem_tokens_per_layer, mem_mask = self.encode_slide_memory(
                patch_embeds=batch_patch_embeds[i],
                coords=batch_coords[i],
                chunk_size=chunk_size,
                device=device,
            )
            # We use the final memory token representation (not per layer) for cross-attn
            mem_list.append(mem_tokens_per_layer[0])  # (1, M, D)
            mem_mask_list.append(mem_mask)            # (1, M)

        # Pad memory across batch. Because each slide produces a different patch M, it pads them to M_max so cross-attention can run in a batch.
        M_max = max(m.shape[1] for m in mem_list)
        mem = torch.zeros((B, M_max, self.d_model), device=device, dtype=mem_list[0].dtype)
        mem_mask = torch.zeros((B, M_max), device=device, dtype=torch.bool)
        for i in range(B):
            m = mem_list[i].squeeze(0)       # (M, D)
            mm = mem_mask_list[i].squeeze(0) # (M,)
            mem[i, : m.shape[0], :] = m
            mem_mask[i, : mm.shape[0]] = mm

        # Report embeddings. Converts report token IDs into vectors. # Zeroes out padding tokens (mask is boolean).
        rep = self.text_emb(report_input_ids)  # (B,T,D)
        rep = rep * report_attn_mask.unsqueeze(-1).to(rep.dtype)

        # Late fusion cross-attention: report queries -> slide memory
        # Queries = report tokens & Keys/Values = slide memory tokens (SSTs)
        # So each report token can “look up” relevant slide info from the compressed memory.
        rep_h = self.report_norm1(rep)
        mem_mask_4d = mem_mask[:, None, None, :]  # (B,1,1,M)
        ca = self.cross_attn(q_in=rep_h, mem=mem, mem_mask=mem_mask_4d)
        rep = rep + self.drop(ca)

        # FFN
        rep_h2 = self.report_norm2(rep)
        rep = rep + self.drop(self.report_ff(rep_h2))

        # Pool report tokens (masked mean)
        denom = report_attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B,1)
        pooled = (rep * report_attn_mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

        logits = self.head(pooled)  # (B, num_labels)
        return logits

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        targets shape:
          - binary: (B,1) or (B,)
          - multilabel: (B,L) with {0,1}
          - multiclass: (B,) int64 with class indices
        """
        if self.task_type == "multiclass":
            if targets.ndim != 1:
                raise ValueError("For multiclass, targets must be (B,)")
            return F.cross_entropy(logits, targets.long())
        if self.task_type == "binary":
            # logits: (B,1) preferred
            if logits.shape[1] != 1:
                raise ValueError("For binary, num_labels must be 1")
            t = targets.view(-1, 1)
            return F.binary_cross_entropy_with_logits(logits, t)
        if self.task_type == "multilabel":
            return F.binary_cross_entropy_with_logits(logits, targets)
        raise ValueError(f"Unknown task_type: {self.task_type}")


# -----------------------------
# Metrics (minimal, no sklearn dependency)
# -----------------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> float:
    # Picks the class with highest score
    # Accuracy = fraction of samples where predicted class == target
    if task_type == "multiclass":
        pred = torch.argmax(logits, dim=1)
        return (pred == targets.long()).float().mean().item()
    # Converts logits to probabilities # Thresholds at 0.5
    # Flattens to (B,) # Compares with {0,1} targets
    if task_type == "binary":
        pred = (torch.sigmoid(logits) > 0.5).view(-1)
        tgt = targets.view(-1) > 0.5
        return (pred == tgt).float().mean().item()
    # Same thresholding But applied per label
    # Accuracy = fraction of correctly predicted label entries (not per sample)
    if task_type == "multilabel":
        pred = (torch.sigmoid(logits) > 0.5)
        tgt = targets > 0.5
        return (pred == tgt).float().mean().item()
    raise ValueError(task_type)


# -----------------------------
# Train / Eval loops
# -----------------------------
@dataclass
class TrainConfig:
    data_root: Path
    cases_subdir: str
    splits_subdir: str

    train_csv: str
    val_csv: str
    test_csv: str

    label_cols: List[str]
    task_type: str
    num_labels: int

    d_v: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float

    sst_every_r: int
    sst_init_std: float
    max_mem_sst: int

    chunk_size: int
    max_report_len: int

    batch_size: int
    num_workers: int

    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    grad_accum: int

    amp: bool
    device: str
    seed: int

    out_dir: Path
    save_best: bool

    use_wandb: bool
    wandb_project: str
    wandb_run_name: str


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train Slide-XL (KV-summarized WSI+Report, late fusion)")

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--cases_subdir", type=str, default="cases")
    p.add_argument("--splits_subdir", type=str, default="splits")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--val_csv", type=str, default="val.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")
    p.add_argument("--label_cols", type=str, required=True,
                   help="Comma-separated label column(s), e.g. 'label' or 'msi,her2'")
    p.add_argument("--task_type", type=str, choices=["binary", "multiclass", "multilabel"], required=True)

    # model dims
    p.add_argument("--d_v", type=int, required=True, help="Patch embedding dimension (input)")
    p.add_argument("--vocab_size", type=int, required=True, help="Report token vocab size")
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_layers", type=int, required=True)
    p.add_argument("--n_heads", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--dropout", type=float, default=0.1)

    # summarization
    p.add_argument("--sst_every_r", type=int, default=64, help="Insert one SST after every r patch tokens")
    p.add_argument("--sst_init_std", type=float, default=0.02)
    p.add_argument("--max_mem_sst", type=int, default=0,
                   help="Max number of SSTs kept in memory (0 = no truncation)")

    # streaming
    p.add_argument("--chunk_size", type=int, default=4096, help="Number of patch tokens per chunk (pre-SST)")
    p.add_argument("--max_report_len", type=int, default=512)

    # training
    p.add_argument("--batch_size", type=int, default=1, help="WSI streaming is heavy; start with 1")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")

    # misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_best", action="store_true")

    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Slide-XL")
    p.add_argument("--wandb_run_name", type=str, default="")

    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    if not label_cols:
        raise ValueError("--label_cols must not be empty")

    # infer num_labels if possible
    if args.task_type == "binary":
        num_labels = 1
    elif args.task_type == "multiclass":
        # must be single label col with integer class id
        if len(label_cols) != 1:
            raise ValueError("multiclass requires exactly one label column containing class indices")
        # num_labels must be provided via --num_classes? We avoid hallucinating; force user.
        raise ValueError("For multiclass, please switch to a class-index CSV and add --num_labels flag in code.")
    else:
        # multilabel: number of label cols
        num_labels = len(label_cols)

    return TrainConfig(
        data_root=data_root,
        cases_subdir=args.cases_subdir,
        splits_subdir=args.splits_subdir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        label_cols=label_cols,
        task_type=args.task_type,
        num_labels=num_labels,
        d_v=args.d_v,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sst_every_r=args.sst_every_r,
        sst_init_std=args.sst_init_std,
        max_mem_sst=args.max_mem_sst,
        chunk_size=args.chunk_size,
        max_report_len=args.max_report_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        amp=args.amp,
        device=args.device,
        seed=args.seed,
        out_dir=out_dir,
        save_best=args.save_best,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cases_dir = cfg.data_root / cfg.cases_subdir
    splits_dir = cfg.data_root / cfg.splits_subdir

    train_ds = WSIFeatureDataset(
        csv_path=splits_dir / cfg.train_csv,
        cases_dir=cases_dir,
        label_cols=cfg.label_cols,
        strict_files=True,
    )
    val_ds = WSIFeatureDataset(
        csv_path=splits_dir / cfg.val_csv,
        cases_dir=cases_dir,
        label_cols=cfg.label_cols,
        strict_files=True,
    )
    test_ds = WSIFeatureDataset(
        csv_path=splits_dir / cfg.test_csv,
        cases_dir=cases_dir,
        label_cols=cfg.label_cols,
        strict_files=True,
    )

    def _collate(recs: Sequence[CaseRecord]) -> Batch:
        return collate_fn(recs, cases_dir=cases_dir, max_report_len=cfg.max_report_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    return train_loader, val_loader, test_loader


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        str(path),
    )


@torch.no_grad()
def evaluate(cfg: TrainConfig, model: SlideXL, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        report_ids = batch.report_input_ids.to(device, non_blocking=True)
        report_mask = batch.report_attn_mask.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        logits = model(
            batch_patch_embeds=batch.patch_embeds,
            batch_coords=batch.coords,
            report_input_ids=report_ids,
            report_attn_mask=report_mask,
            chunk_size=cfg.chunk_size,
        )
        loss = model.loss_fn(logits, labels)
        acc = accuracy_from_logits(logits, labels, cfg.task_type)

        total_loss += float(loss.item())
        total_acc += float(acc)
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


def train(cfg: TrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg_dict.items()}
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))


    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    train_loader, val_loader, test_loader = build_loaders(cfg)

    model = SlideXL(
        d_v=cfg.d_v,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        sst_every_r=cfg.sst_every_r,
        sst_init_std=cfg.sst_init_std,
        max_mem_sst=cfg.max_mem_sst,
        num_labels=cfg.num_labels,
        task_type=cfg.task_type,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    if cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --use_wandb was set.")
        run_name = cfg.wandb_run_name or f"slide_xl_{int(time.time())}"
        wandb.init(project=cfg.wandb_project, name=run_name, config=dataclasses.asdict(cfg))

    best_val = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_steps = 0

        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            report_ids = batch.report_input_ids.to(device, non_blocking=True)
            report_mask = batch.report_attn_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(
                    batch_patch_embeds=batch.patch_embeds,
                    batch_coords=batch.coords,
                    report_input_ids=report_ids,
                    report_attn_mask=report_mask,
                    chunk_size=cfg.chunk_size,
                )
                loss = model.loss_fn(logits, labels) / cfg.grad_accum

            scaler.scale(loss).backward()

            # metrics on unscaled loss
            acc = accuracy_from_logits(logits.detach(), labels, cfg.task_type)
            running_loss += float(loss.item()) * cfg.grad_accum
            running_acc += float(acc)
            n_steps += 1
            global_step += 1

            if step % cfg.grad_accum == 0:
                # gradient clipping
                scaler.unscale_(opt)
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % max(1, (10 // max(1, cfg.batch_size))) == 0:
                avg_loss = running_loss / max(1, n_steps)
                avg_acc = running_acc / max(1, n_steps)
                LOGGER.info(
                    "Epoch %d | Step %d/%d | loss=%.4f | acc=%.4f",
                    epoch, step, len(train_loader), avg_loss, avg_acc
                )
                if cfg.use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/acc": avg_acc, "epoch": epoch}, step=global_step)

        # Validate
        val_metrics = evaluate(cfg, model, val_loader, device)
        LOGGER.info("Epoch %d | VAL | loss=%.4f | acc=%.4f", epoch, val_metrics["loss"], val_metrics["acc"])
        if cfg.use_wandb:
            wandb.log({"val/loss": val_metrics["loss"], "val/acc": val_metrics["acc"], "epoch": epoch}, step=global_step)

        # Save
        if cfg.save_best:
            if val_metrics["acc"] > best_val:
                best_val = val_metrics["acc"]
                save_checkpoint(cfg.out_dir / "best.pt", model, opt, epoch, best_val)
                LOGGER.info("Saved best checkpoint: acc=%.4f", best_val)
        else:
            save_checkpoint(cfg.out_dir / f"epoch_{epoch}.pt", model, opt, epoch, best_val)

    # Test (best if available)
    if cfg.save_best and (cfg.out_dir / "best.pt").exists():
        ckpt = torch.load(str(cfg.out_dir / "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        LOGGER.info("Loaded best checkpoint for test (epoch=%s, best_metric=%.4f)", ckpt.get("epoch"), ckpt.get("best_metric", -1.0))

    test_metrics = evaluate(cfg, model, test_loader, device)
    LOGGER.info("TEST | loss=%.4f | acc=%.4f", test_metrics["loss"], test_metrics["acc"])
    if cfg.use_wandb:
        wandb.log({"test/loss": test_metrics["loss"], "test/acc": test_metrics["acc"]}, step=global_step)
        wandb.finish()

    (cfg.out_dir / "final_metrics.json").write_text(json.dumps(test_metrics, indent=2))


def main() -> None:
    setup_logging("INFO")
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
