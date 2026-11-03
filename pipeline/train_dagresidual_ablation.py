#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers.modeling_utils as hf_modeling_utils

NUM_LABELS = 6
AUTO_REGION_FIELDS = [
    "region_id",
    "region_name",
    "region",
    "province",
    "province_name",
    "area",
    "area_name",
    "cluster_id",
    "cluster_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--raw_ts_shape", type=str, default="8,13")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--raw_ts_prefix_tokens", type=int, default=4)
    parser.add_argument("--raw_ts_hidden_dim", type=int, default=512)
    parser.add_argument("--raw_ts_use_revin", action="store_true", default=False)
    parser.add_argument("--raw_ts_use_multiscale", action="store_true", default=False)
    parser.add_argument("--raw_ts_preprefix_graph_enable", action="store_true", default=False)
    parser.add_argument("--raw_ts_preprefix_graph_hidden_dim", type=int, default=256)
    parser.add_argument("--raw_ts_preprefix_graph_alpha_init", type=float, default=-3.0)
    parser.add_argument("--raw_ts_preprefix_graph_dropout", type=float, default=0.0)
    parser.add_argument("--raw_ts_preprefix_graph_norm", type=str, default="indegree", choices=["indegree", "none"])
    parser.add_argument("--raw_ts_preprefix_graph_layers", type=int, default=1)
    parser.add_argument("--raw_ts_preprefix_graph_attention", action="store_true", default=False)
    parser.add_argument("--pcd_graph_enable", action="store_true", default=False)
    parser.add_argument("--pcd_hidden_dim", type=int, default=128)
    parser.add_argument("--pcd_transport_layers", type=int, default=2)
    parser.add_argument("--pcd_transport_dropout", type=float, default=0.0)
    parser.add_argument("--pcd_temporal_num_layers", type=int, default=1)
    parser.add_argument("--pcd_spatial_lambda", type=float, default=0.0)
    parser.add_argument("--pcd_root_lambda", type=float, default=0.0)
    parser.add_argument("--pcd_temporal_lambda", type=float, default=0.0)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--dual_head_by_root", action="store_true", default=False)
    parser.add_argument("--weighted_loss", action="store_true", default=False)
    parser.add_argument("--max_class_weight", type=float, default=3.0)

    parser.add_argument("--root_aware_loss", action="store_true", default=False)
    parser.add_argument("--root_weight", type=float, default=1.5)
    parser.add_argument("--root_label_extra", type=str, default="3:2.0,4:3.0")
    parser.add_argument("--max_root_sample_weight", type=float, default=4.0)

    parser.add_argument("--root_region_env_fusion", action="store_true", default=False)
    parser.add_argument("--root_region_env_weight", type=float, default=0.5)
    parser.add_argument("--root_region_exclude_self", action="store_true", default=False)
    parser.add_argument("--region_group_primary_key", type=str, default="target_date")

    parser.add_argument("--dag_residual_enable", action="store_true", default=False)
    parser.add_argument("--dag_residual_hidden_dim", type=int, default=256)
    parser.add_argument("--dag_residual_alpha_init", type=float, default=-3.0)
    parser.add_argument("--dag_residual_dropout", type=float, default=0.0)
    parser.add_argument("--dag_residual_norm", type=str, default="indegree", choices=["indegree", "none"])
    parser.add_argument("--dag_residual_layers", type=int, default=1)
    parser.add_argument("--dag_residual_attention", action="store_true", default=False)
    parser.add_argument("--dag_residual_edge_gate", action="store_true", default=False)
    parser.add_argument("--dag_residual_gate_hidden_dim", type=int, default=128)

    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default="joint_score")
    parser.add_argument("--greater_is_better", action="store_true", default=True)
    parser.add_argument("--best_macro_tol", type=float, default=0.003)
    parser.add_argument("--joint_stage_bonus", type=float, default=10.0)

    # ---- ablation-specific flags ----
    parser.add_argument("--ablation_mask", type=str, default=None,
                        help="Comma-separated column indices to KEEP in raw_ts. "
                             "Unspecified columns are zeroed out. e.g. '0,11,12' keeps PM2.5+time only.")
    parser.add_argument("--no_lora", action="store_true", default=False,
                        help="Freeze LLM weights (no LoRA). Only train projector + classifier + dag_residual.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_shape(text: str) -> Tuple[int, int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"--raw_ts_shape must look like 8,13, got: {text}")
    return vals[0], vals[1]


def parse_root_label_extra(spec: str) -> Dict[int, float]:
    spec = (spec or "").strip()
    if not spec:
        return {}
    out: Dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        out[int(k.strip())] = float(v.strip())
    return out


def normalize_raw_ts(raw_ts: Any, expect_shape: Tuple[int, int]) -> np.ndarray:
    h, w = expect_shape
    arr = np.asarray(raw_ts, dtype=np.float32)
    if arr.ndim == 1 and arr.size == h * w:
        arr = arr.reshape(h, w)
    if arr.shape != (h, w):
        raise ValueError(f"Unexpected raw_ts shape: expected {(h, w)}, got {arr.shape}")
    return arr.astype(np.float32)


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _sanitize_parallel_plan(obj: Any) -> None:
    for attr in ["tp_plan", "pp_plan", "base_model_tp_plan", "base_model_pp_plan", "_tp_plan", "_pp_plan"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is None:
                setattr(obj, attr, {})
            elif isinstance(val, dict):
                setattr(obj, attr, {k: v for k, v in val.items() if v is not None})


def _normalize_parallel_styles() -> None:
    fallback = {None, "colwise", "rowwise", "colwise_rep", "rowwise_rep", "gather", "replicate", "local", "sequence_parallel"}
    styles = getattr(hf_modeling_utils, "ALL_PARALLEL_STYLES", None)
    if styles is None:
        hf_modeling_utils.ALL_PARALLEL_STYLES = fallback
        return
    if isinstance(styles, tuple):
        hf_modeling_utils.ALL_PARALLEL_STYLES = tuple(list(styles) + [x for x in fallback if x not in styles])
        return
    if isinstance(styles, list):
        for x in fallback:
            if x not in styles:
                styles.append(x)
        return
    if isinstance(styles, set):
        styles.update(fallback)
        return
    try:
        hf_modeling_utils.ALL_PARALLEL_STYLES = set(styles) | fallback
    except Exception:
        hf_modeling_utils.ALL_PARALLEL_STYLES = fallback


def compute_class_weights_from_group_index(group_rows: List[Dict[str, Any]], num_labels: int = NUM_LABELS, max_class_weight: float = 3.0) -> torch.Tensor:
    counts = np.zeros((num_labels,), dtype=np.float64)
    for row in group_rows:
        labels = np.load(row["labels_path"])
        for y in labels.tolist():
            y = int(y)
            if 0 <= y < num_labels:
                counts[y] += 1.0
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    if max_class_weight is not None and max_class_weight > 0:
        weights = np.minimum(weights, float(max_class_weight))
    return torch.tensor(weights, dtype=torch.float32)


class RevINRawTs(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x = (x - mean) / std
        return x * self.affine_weight + self.affine_bias


class MultiScaleRawTsEncoder(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)

    def forward(self, raw_ts: torch.Tensor) -> torch.Tensor:
        coarse = raw_ts.unfold(dimension=1, size=2, step=2).mean(dim=-1)
        if coarse.shape[1] * 2 != raw_ts.shape[1]:
            return raw_ts
        coarse_up = coarse.repeat_interleave(2, dim=1)
        residual = raw_ts - coarse_up
        return torch.cat([raw_ts, coarse_up, residual], dim=1)


class DagResidualBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = 256,
        alpha_init: float = -3.0,
        dropout: float = 0.0,
        norm_mode: str = "indegree",
        num_layers: int = 1,
        use_attention: bool = False,
        use_edge_gate: bool = False,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)
        self.dropout_p = float(dropout)
        self.norm_mode = str(norm_mode)
        self.num_layers = int(num_layers)
        self.use_attention = bool(use_attention)
        self.use_edge_gate = bool(use_edge_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.ModuleDict({
                "msg_proj": nn.Linear(self.model_dim, self.model_dim, bias=False),
                "pre_norm": nn.LayerNorm(self.model_dim),
                "out_proj": nn.Sequential(
                    nn.Linear(self.model_dim, self.hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_dim, self.model_dim, bias=False),
                ),
            })
            self.layers.append(layer)

        if self.use_attention:
            self.attn_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
            self.attn_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self._ensure_edge_gate_module()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    @staticmethod
    def remap_old_state_dict(sd: dict) -> dict:
        """Remap old single-layer keys (msg_proj.*, pre_norm.*, out_proj.*) to layers.0.*"""
        new_sd = {}
        for k, v in sd.items():
            if k.startswith(("msg_proj.", "pre_norm.", "out_proj.")):
                new_sd[f"layers.0.{k}"] = v
            else:
                new_sd[k] = v
        return new_sd

    def _ensure_edge_gate_module(self) -> None:
        if not self.use_edge_gate:
            return
        if hasattr(self, "edge_gate"):
            return
        self.edge_gate = nn.Sequential(
            nn.Linear(self.model_dim * 2 + 2, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.gate_hidden_dim, 1),
        )

    def _compute_edge_gate(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        dag_level: Optional[torch.Tensor],
        is_root: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n = h.shape[0]
        parent_feat = h.unsqueeze(1).expand(n, n, -1)
        child_feat = h.unsqueeze(0).expand(n, n, -1)
        if dag_level is None:
            level_delta = torch.zeros((n, n, 1), device=h.device, dtype=h.dtype)
        else:
            level = dag_level.to(device=h.device, dtype=h.dtype).view(n, 1)
            level_delta = (level - level.T).unsqueeze(-1)
        if is_root is None:
            child_root = torch.zeros((n, n, 1), device=h.device, dtype=h.dtype)
        else:
            child_root = is_root.to(device=h.device, dtype=h.dtype).view(1, n, 1).expand(n, n, 1)
        gate_in = torch.cat([parent_feat, child_feat, level_delta, child_root], dim=-1)
        gate = torch.sigmoid(self.edge_gate(gate_in)).squeeze(-1)
        return gate * (A > 0).to(dtype=gate.dtype)

    def _compute_adj_norm(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        dag_level: Optional[torch.Tensor],
        is_root: Optional[torch.Tensor],
    ) -> torch.Tensor:
        A_eff = A
        if self.use_edge_gate:
            A_eff = A_eff * self._compute_edge_gate(h, A, dag_level=dag_level, is_root=is_root)
        if self.use_attention:
            Q = self.attn_q(h)
            K = self.attn_k(h)
            scale = float(self.model_dim) ** 0.5
            scores = (Q @ K.T) / scale
            scores = scores.masked_fill(A_eff <= 0, float("-inf"))
            A_norm = torch.softmax(scores, dim=-1)
            A_norm = A_norm.nan_to_num(0.0)
            if self.use_edge_gate:
                A_norm = A_norm * (A_eff > 0).to(dtype=A_norm.dtype)
                A_norm = A_norm / A_norm.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return A_norm
        indeg = A_eff.sum(dim=1, keepdim=True)
        if self.norm_mode == "indegree":
            return A_eff / indeg.clamp(min=1.0)
        if self.norm_mode == "none":
            return A_eff
        raise ValueError(f"Unsupported dag_residual_norm: {self.norm_mode}")

    def forward(
        self,
        pooled: torch.Tensor,
        dag_adj: torch.Tensor,
        dag_level: Optional[torch.Tensor] = None,
        is_root: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dag_adj is None:
            return pooled
        if dag_adj.ndim != 2:
            raise ValueError(f"dag_adj must have shape [N,N], got {tuple(dag_adj.shape)}")
        if pooled.ndim != 2:
            raise ValueError(f"pooled must have shape [N,D], got {tuple(pooled.shape)}")
        if dag_adj.shape[0] != dag_adj.shape[1] or dag_adj.shape[0] != pooled.shape[0]:
            raise ValueError(f"dag_adj and pooled dimensions do not match: dag_adj={tuple(dag_adj.shape)}, pooled={tuple(pooled.shape)}")

        base_dtype = pooled.dtype
        A = dag_adj.to(device=pooled.device, dtype=torch.float32).clamp(min=0.0)
        h = pooled.float()
        for layer in self.layers:
            A_norm = self._compute_adj_norm(h, A, dag_level=dag_level, is_root=is_root)
            has_parent = (A_norm.sum(dim=1, keepdim=True) > 0).float()
            m = A_norm @ layer["msg_proj"](h)
            u = layer["pre_norm"](m)
            u = self.act(u)
            u = layer["out_proj"](u)
            u = self.dropout(u)
            u = u * has_parent
            alpha = torch.sigmoid(self.alpha_logit)
            h = h + alpha * u

        return h.to(dtype=base_dtype)


class LocalInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, raw_ts: torch.Tensor) -> torch.Tensor:
        return self.net(raw_ts)


class SpatialTransportBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                "msg_proj": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "pre_norm": nn.LayerNorm(self.hidden_dim),
                "out_proj": nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                ),
            }))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

    def forward(self, states: torch.Tensor, dag_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if dag_adj is None:
            zero_delta = torch.zeros_like(states)
            return states, zero_delta
        if states.ndim != 3:
            raise ValueError(f"states must have shape [N,T,D], got {tuple(states.shape)}")
        if dag_adj.ndim != 2 or dag_adj.shape[0] != dag_adj.shape[1] or dag_adj.shape[0] != states.shape[0]:
            raise ValueError(f"dag_adj and states dimensions do not match: dag_adj={tuple(dag_adj.shape)}, states={tuple(states.shape)}")

        base_dtype = states.dtype
        A = dag_adj.to(device=states.device, dtype=torch.float32).clamp(min=0.0)
        indeg = A.sum(dim=1, keepdim=True)
        A_norm = A / indeg.clamp(min=1.0)
        has_parent = (indeg.squeeze(1) > 0).float().view(-1, 1, 1)

        h = states.float()
        delta_total = torch.zeros_like(h)
        alpha = torch.sigmoid(self.alpha_logit)
        for layer in self.layers:
            msg = torch.einsum("nm,mtd->ntd", A_norm, layer["msg_proj"](h))
            delta = layer["pre_norm"](msg)
            delta = self.act(delta)
            delta = layer["out_proj"](delta)
            delta = self.dropout(delta)
            delta = delta * has_parent
            h = h + alpha * delta
            delta_total = delta_total + alpha * delta
        return h.to(dtype=base_dtype), delta_total.to(dtype=base_dtype)


class TemporalMixBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                "time_mix_k": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "time_mix_v": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "time_mix_r": nn.Parameter(torch.full((1, 1, self.hidden_dim), 0.5)),
                "key": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "value": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "receptance": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "output": nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                "ln": nn.LayerNorm(self.hidden_dim),
            }))
        self.dropout = nn.Dropout(self.dropout_p)

    @staticmethod
    def _shift(states: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(states[:, :1, :])
        return torch.cat([zero, states[:, :-1, :]], dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        h = states
        for layer in self.layers:
            prev = self._shift(h)
            mix_k = h * layer["time_mix_k"] + prev * (1.0 - layer["time_mix_k"])
            mix_v = h * layer["time_mix_v"] + prev * (1.0 - layer["time_mix_v"])
            mix_r = h * layer["time_mix_r"] + prev * (1.0 - layer["time_mix_r"])
            k = layer["key"](mix_k)
            v = layer["value"](mix_v)
            r = torch.sigmoid(layer["receptance"](mix_r))
            mixed = layer["output"](r * torch.tanh(k) * v)
            mixed = self.dropout(mixed)
            h = layer["ln"](h + mixed)
        return h


class RawTsPrefixSequenceClassifier(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        raw_ts_shape: Tuple[int, int] = (8, 13),
        prefix_tokens: int = 4,
        raw_ts_hidden_dim: int = 512,
        num_labels: int = NUM_LABELS,
        classifier_dropout: float = 0.1,
        dual_head_by_root: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        pooling_mode: str = "mean",
        root_aware_loss: bool = False,
        root_weight: float = 1.5,
        root_label_extra: Optional[Dict[int, float]] = None,
        max_root_sample_weight: float = 4.0,
        dag_residual_enable: bool = False,
        dag_residual_hidden_dim: int = 256,
        dag_residual_alpha_init: float = -3.0,
        dag_residual_dropout: float = 0.0,
        dag_residual_norm: str = "indegree",
        dag_residual_layers: int = 1,
        dag_residual_attention: bool = False,
        dag_residual_edge_gate: bool = False,
        dag_residual_gate_hidden_dim: int = 128,
        raw_ts_use_revin: bool = False,
        raw_ts_use_multiscale: bool = False,
        raw_ts_preprefix_graph_enable: bool = False,
        raw_ts_preprefix_graph_hidden_dim: int = 256,
        raw_ts_preprefix_graph_alpha_init: float = -3.0,
        raw_ts_preprefix_graph_dropout: float = 0.0,
        raw_ts_preprefix_graph_norm: str = "indegree",
        raw_ts_preprefix_graph_layers: int = 1,
        raw_ts_preprefix_graph_attention: bool = False,
        pcd_graph_enable: bool = False,
        pcd_hidden_dim: int = 128,
        pcd_transport_layers: int = 2,
        pcd_transport_dropout: float = 0.0,
        pcd_temporal_num_layers: int = 1,
        pcd_spatial_lambda: float = 0.0,
        pcd_root_lambda: float = 0.0,
        pcd_temporal_lambda: float = 0.0,
    ):
        super().__init__()
        self.lm = lm
        self.raw_ts_shape = tuple(raw_ts_shape)
        self.prefix_tokens = int(prefix_tokens)
        self.num_labels = int(num_labels)
        self.pooling_mode = str(pooling_mode)
        self.dual_head_by_root = bool(dual_head_by_root)
        self.root_aware_loss = bool(root_aware_loss)
        self.root_weight = float(root_weight)
        self.root_label_extra = dict(root_label_extra or {})
        self.max_root_sample_weight = float(max_root_sample_weight)
        self.dag_residual_enable = bool(dag_residual_enable)
        self.dag_residual_hidden_dim = int(dag_residual_hidden_dim)
        self.dag_residual_alpha_init = float(dag_residual_alpha_init)
        self.dag_residual_dropout = float(dag_residual_dropout)
        self.dag_residual_norm = str(dag_residual_norm)
        self.dag_residual_layers = int(dag_residual_layers)
        self.dag_residual_attention = bool(dag_residual_attention)
        self.dag_residual_edge_gate = bool(dag_residual_edge_gate)
        self.dag_residual_gate_hidden_dim = int(dag_residual_gate_hidden_dim)
        self.raw_ts_use_revin = bool(raw_ts_use_revin)
        self.raw_ts_use_multiscale = bool(raw_ts_use_multiscale)
        self.raw_ts_preprefix_graph_enable = bool(raw_ts_preprefix_graph_enable)
        self.raw_ts_preprefix_graph_hidden_dim = int(raw_ts_preprefix_graph_hidden_dim)
        self.raw_ts_preprefix_graph_alpha_init = float(raw_ts_preprefix_graph_alpha_init)
        self.raw_ts_preprefix_graph_dropout = float(raw_ts_preprefix_graph_dropout)
        self.raw_ts_preprefix_graph_norm = str(raw_ts_preprefix_graph_norm)
        self.raw_ts_preprefix_graph_layers = int(raw_ts_preprefix_graph_layers)
        self.raw_ts_preprefix_graph_attention = bool(raw_ts_preprefix_graph_attention)
        self.pcd_graph_enable = bool(pcd_graph_enable)
        self.pcd_hidden_dim = int(pcd_hidden_dim)
        self.pcd_transport_layers = int(pcd_transport_layers)
        self.pcd_transport_dropout = float(pcd_transport_dropout)
        self.pcd_temporal_num_layers = int(pcd_temporal_num_layers)
        self.pcd_spatial_lambda = float(pcd_spatial_lambda)
        self.pcd_root_lambda = float(pcd_root_lambda)
        self.pcd_temporal_lambda = float(pcd_temporal_lambda)

        embed_layer = self.lm.get_input_embeddings()
        hidden_size = int(embed_layer.embedding_dim)
        effective_shape = (self.raw_ts_shape[0] * (3 if self.raw_ts_use_multiscale else 1), self.raw_ts_shape[1])
        raw_dim = int(np.prod(effective_shape))
        mid_dim = int(max(raw_ts_hidden_dim, hidden_size))

        self.raw_ts_revin = RevINRawTs(self.raw_ts_shape[1]) if self.raw_ts_use_revin else None
        self.raw_ts_multiscale = MultiScaleRawTsEncoder(*self.raw_ts_shape) if self.raw_ts_use_multiscale else None
        effective_seq_len = effective_shape[0]
        effective_feat_dim = effective_shape[1]
        pcd_output_dim = self.pcd_hidden_dim * effective_seq_len

        self.local_interaction_encoder = (
            LocalInteractionEncoder(input_dim=effective_feat_dim, hidden_dim=self.pcd_hidden_dim)
            if self.pcd_graph_enable
            else None
        )
        self.spatial_transport_block = (
            SpatialTransportBlock(
                hidden_dim=self.pcd_hidden_dim,
                num_layers=self.pcd_transport_layers,
                dropout=self.pcd_transport_dropout,
            )
            if self.pcd_graph_enable
            else None
        )
        self.temporal_mix_block = (
            TemporalMixBlock(
                hidden_dim=self.pcd_hidden_dim,
                num_layers=self.pcd_temporal_num_layers,
                dropout=self.pcd_transport_dropout,
            )
            if self.pcd_graph_enable
            else None
        )

        self.raw_ts_projector = nn.Sequential(
            nn.LayerNorm(pcd_output_dim if self.pcd_graph_enable else raw_dim),
            nn.Linear(pcd_output_dim if self.pcd_graph_enable else raw_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, self.prefix_tokens * hidden_size),
        )
        self.raw_ts_preprefix_graph = (
            DagResidualBlock(
                model_dim=effective_shape[0] * effective_shape[1],
                hidden_dim=self.raw_ts_preprefix_graph_hidden_dim,
                alpha_init=self.raw_ts_preprefix_graph_alpha_init,
                dropout=self.raw_ts_preprefix_graph_dropout,
                norm_mode=self.raw_ts_preprefix_graph_norm,
                num_layers=self.raw_ts_preprefix_graph_layers,
                use_attention=self.raw_ts_preprefix_graph_attention,
            )
            if self.raw_ts_preprefix_graph_enable
            else None
        )
        self.dropout = nn.Dropout(float(classifier_dropout))
        if self.dual_head_by_root:
            self.root_classifier = nn.Linear(hidden_size, self.num_labels)
            self.nonroot_classifier = nn.Linear(hidden_size, self.num_labels)
            self.classifier = None
        else:
            self.classifier = nn.Linear(hidden_size, self.num_labels)
            self.root_classifier = None
            self.nonroot_classifier = None
        self.dag_residual_block = (
            DagResidualBlock(
                model_dim=hidden_size,
                hidden_dim=self.dag_residual_hidden_dim,
                alpha_init=self.dag_residual_alpha_init,
                dropout=self.dag_residual_dropout,
                norm_mode=self.dag_residual_norm,
                num_layers=self.dag_residual_layers,
                use_attention=self.dag_residual_attention,
                use_edge_gate=self.dag_residual_edge_gate,
                gate_hidden_dim=self.dag_residual_gate_hidden_dim,
            )
            if self.dag_residual_enable
            else None
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None

    @property
    def config(self):
        return self.lm.config

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_prefixed_inputs(self, input_ids, attention_mask, raw_ts, dag_adj=None, is_root=None):
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        bs = input_embeds.size(0)
        hidden = input_embeds.size(-1)
        raw_ts_in = raw_ts.to(device=input_embeds.device, dtype=torch.float32)
        if self.raw_ts_revin is not None:
            raw_ts_in = self.raw_ts_revin(raw_ts_in)
        if self.raw_ts_multiscale is not None:
            raw_ts_in = self.raw_ts_multiscale(raw_ts_in)

        transport_delta = None
        if self.pcd_graph_enable and self.local_interaction_encoder is not None:
            transport_states = self.local_interaction_encoder(raw_ts_in)
            if self.spatial_transport_block is not None:
                transport_states, transport_delta = self.spatial_transport_block(transport_states, dag_adj)
            if self.temporal_mix_block is not None:
                transport_states = self.temporal_mix_block(transport_states)
            raw_flat = transport_states.reshape(bs, -1)
        else:
            raw_flat = raw_ts_in.reshape(bs, -1)
            if self.raw_ts_preprefix_graph is not None and dag_adj is not None:
                raw_flat = self.raw_ts_preprefix_graph(raw_flat, dag_adj)

        prefix = self.raw_ts_projector(raw_flat).view(bs, self.prefix_tokens, hidden)
        prefix = prefix.to(dtype=input_embeds.dtype)
        prefix_mask = torch.ones((bs, self.prefix_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_embeds = torch.cat([prefix, input_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        aux_losses = self._compute_pcd_aux_losses(transport_delta=transport_delta, is_root=is_root) if self.pcd_graph_enable else {}
        return full_embeds, full_mask, aux_losses

    def _compute_sample_weights(self, labels: torch.Tensor, is_root: Optional[torch.Tensor]) -> torch.Tensor:
        sw = torch.ones_like(labels, dtype=torch.float32, device=labels.device)
        if self.class_weights is not None:
            cw = self.class_weights.to(labels.device)
            sw = sw * cw[labels]
        if self.root_aware_loss and is_root is not None:
            is_root = is_root.to(labels.device).float()
            sw = sw * (1.0 + (self.root_weight - 1.0) * is_root)
            for cls_id, extra_mult in self.root_label_extra.items():
                mask = (is_root > 0.5) & (labels == int(cls_id))
                if mask.any():
                    sw = torch.where(mask, sw * float(extra_mult), sw)
            if self.max_root_sample_weight is not None and self.max_root_sample_weight > 0:
                sw = torch.clamp(sw, max=float(self.max_root_sample_weight))
        return sw

    def _compute_pcd_aux_losses(
        self,
        transport_delta: Optional[torch.Tensor],
        is_root: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if transport_delta is None:
            return {}
        aux_losses: Dict[str, torch.Tensor] = {}
        delta = transport_delta.float()
        if self.pcd_spatial_lambda > 0:
            aux_losses["pcd_spatial"] = delta.mean(dim=0).abs().mean() * self.pcd_spatial_lambda
        if self.pcd_root_lambda > 0 and is_root is not None:
            root_mask = is_root.to(delta.device).float().view(-1, 1, 1)
            denom = root_mask.sum().clamp_min(1.0) * delta.shape[1] * delta.shape[2]
            aux_losses["pcd_root"] = ((delta.abs() * root_mask).sum() / denom) * self.pcd_root_lambda
        if self.pcd_temporal_lambda > 0 and delta.shape[1] > 1:
            aux_losses["pcd_temporal"] = ((delta[:, 1:, :] - delta[:, :-1, :]) ** 2).mean() * self.pcd_temporal_lambda
        return aux_losses

    def _select_logits(self, pooled: torch.Tensor, is_root: Optional[torch.Tensor] = None) -> torch.Tensor:
        dropped = self.dropout(pooled)
        if not self.dual_head_by_root:
            return self.classifier(dropped)

        root_logits = self.root_classifier(dropped)
        nonroot_logits = self.nonroot_classifier(dropped)
        if is_root is None:
            return nonroot_logits
        root_mask = is_root.to(device=dropped.device).view(-1, 1).float()
        return root_mask * root_logits + (1.0 - root_mask) * nonroot_logits

    def forward(self, input_ids=None, attention_mask=None, raw_ts=None, dag_adj=None, dag_level=None, labels=None, is_root=None, **kwargs):
        if raw_ts is None:
            raise ValueError("forward() is missing raw_ts")
        if input_ids is None or attention_mask is None:
            raise ValueError("forward() is missing input_ids/attention_mask")
        if self.dag_residual_enable and dag_adj is None:
            raise ValueError("forward() requires dag_adj when dag_residual_enable=True")

        inputs_embeds, full_mask, aux_losses = self._build_prefixed_inputs(
            input_ids,
            attention_mask,
            raw_ts,
            dag_adj=dag_adj,
            is_root=is_root,
        )
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("labels", None)
        kwargs.pop("is_root", None)
        kwargs.pop("dag_adj", None)
        kwargs.pop("dag_level", None)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        if self.pooling_mode == "last":
            last_index = full_mask.long().sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        else:
            mask = full_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if self.dag_residual_enable and self.dag_residual_block is not None:
            pooled = self.dag_residual_block(pooled, dag_adj, dag_level=dag_level, is_root=is_root)

        logits = self._select_logits(pooled, is_root=is_root)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            sample_weights = self._compute_sample_weights(labels=labels, is_root=is_root)
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            for aux_loss in aux_losses.values():
                loss = loss + aux_loss.to(loss.device, dtype=loss.dtype)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def save_final_artifacts(model: RawTsPrefixSequenceClassifier, tokenizer, out_dir: str, args: argparse.Namespace) -> None:
    ensure_dir(out_dir)
    adapter_dir = str(Path(out_dir) / "final_adapter")
    ensure_dir(adapter_dir)
    if not getattr(args, "no_lora", False):
        # Only save LoRA adapter when LoRA is used
        if hasattr(model.lm, "save_pretrained"):
            model.lm.save_pretrained(adapter_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(adapter_dir)
    torch.save(model.raw_ts_projector.state_dict(), str(Path(adapter_dir) / "raw_ts_projector.pt"))
    if getattr(model, "raw_ts_revin", None) is not None:
        torch.save(model.raw_ts_revin.state_dict(), str(Path(adapter_dir) / "raw_ts_revin.pt"))
    if getattr(model, "local_interaction_encoder", None) is not None:
        torch.save(model.local_interaction_encoder.state_dict(), str(Path(adapter_dir) / "local_interaction_encoder.pt"))
    if getattr(model, "spatial_transport_block", None) is not None:
        torch.save(model.spatial_transport_block.state_dict(), str(Path(adapter_dir) / "spatial_transport_block.pt"))
    if getattr(model, "temporal_mix_block", None) is not None:
        torch.save(model.temporal_mix_block.state_dict(), str(Path(adapter_dir) / "temporal_mix_block.pt"))
    if getattr(model, "raw_ts_preprefix_graph", None) is not None:
        torch.save(model.raw_ts_preprefix_graph.state_dict(), str(Path(adapter_dir) / "raw_ts_preprefix_graph.pt"))
    if getattr(model, "dual_head_by_root", False):
        torch.save(model.root_classifier.state_dict(), str(Path(adapter_dir) / "root_classifier.pt"))
        torch.save(model.nonroot_classifier.state_dict(), str(Path(adapter_dir) / "nonroot_classifier.pt"))
    else:
        torch.save(model.classifier.state_dict(), str(Path(adapter_dir) / "classifier_head.pt"))
    if model.dag_residual_enable and model.dag_residual_block is not None:
        torch.save(model.dag_residual_block.state_dict(), str(Path(adapter_dir) / "dag_residual_block.pt"))
    cls_cfg = {
        "num_labels": int(model.num_labels),
        "classifier_dropout": float(getattr(args, "classifier_dropout", 0.1)),
        "dual_head_by_root": bool(getattr(args, "dual_head_by_root", False)),
        "pooling_mode": str(getattr(args, "pooling_mode", "mean")),
        "root_aware_loss": bool(getattr(args, "root_aware_loss", False)),
        "root_weight": float(getattr(args, "root_weight", None) or 1.5),
        "root_label_extra": parse_root_label_extra(getattr(args, "root_label_extra", None) or ""),
        "max_root_sample_weight": float(getattr(args, "max_root_sample_weight", None) or 4.0),
    }
    raw_ts_cfg = {
        "raw_ts_shape": list(model.raw_ts_shape),
        "prefix_tokens": int(model.prefix_tokens),
        "raw_ts_hidden_dim": int(getattr(args, "raw_ts_hidden_dim", 512)),
        "raw_ts_use_revin": bool(getattr(args, "raw_ts_use_revin", False)),
        "raw_ts_use_multiscale": bool(getattr(args, "raw_ts_use_multiscale", False)),
        "raw_ts_preprefix_graph_enable": bool(getattr(args, "raw_ts_preprefix_graph_enable", False)),
        "raw_ts_preprefix_graph_hidden_dim": int(getattr(args, "raw_ts_preprefix_graph_hidden_dim", 256)),
        "raw_ts_preprefix_graph_alpha_init": float(getattr(args, "raw_ts_preprefix_graph_alpha_init", -3.0)),
        "raw_ts_preprefix_graph_dropout": float(getattr(args, "raw_ts_preprefix_graph_dropout", 0.0)),
        "raw_ts_preprefix_graph_norm": str(getattr(args, "raw_ts_preprefix_graph_norm", "indegree")),
        "raw_ts_preprefix_graph_layers": int(getattr(args, "raw_ts_preprefix_graph_layers", 1)),
        "raw_ts_preprefix_graph_attention": bool(getattr(args, "raw_ts_preprefix_graph_attention", False)),
        "pcd_graph_enable": bool(getattr(args, "pcd_graph_enable", False)),
        "pcd_hidden_dim": int(getattr(args, "pcd_hidden_dim", 128)),
        "pcd_transport_layers": int(getattr(args, "pcd_transport_layers", 2)),
        "pcd_transport_dropout": float(getattr(args, "pcd_transport_dropout", 0.0)),
        "pcd_temporal_num_layers": int(getattr(args, "pcd_temporal_num_layers", 1)),
        "pcd_spatial_lambda": float(getattr(args, "pcd_spatial_lambda", 0.0)),
        "pcd_root_lambda": float(getattr(args, "pcd_root_lambda", 0.0)),
        "pcd_temporal_lambda": float(getattr(args, "pcd_temporal_lambda", 0.0)),
    }
    root_region_cfg = {
        "root_region_env_fusion": bool(getattr(args, "root_region_env_fusion", False)),
        "root_region_env_weight": float(getattr(args, "root_region_env_weight", 0.5)),
        "root_region_exclude_self": bool(getattr(args, "root_region_exclude_self", False)),
        "region_group_primary_key": str(getattr(args, "region_group_primary_key", "target_date")),
        "auto_region_fields": list(AUTO_REGION_FIELDS),
    }
    dag_residual_cfg = {
        "dag_residual_enable": bool(getattr(args, "dag_residual_enable", False)),
        "dag_residual_hidden_dim": int(getattr(args, "dag_residual_hidden_dim", 256)),
        "dag_residual_alpha_init": float(getattr(args, "dag_residual_alpha_init", -3.0)),
        "dag_residual_dropout": float(getattr(args, "dag_residual_dropout", 0.0)),
        "dag_residual_norm": str(getattr(args, "dag_residual_norm", "indegree")),
        "dag_residual_layers": int(getattr(args, "dag_residual_layers", 1)),
        "dag_residual_attention": bool(getattr(args, "dag_residual_attention", False)),
        "dag_residual_edge_gate": bool(getattr(args, "dag_residual_edge_gate", False)),
        "dag_residual_gate_hidden_dim": int(getattr(args, "dag_residual_gate_hidden_dim", 128)),
    }
    save_json(str(Path(adapter_dir) / "cls_config.json"), cls_cfg)
    save_json(str(Path(adapter_dir) / "raw_ts_config.json"), raw_ts_cfg)
    save_json(str(Path(adapter_dir) / "root_region_config.json"), root_region_cfg)
    save_json(str(Path(adapter_dir) / "dag_residual_config.json"), dag_residual_cfg)


class DagGroupDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split_name: str,
        tokenizer,
        raw_ts_shape: Tuple[int, int],
        max_input_length: int,
        use_chat_template: bool,
        root_region_env_fusion: bool = False,
        root_region_env_weight: float = 0.5,
        root_region_exclude_self: bool = False,
        ablation_keep_cols: Optional[List[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.raw_ts_shape = raw_ts_shape
        self.max_input_length = int(max_input_length)
        self.use_chat_template = bool(use_chat_template)
        self.root_region_env_fusion = bool(root_region_env_fusion)
        self.root_region_env_weight = float(root_region_env_weight)
        self.root_region_exclude_self = bool(root_region_exclude_self)
        self.ablation_keep_cols = ablation_keep_cols

        idx_path = self.data_dir / "assets" / split_name / "group_index.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(idx_path)
        self.rows = []
        with idx_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def _render_prompt(self, prompt: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _maybe_fuse_root_raw_ts(self, raw_ts: np.ndarray, is_root: np.ndarray) -> np.ndarray:
        if not self.root_region_env_fusion:
            return raw_ts
        total = raw_ts.sum(axis=0, keepdims=False).astype(np.float32)
        count = int(raw_ts.shape[0])
        out = raw_ts.astype(np.float32).copy()
        for i in range(count):
            if int(is_root[i]) != 1:
                continue
            if self.root_region_exclude_self and count > 1:
                region_mean = (total - raw_ts[i]) / float(count - 1)
            else:
                region_mean = total / float(max(count, 1))
            out[i] = (raw_ts[i] + float(self.root_region_env_weight) * region_mean).astype(np.float32)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        city_names = _load_json(Path(row["city_names_path"]), [])
        prompt_entries = _load_json(Path(row["prompts_path"]), [])
        prompts = []
        if prompt_entries and isinstance(prompt_entries[0], dict):
            prompts = [str(x.get("prompt", "")) for x in prompt_entries]
        else:
            prompts = [str(x) for x in prompt_entries]

        raw_ts = np.load(row["raw_ts_path"]).astype(np.float32)
        labels = np.load(row["labels_path"]).astype(np.int64)
        dag_adj = np.load(row["dag_adj_path"]).astype(np.float32)
        dag_level = np.load(row["dag_level_path"]).astype(np.int64)
        is_root = np.load(row["is_root_path"]).astype(np.int64)

        if raw_ts.ndim != 3 or raw_ts.shape[1:] != self.raw_ts_shape:
            raise ValueError(f"Unexpected raw_ts shape: {raw_ts.shape}, expect=(*,{self.raw_ts_shape[0]},{self.raw_ts_shape[1]})")
        if dag_adj.ndim != 2 or dag_adj.shape[0] != dag_adj.shape[1] or dag_adj.shape[0] != raw_ts.shape[0]:
            raise ValueError(f"Unexpected dag_adj shape: {dag_adj.shape}, raw_ts={raw_ts.shape}")
        if len(prompts) != raw_ts.shape[0]:
            raise ValueError(f"number of prompts does not match raw_ts: prompts={len(prompts)}, raw_ts={raw_ts.shape}")
        if len(city_names) != raw_ts.shape[0]:
            raise ValueError(f"number of city_names does not match raw_ts: city_names={len(city_names)}, raw_ts={raw_ts.shape}")

        raw_ts = self._maybe_fuse_root_raw_ts(raw_ts=raw_ts, is_root=is_root)

        # Ablation: zero out columns not in keep list
        if self.ablation_keep_cols is not None:
            mask = np.zeros(raw_ts.shape[-1], dtype=np.float32)
            for c in self.ablation_keep_cols:
                if 0 <= c < raw_ts.shape[-1]:
                    mask[c] = 1.0
            raw_ts = raw_ts * mask[np.newaxis, np.newaxis, :]

        token_ids: List[List[int]] = []
        for prompt in prompts:
            text = self._render_prompt(prompt)
            enc = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(enc) > self.max_input_length:
                enc = enc[-self.max_input_length:]
            token_ids.append(enc)

        return {
            "group_id": str(row["group_id"]),
            "target_date": str(row.get("target_date", row["group_id"])),
            "city_names": [str(x) for x in city_names],
            "token_ids": token_ids,
            "raw_ts": raw_ts,
            "labels": labels,
            "dag_adj": dag_adj,
            "dag_level": dag_level,
            "is_root": is_root,
        }


class DagGroupCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError(f"dagresidual v1 supports only batch_size=1, got {len(features)}")
        x = features[0]
        pad_id = self.tokenizer.pad_token_id
        token_ids = x["token_ids"]
        max_len = max(len(t) for t in token_ids)
        input_ids: List[torch.Tensor] = []
        attention_mask: List[torch.Tensor] = []
        for ids in token_ids:
            pad_len = max_len - len(ids)
            input_ids.append(torch.tensor([pad_id] * pad_len + ids, dtype=torch.long))
            attention_mask.append(torch.tensor([0] * pad_len + [1] * len(ids), dtype=torch.long))
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "raw_ts": torch.tensor(x["raw_ts"], dtype=torch.float32),
            "labels": torch.tensor(x["labels"], dtype=torch.long),
            "dag_adj": torch.tensor(x["dag_adj"], dtype=torch.float32),
            "dag_level": torch.tensor(x["dag_level"], dtype=torch.long),
            "is_root": torch.tensor(x["is_root"], dtype=torch.float32),
        }


def compute_metrics_from_arrays(y_true: List[int], y_pred: List[int], num_labels: int = NUM_LABELS) -> Dict[str, float]:
    n = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / n if n > 0 else 0.0
    per_f1 = []
    tp_global = 0
    fp_global = 0
    fn_global = 0
    for lab in range(num_labels):
        tp = sum(int((yt == lab) and (yp == lab)) for yt, yp in zip(y_true, y_pred))
        fp = sum(int((yt != lab) and (yp == lab)) for yt, yp in zip(y_true, y_pred))
        fn = sum(int((yt == lab) and (yp != lab)) for yt, yp in zip(y_true, y_pred))
        tp_global += tp
        fp_global += fp
        fn_global += fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_f1.append(f1)
    micro_precision = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0.0
    micro_recall = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
    macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0
    return {"accuracy": accuracy, "micro_f1": micro_f1, "macro_f1": macro_f1}


def compute_root_nonroot_metrics(y_true: List[int], y_pred: List[int], is_root: List[int], num_labels: int = NUM_LABELS) -> Dict[str, float]:
    base = compute_metrics_from_arrays(y_true, y_pred, num_labels=num_labels)
    root_idx = [i for i, v in enumerate(is_root) if int(v) == 1]
    nonroot_idx = [i for i, v in enumerate(is_root) if int(v) == 0]
    root_true = [y_true[i] for i in root_idx]
    root_pred = [y_pred[i] for i in root_idx]
    nonroot_true = [y_true[i] for i in nonroot_idx]
    nonroot_pred = [y_pred[i] for i in nonroot_idx]
    root_metrics = compute_metrics_from_arrays(root_true, root_pred, num_labels=num_labels) if root_true else {"accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
    nonroot_metrics = compute_metrics_from_arrays(nonroot_true, nonroot_pred, num_labels=num_labels) if nonroot_true else {"accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
    out = dict(base)
    out["root_macro_f1"] = float(root_metrics["macro_f1"])
    out["nonroot_macro_f1"] = float(nonroot_metrics["macro_f1"])
    out["root_micro_f1"] = float(root_metrics["micro_f1"])
    out["nonroot_micro_f1"] = float(nonroot_metrics["micro_f1"])
    out["root_count"] = float(len(root_true))
    out["nonroot_count"] = float(len(nonroot_true))
    return out


def trainer_compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    y_pred = np.asarray(logits).argmax(axis=-1).reshape(-1).tolist()
    y_true = np.asarray(labels).reshape(-1).tolist()
    return compute_metrics_from_arrays(y_true, y_pred)


class RootAwareBestTrainer(Trainer):
    def __init__(self, *args, num_labels: int = NUM_LABELS, best_macro_tol: float = 0.003, joint_stage_bonus: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = int(num_labels)
        self.best_macro_tol = float(best_macro_tol)
        self.joint_stage_bonus = float(joint_stage_bonus)
        self.best_eval_macro_seen = -1.0

    @torch.no_grad()
    def _collect_eval_predictions(self, eval_dataset=None) -> Tuple[List[int], List[int], List[int]]:
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        was_training = model.training
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        is_root_all: List[int] = []
        for inputs in dataloader:
            labels = inputs["labels"].detach().cpu().numpy().reshape(-1).tolist()
            is_root = inputs["is_root"].detach().cpu().numpy().reshape(-1).tolist()
            prepared = self._prepare_inputs(inputs)
            outputs = model(**prepared)
            logits = outputs.logits.detach().cpu().numpy()
            pred = np.asarray(logits).argmax(axis=-1).reshape(-1).tolist()
            y_true.extend([int(x) for x in labels])
            y_pred.extend([int(x) for x in pred])
            is_root_all.extend([int(float(x) != 0.0) for x in is_root])
        if was_training:
            model.train()
        return y_true, y_pred, is_root_all

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        target_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if target_dataset is None:
            return metrics
        y_true, y_pred, is_root_all = self._collect_eval_predictions(target_dataset)
        slice_metrics = compute_root_nonroot_metrics(y_true, y_pred, is_root_all, num_labels=self.num_labels)
        current_macro = float(slice_metrics["macro_f1"])
        ref_best_macro = max(self.best_eval_macro_seen, current_macro)
        within_tol = current_macro >= (ref_best_macro - self.best_macro_tol)
        joint_score = (self.joint_stage_bonus if within_tol else 0.0) + float(slice_metrics["root_macro_f1"]) + 1e-3 * current_macro
        extra_metrics = {
            f"{metric_key_prefix}_root_macro_f1": float(slice_metrics["root_macro_f1"]),
            f"{metric_key_prefix}_nonroot_macro_f1": float(slice_metrics["nonroot_macro_f1"]),
            f"{metric_key_prefix}_root_micro_f1": float(slice_metrics["root_micro_f1"]),
            f"{metric_key_prefix}_nonroot_micro_f1": float(slice_metrics["nonroot_micro_f1"]),
            f"{metric_key_prefix}_root_count": float(slice_metrics["root_count"]),
            f"{metric_key_prefix}_nonroot_count": float(slice_metrics["nonroot_count"]),
            f"{metric_key_prefix}_within_macro_tolerance": 1.0 if within_tol else 0.0,
            f"{metric_key_prefix}_joint_score": float(joint_score),
        }
        metrics.update(extra_metrics)
        self.log(extra_metrics)
        self.best_eval_macro_seen = max(self.best_eval_macro_seen, current_macro)
        return metrics


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs: Dict[str, Any] = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=bool(args.greater_is_better),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    if args.bf16:
        kwargs["bf16"] = True
    else:
        kwargs["fp16"] = torch.cuda.is_available()
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    _normalize_parallel_styles()

    raw_ts_shape = parse_shape(args.raw_ts_shape)
    root_label_extra = parse_root_label_extra(args.root_label_extra)

    cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    _sanitize_parallel_plan(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    base_lm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    _sanitize_parallel_plan(base_lm.config)
    if args.load_in_4bit:
        # Lightweight alternative to prepare_model_for_kbit_training:
        # freeze all params + enable gradient on input embeddings, but skip
        # the expensive float32 cast that causes OOM on memory-constrained GPUs.
        for param in base_lm.parameters():
            param.requires_grad = False
        base_lm.enable_input_require_grads()
        base_lm.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )
    if args.no_lora:
        # Freeze all LLM params, no LoRA
        for param in base_lm.parameters():
            param.requires_grad = False
        base_lm.enable_input_require_grads()
        print("[ABLATION] no_lora=True: LLM weights frozen, only projector+classifier+dag_residual trainable")
    else:
        base_lm = get_peft_model(base_lm, lora_cfg)

    # Parse ablation_mask
    ablation_keep_cols = None
    if args.ablation_mask is not None:
        ablation_keep_cols = [int(x.strip()) for x in args.ablation_mask.split(",") if x.strip()]
        print(f"[ABLATION] ablation_mask: keep columns {ablation_keep_cols}")

    train_ds = DagGroupDataset(
        data_dir=args.data_dir,
        split_name=args.train_split,
        tokenizer=tokenizer,
        raw_ts_shape=raw_ts_shape,
        max_input_length=args.max_input_length,
        use_chat_template=args.use_chat_template,
        root_region_env_fusion=args.root_region_env_fusion,
        root_region_env_weight=args.root_region_env_weight,
        root_region_exclude_self=args.root_region_exclude_self,
        ablation_keep_cols=ablation_keep_cols,
    )
    eval_ds = DagGroupDataset(
        data_dir=args.data_dir,
        split_name=args.eval_split,
        tokenizer=tokenizer,
        raw_ts_shape=raw_ts_shape,
        max_input_length=args.max_input_length,
        use_chat_template=args.use_chat_template,
        root_region_env_fusion=args.root_region_env_fusion,
        root_region_env_weight=args.root_region_env_weight,
        root_region_exclude_self=args.root_region_exclude_self,
        ablation_keep_cols=ablation_keep_cols,
    ) if args.eval_split else None

    class_weights = compute_class_weights_from_group_index(train_ds.rows, max_class_weight=args.max_class_weight) if args.weighted_loss else None
    model = RawTsPrefixSequenceClassifier(
        lm=base_lm,
        raw_ts_shape=raw_ts_shape,
        prefix_tokens=args.raw_ts_prefix_tokens,
        raw_ts_hidden_dim=args.raw_ts_hidden_dim,
        num_labels=NUM_LABELS,
        classifier_dropout=args.classifier_dropout,
        dual_head_by_root=args.dual_head_by_root,
        class_weights=class_weights,
        pooling_mode=args.pooling_mode,
        root_aware_loss=args.root_aware_loss,
        root_weight=args.root_weight,
        root_label_extra=root_label_extra,
        max_root_sample_weight=args.max_root_sample_weight,
        dag_residual_enable=args.dag_residual_enable,
        dag_residual_hidden_dim=args.dag_residual_hidden_dim,
        dag_residual_alpha_init=args.dag_residual_alpha_init,
        dag_residual_dropout=args.dag_residual_dropout,
        dag_residual_norm=args.dag_residual_norm,
        dag_residual_layers=getattr(args, "dag_residual_layers", 1),
        dag_residual_attention=getattr(args, "dag_residual_attention", False),
        dag_residual_edge_gate=getattr(args, "dag_residual_edge_gate", False),
        dag_residual_gate_hidden_dim=getattr(args, "dag_residual_gate_hidden_dim", 128),
        raw_ts_use_revin=getattr(args, "raw_ts_use_revin", False),
        raw_ts_use_multiscale=getattr(args, "raw_ts_use_multiscale", False),
        raw_ts_preprefix_graph_enable=getattr(args, "raw_ts_preprefix_graph_enable", False),
        raw_ts_preprefix_graph_hidden_dim=getattr(args, "raw_ts_preprefix_graph_hidden_dim", 256),
        raw_ts_preprefix_graph_alpha_init=getattr(args, "raw_ts_preprefix_graph_alpha_init", -3.0),
        raw_ts_preprefix_graph_dropout=getattr(args, "raw_ts_preprefix_graph_dropout", 0.0),
        raw_ts_preprefix_graph_norm=getattr(args, "raw_ts_preprefix_graph_norm", "indegree"),
        raw_ts_preprefix_graph_layers=getattr(args, "raw_ts_preprefix_graph_layers", 1),
        raw_ts_preprefix_graph_attention=getattr(args, "raw_ts_preprefix_graph_attention", False),
        pcd_graph_enable=getattr(args, "pcd_graph_enable", False),
        pcd_hidden_dim=getattr(args, "pcd_hidden_dim", 128),
        pcd_transport_layers=getattr(args, "pcd_transport_layers", 2),
        pcd_transport_dropout=getattr(args, "pcd_transport_dropout", 0.0),
        pcd_temporal_num_layers=getattr(args, "pcd_temporal_num_layers", 1),
        pcd_spatial_lambda=getattr(args, "pcd_spatial_lambda", 0.0),
        pcd_root_lambda=getattr(args, "pcd_root_lambda", 0.0),
        pcd_temporal_lambda=getattr(args, "pcd_temporal_lambda", 0.0),
    )

    collator = DagGroupCollator(tokenizer)
    training_args = build_training_args(args)
    trainer = RootAwareBestTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=trainer_compute_metrics if eval_ds is not None else None,
        num_labels=NUM_LABELS,
        best_macro_tol=args.best_macro_tol,
        joint_stage_bonus=args.joint_stage_bonus,
    )

    print("=" * 100)
    print("YRD graph-aware classification training (root-region env fusion + model-side dag residual)")
    print(f"model_name_or_path      = {args.model_name_or_path}")
    print(f"data_dir                = {args.data_dir}")
    print(f"out_dir                 = {args.out_dir}")
    print(f"train_split             = {args.train_split}")
    print(f"eval_split              = {args.eval_split}")
    print(f"max_input_length        = {args.max_input_length}")
    print(f"epochs                  = {args.num_train_epochs}")
    print(f"lr                      = {args.learning_rate}")
    print(f"load_in_4bit            = {args.load_in_4bit}")
    print(f"bf16                    = {args.bf16}")
    print(f"raw_ts_shape            = {raw_ts_shape}")
    print(f"num_train_groups        = {len(train_ds)}")
    print(f"num_eval_groups         = {len(eval_ds) if eval_ds is not None else 0}")
    print(f"weighted_loss           = {args.weighted_loss}")
    print(f"dual_head_by_root       = {args.dual_head_by_root}")
    if class_weights is not None:
        print(f"class_weights           = {[round(float(x), 4) for x in class_weights.tolist()]}")
    print(f"root_aware_loss         = {args.root_aware_loss}")
    if args.root_aware_loss:
        print(f"root_weight             = {args.root_weight}")
        print(f"root_label_extra        = {root_label_extra}")
        print(f"max_root_sample_weight  = {args.max_root_sample_weight}")
    print(f"root_region_env_fusion  = {args.root_region_env_fusion}")
    if args.root_region_env_fusion:
        print(f"root_region_env_weight  = {args.root_region_env_weight}")
        print(f"root_region_exclude_self= {args.root_region_exclude_self}")
        print(f"region_group_primary_key= {args.region_group_primary_key}")
    print(f"dag_residual_enable     = {args.dag_residual_enable}")
    if args.dag_residual_enable:
        print(f"dag_residual_hidden_dim = {args.dag_residual_hidden_dim}")
        print(f"dag_residual_alpha_init = {args.dag_residual_alpha_init}")
        print(f"dag_residual_dropout    = {args.dag_residual_dropout}")
        print(f"dag_residual_norm       = {args.dag_residual_norm}")
    print("save_strategy           = epoch")
    print(f"save_total_limit        = {args.save_total_limit}")
    print(f"metric_for_best_model   = {args.metric_for_best_model}")
    print(f"best_macro_tol          = {args.best_macro_tol}")
    print(f"joint_stage_bonus       = {args.joint_stage_bonus}")
    print(f"greater_is_better       = {args.greater_is_better}")

    train_result = trainer.train()
    train_metrics = train_result.metrics
    save_json(str(Path(args.out_dir) / "train_metrics.json"), train_metrics)

    eval_metrics = {}
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        save_json(str(Path(args.out_dir) / "eval_metrics.json"), eval_metrics)

    save_final_artifacts(model, tokenizer, args.out_dir, args)

    trainer_state_summary = {}
    try:
        state = trainer.state
        trainer_state_summary = {
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "global_step": state.global_step,
            "epoch": state.epoch,
        }
        save_json(str(Path(args.out_dir) / "trainer_state_summary.json"), trainer_state_summary)
    except Exception:
        pass

    save_json(
        str(Path(args.out_dir) / "run_config.json"),
        {
            "model_name_or_path": args.model_name_or_path,
            "data_dir": args.data_dir,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "raw_ts_shape": list(raw_ts_shape),
            "max_input_length": args.max_input_length,
            "use_chat_template": bool(args.use_chat_template),
            "weighted_loss": bool(args.weighted_loss),
            "class_weights": [float(x) for x in class_weights.tolist()] if class_weights is not None else None,
            "root_aware_loss": bool(args.root_aware_loss),
            "root_weight": float(args.root_weight),
            "root_label_extra": root_label_extra,
            "max_root_sample_weight": float(args.max_root_sample_weight),
            "root_region_env_fusion": bool(args.root_region_env_fusion),
            "root_region_env_weight": float(args.root_region_env_weight),
            "root_region_exclude_self": bool(args.root_region_exclude_self),
            "region_group_primary_key": str(args.region_group_primary_key),
            "dag_residual_enable": bool(args.dag_residual_enable),
            "dag_residual_hidden_dim": int(args.dag_residual_hidden_dim),
            "dag_residual_alpha_init": float(args.dag_residual_alpha_init),
            "dag_residual_dropout": float(args.dag_residual_dropout),
            "dag_residual_norm": str(args.dag_residual_norm),
            "save_total_limit": int(args.save_total_limit),
            "metric_for_best_model": str(args.metric_for_best_model),
            "greater_is_better": bool(args.greater_is_better),
            "best_macro_tol": float(args.best_macro_tol),
            "joint_stage_bonus": float(args.joint_stage_bonus),
            "num_labels": NUM_LABELS,
            "ablation_mask": args.ablation_mask,
            "no_lora": bool(args.no_lora),
        },
    )

    print("\nTraining finished.")
    print("Note: final_adapter holds the best checkpoint, not the last epoch.")
    print(f"Saved:\n  {args.out_dir}\n  {Path(args.out_dir) / 'final_adapter'}")
    if trainer_state_summary:
        print(f"trainer_state_summary = {trainer_state_summary}")
    if eval_metrics:
        print(f"eval_metrics = {eval_metrics}")


if __name__ == "__main__":
    main()
