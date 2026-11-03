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

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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


class DagResidualBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = 256,
        alpha_init: float = -3.0,
        dropout: float = 0.0,
        norm_mode: str = "indegree",
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)
        self.dropout_p = float(dropout)
        self.norm_mode = str(norm_mode)

        self.msg_proj = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.pre_norm = nn.LayerNorm(self.model_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.out_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.model_dim, bias=False),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, pooled: torch.Tensor, dag_adj: torch.Tensor, dag_level: Optional[torch.Tensor] = None, is_root: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        indeg = A.sum(dim=1, keepdim=True)
        if self.norm_mode == "indegree":
            A_norm = A / indeg.clamp(min=1.0)
        elif self.norm_mode == "none":
            A_norm = A
        else:
            raise ValueError(f"Unsupported dag_residual_norm: {self.norm_mode}")

        h = pooled.float()
        m = A_norm @ self.msg_proj(h)
        u = self.pre_norm(m)
        u = self.act(u)
        u = self.out_proj(u)
        u = self.dropout(u)

        has_parent = (indeg.squeeze(1) > 0).float().unsqueeze(1)
        u = u * has_parent

        alpha = torch.sigmoid(self.alpha_logit)
        out = h + alpha * u
        return out.to(dtype=base_dtype)


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

        embed_layer = self.lm.get_input_embeddings()
        hidden_size = int(embed_layer.embedding_dim)
        raw_dim = int(np.prod(self.raw_ts_shape))
        mid_dim = int(max(raw_ts_hidden_dim, hidden_size))

        self.raw_ts_projector = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, self.prefix_tokens * hidden_size),
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

    def _build_prefixed_inputs(self, input_ids, attention_mask, raw_ts):
        input_embeds = self.lm.get_input_embeddings()(input_ids)
        bs = input_embeds.size(0)
        hidden = input_embeds.size(-1)
        # The group dataset performs root-region fusion first. Mask only now,
        # before numerical projection, as specified by the paper.
        if self.raw_ts_shape == (8, 13):
            keep = torch.zeros((13,), device=raw_ts.device, dtype=raw_ts.dtype)
            keep[[0, 11, 12]] = 1
            raw_ts = raw_ts * keep
        raw_flat = raw_ts.to(device=input_embeds.device, dtype=torch.float32).reshape(bs, -1)
        prefix = self.raw_ts_projector(raw_flat).view(bs, self.prefix_tokens, hidden)
        prefix = prefix.to(dtype=input_embeds.dtype)
        prefix_mask = torch.ones((bs, self.prefix_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_embeds = torch.cat([prefix, input_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return full_embeds, full_mask

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

        inputs_embeds, full_mask = self._build_prefixed_inputs(input_ids, attention_mask, raw_ts)
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
            # Paper objective is (1/N) sum_i w_i CE_i, not weight-normalized CE.
            loss = (per_sample_loss * sample_weights).mean()
        return SequenceClassifierOutput(loss=loss, logits=logits)


def save_final_artifacts(model: RawTsPrefixSequenceClassifier, tokenizer, out_dir: str, args: argparse.Namespace) -> None:
    ensure_dir(out_dir)
    adapter_dir = str(Path(out_dir) / "final_adapter")
    ensure_dir(adapter_dir)
    if hasattr(model.lm, "save_pretrained"):
        model.lm.save_pretrained(adapter_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(adapter_dir)
    torch.save(model.raw_ts_projector.state_dict(), str(Path(adapter_dir) / "raw_ts_projector.pt"))
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
        "root_weight": float(getattr(args, "root_weight", 1.5)),
        "root_label_extra": parse_root_label_extra(getattr(args, "root_label_extra", "")),
        "max_root_sample_weight": float(getattr(args, "max_root_sample_weight", 4.0)),
    }
    raw_ts_cfg = {
        "raw_ts_shape": list(model.raw_ts_shape),
        "prefix_tokens": int(model.prefix_tokens),
        "raw_ts_hidden_dim": int(getattr(args, "raw_ts_hidden_dim", 512)),
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    except Exception:
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
    base_lm = get_peft_model(base_lm, lora_cfg)

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
