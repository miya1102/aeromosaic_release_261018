#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase F1: Adaptive class-balanced training wrapper.

Layered on top of the v1 official_best script via monkey-patching.
New CLI flags (parsed before the base script sees sys.argv):
  --auto_num_labels      (default on)  infer label count from data
  --no_auto_num_labels   disable auto inference
  --auto_class_weights   use effective-number weighting
  --class_balance_beta F (default 0.999)
  --focal_gamma F        (default 0.0, i.e. off; 2.0 to enable)
"""

from __future__ import annotations

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# ---- parse our custom flags BEFORE importing the base script ----
_auto_num_labels = True
_auto_class_weights = False
_class_balance_beta = 0.999
_focal_gamma = 0.0

_filtered = []
_i = 0
_argv = sys.argv[1:]
while _i < len(_argv):
    a = _argv[_i]
    if a == "--auto_num_labels":
        _auto_num_labels = True; _i += 1
    elif a == "--no_auto_num_labels":
        _auto_num_labels = False; _i += 1
    elif a == "--auto_class_weights":
        _auto_class_weights = True; _i += 1
    elif a == "--class_balance_beta":
        _class_balance_beta = float(_argv[_i+1]); _i += 2
    elif a == "--focal_gamma":
        _focal_gamma = float(_argv[_i+1]); _i += 2
    else:
        _filtered.append(a); _i += 1

# If auto_class_weights, inject --weighted_loss for the base parser
if _auto_class_weights and "--weighted_loss" not in _filtered:
    _filtered.append("--weighted_loss")
sys.argv = [sys.argv[0]] + _filtered

# Now import the base script
from train_yrd_graph_cls_v2_rootregion_best_dagresidual import *
import train_yrd_graph_cls_v2_rootregion_best_dagresidual as _mod


# ---- Step 1: auto_num_labels ----

def infer_num_labels(group_rows, default=6):
    labels_seen = set()
    for row in group_rows:
        labels = np.load(row["labels_path"])
        labels_seen.update(int(y) for y in labels.tolist())
    return max(labels_seen) + 1 if labels_seen else default


# ---- Step 2: effective-number class weights ----

def compute_effective_number_weights(group_rows, num_labels, beta=0.999):
    counts = np.zeros(num_labels, dtype=np.float64)
    for row in group_rows:
        labels = np.load(row["labels_path"])
        for y in labels.tolist():
            y = int(y)
            if 0 <= y < num_labels:
                counts[y] += 1.0
    counts[counts <= 0] = 1.0
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ---- Step 3: focal loss patch ----

_OrigCls = _mod.RawTsPrefixSequenceClassifier
_orig_forward = _OrigCls.forward


def _forward_focal(self, input_ids=None, attention_mask=None, raw_ts=None,
                   dag_adj=None, dag_level=None, labels=None,
                   is_root=None, **kwargs):
    gamma = getattr(self, "_focal_gamma", 0.0)
    if gamma <= 0:
        return _orig_forward(self, input_ids=input_ids,
                             attention_mask=attention_mask, raw_ts=raw_ts,
                             dag_adj=dag_adj, dag_level=dag_level,
                             labels=labels, is_root=is_root, **kwargs)

    # Replicate forward but with focal modulation on the loss
    if raw_ts is None:
        raise ValueError("forward() is missing raw_ts")
    if input_ids is None or attention_mask is None:
        raise ValueError("forward() is missing input_ids/attention_mask")
    if self.dag_residual_enable and dag_adj is None:
        raise ValueError("forward() requires dag_adj when dag_residual_enable=True")

    inputs_embeds, full_mask = self._build_prefixed_inputs(
        input_ids, attention_mask, raw_ts)
    for k in ("inputs_embeds", "input_ids", "labels", "is_root",
              "dag_adj", "dag_level"):
        kwargs.pop(k, None)

    outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                      output_hidden_states=True, return_dict=True,
                      use_cache=False, **kwargs)
    hidden = outputs.hidden_states[-1]
    if self.pooling_mode == "last":
        idx = full_mask.long().sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    else:
        mask = full_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    if self.dag_residual_enable and self.dag_residual_block is not None:
        pooled = self.dag_residual_block(
            pooled, dag_adj, dag_level=dag_level, is_root=is_root)

    logits = self._select_logits(pooled, is_root=is_root)
    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** gamma) * ce
        sw = self._compute_sample_weights(labels=labels, is_root=is_root)
        loss = (focal * sw).sum() / sw.sum().clamp_min(1e-6)

    from transformers.modeling_outputs import SequenceClassifierOutput
    return SequenceClassifierOutput(loss=loss, logits=logits)


_OrigCls.forward = _forward_focal


# ---- Patch the model-creation part of main() ----

_orig_main = _mod.main


def main_balanced():
    """Wraps the original main() to inject F1 features."""
    # We need to intercept model creation. The cleanest way is to
    # monkey-patch NUM_LABELS and compute_class_weights_from_group_index.

    # Step 1: auto_num_labels — we patch after dataset is loaded
    # Step 2: auto_class_weights — we replace the weight function

    if _auto_class_weights:
        _orig_compute = _mod.compute_class_weights_from_group_index

        def _compute_balanced(group_rows, num_labels=None, **kw):
            nl = num_labels or _mod.NUM_LABELS
            w = compute_effective_number_weights(
                group_rows, nl, beta=_class_balance_beta)
            print(f"[F1] effective-number weights (beta={_class_balance_beta}): "
                  f"{[round(float(x), 4) for x in w.tolist()]}")
            return w

        _mod.compute_class_weights_from_group_index = _compute_balanced

    # Step 1: patch NUM_LABELS via the model constructor
    if _auto_num_labels:
        _orig_init = _OrigCls.__init__

        def _init_auto(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # After init, store focal gamma
            self._focal_gamma = _focal_gamma

        # We need a different approach: intercept after dataset creation
        # Patch the module-level NUM_LABELS right before model creation
        pass

    # The simplest reliable approach: just call the original main,
    # but patch NUM_LABELS dynamically.
    # We do this by wrapping RawTsPrefixSequenceClassifier.__init__
    _orig_cls_init = _OrigCls.__init__

    _num_labels_override = [None]  # mutable container for closure

    def _patched_init(self, *a, **kw):
        if _num_labels_override[0] is not None:
            kw["num_labels"] = _num_labels_override[0]
        _orig_cls_init(self, *a, **kw)
        self._focal_gamma = _focal_gamma

    _OrigCls.__init__ = _patched_init

    # Patch DagGroupDataset to capture rows for num_labels inference
    _orig_ds_init = _mod.DagGroupDataset.__init__
    _train_rows = [None]

    def _ds_init_capture(self, *a, **kw):
        _orig_ds_init(self, *a, **kw)
        split = kw.get("split_name", a[1] if len(a) > 1 else "")
        if split == "train" or (hasattr(self, "split_name") and self.split_name == "train"):
            _train_rows[0] = self.rows
            if _auto_num_labels:
                nl = infer_num_labels(self.rows, default=_mod.NUM_LABELS)
                _num_labels_override[0] = nl
                _mod.NUM_LABELS = nl
                print(f"[F1] auto_num_labels: inferred {nl} classes (from train)")
        elif _auto_num_labels and hasattr(self, "rows"):
            nl_val = infer_num_labels(self.rows, default=0)
            prev = _num_labels_override[0] or _mod.NUM_LABELS
            if nl_val > prev:
                _num_labels_override[0] = nl_val
                _mod.NUM_LABELS = nl_val
                print(f"[F1] auto_num_labels: expanded to {nl_val} classes (val has more)")

    _mod.DagGroupDataset.__init__ = _ds_init_capture

    # Also patch compute_class_weights to use the correct num_labels
    if _auto_class_weights:
        _orig_cw = _mod.compute_class_weights_from_group_index

        def _cw_patched(group_rows, num_labels=None, **kw):
            nl = _num_labels_override[0] or _mod.NUM_LABELS
            return compute_effective_number_weights(
                group_rows, nl, beta=_class_balance_beta)

        _mod.compute_class_weights_from_group_index = _cw_patched

    # Print config
    print("=" * 80)
    print("[F1] Phase F1: Adaptive class-balanced training")
    print(f"[F1]   auto_num_labels    = {_auto_num_labels}")
    print(f"[F1]   auto_class_weights = {_auto_class_weights} (beta={_class_balance_beta})")
    print(f"[F1]   focal_gamma        = {_focal_gamma}")
    print("=" * 80)

    # Run original main
    _orig_main()

    # Save F1 config alongside the run
    import argparse as _ap
    args = _mod.parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        f1_cfg = {
            "auto_num_labels": _auto_num_labels,
            "actual_num_labels": _num_labels_override[0] or _mod.NUM_LABELS,
            "auto_class_weights": _auto_class_weights,
            "class_balance_beta": _class_balance_beta,
            "focal_gamma": _focal_gamma,
        }
        with open(out_dir / "f1_config.json", "w") as f:
            json.dump(f1_cfg, f, indent=2)


if __name__ == "__main__":
    main_balanced()
