"""cross-modal-retrieval 的接线。"""
from __future__ import annotations

import sys

from .base import TaskProfile


class CrossModalRetrievalProfile(TaskProfile):
    def grade_argv(self, workspace, slot, summary_path, extra):
        return [
            sys.executable,
            str(workspace / "grading" / "grade.py"),
            "--root", str(workspace / "output" / slot),
            "--table", "qs_image",
            "--json-out", str(summary_path),
            "--no-markdown",
            *extra,
        ]

    def scalarize(self, summary):
        """Flatten grade.py's row payload; the row average is the headline number.

        ``{"rows": [{"row", "average", "columns": [{"id", "value", "sources"}]}]}``.
        grade.py drops a row entirely when any of the 16 corruption columns is missing,
        so a partial sweep yields no objective rather than a flattering average over
        whichever corruptions happened to survive.
        """
        metrics: dict[str, float] = {}
        averages: list[float] = []
        for row in summary.get("rows") or []:
            row_key = row.get("row")
            for column in row.get("columns") or []:
                value = column.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cell = f"COCO-C.{column.get('id')}"
                    metrics[cell] = float(value)
                    metrics[f"{cell}.count"] = float(len(column.get("sources") or []))
            average = row.get("average")
            if isinstance(average, (int, float)) and not isinstance(average, bool):
                averages.append(float(average))
                metrics[f"diag_{row_key}.average"] = float(average)
        if averages:
            metrics["official_score"] = sum(averages) / len(averages)
        return metrics


CMR_SYSTEM_PROMPT = '''You design test-time adaptation methods for cross-modal retrieval under query shift. \
Reply with JSON only: {"candidates": [{"program": "<python source>"}]}. Each program is the full source \
of one module that defines a top-level `class LDMTTA(nn.Module)`, imported as \
`from models.tta_baselines.ldm_method import LDMTTA` and constructed as \
`LDMTTA(model, optimizer, steps=args.tta_steps)`.

Contract, enforced by the task's own main.py:
- `forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False)` is the
  only entry point. `x` is one mini-batch of QUERY images (B x 3 x 384 x 384, already on `device`).
- Return the similarity matrix for that batch, shape B x N_gallery. Those rows ARE the scored
  retrieval predictions: row i ranks every gallery caption for query image i. Return the
  post-adaptation scores.
- `self.model` is a DistributedDataParallel wrapper, so reach the network through `self.model.module`:
    `forward_output(x, device, args)` -> image_feat @ text_features.T / args.temperature (0.01 here)
    `encode_image(x)` -> L2-normalized query features (B x 256), differentiable
    `self.model.module.text_features` -> the FIXED gallery text features (N x 256, L2-normalized)
- Only the LayerNorm weight/bias of the visual encoder require grad, and `self.optimizer` (AdamW,
  lr 3e-4) already holds exactly those. Do not unfreeze anything else and do not rebuild the optimizer.
- Adaptation is ONLINE and continual: each batch arrives once, in order, with no labels, no source
  data and no reset in between. Do `self.steps` (=3) forward/adapt steps per batch and call
  `self.optimizer.zero_grad()` after each `self.optimizer.step()`.
- You MUST call `metric_logger.update(loss_total=<float>, lr=<float>)` once per batch; main.py
  averages those meters afterwards and a meter with zero updates crashes the run.
- The caller holds `torch.no_grad()`, so decorate the adapting helper with `@torch.enable_grad()`.
- torch, torch.nn as nn, torch.nn.functional as F and numpy are importable. No network access.

Reference: the task's own Tent baseline, which your program must match in interface:
```python
{tent}
```

What is scored: Recall@1 of image-to-text retrieval (txt_r1) on COCO-C with BLIP ViT-B/16, over all
16 image corruptions at severity 5, averaged. The source model without adaptation sits around 45 on
the paper's protocol and the published TTA baselines land between 43 and 49 -- entropy minimization
alone is not enough, and on some corruptions it actively hurts. The signal worth exploiting lives in
the entropy of the sims, the modality gap between query and gallery features, the uniformity of the
query features, and in deciding which queries to trust at all.

Rules that decide whether a candidate is scored at all:
- One corruption that crashes or emits no metrics invalidates the whole candidate rather than
  lowering its score. Guard against empty selections, division by zero and NaN.
- Stay near Tent's wall clock: 3 steps per batch, no extra sweeps over the gallery, no Python loop
  over the thousands of gallery entries.
- Do not hardcode the gallery size, the batch size or a corruption name, and do not import another
  module from models.tta_baselines -- write the method yourself.

Hard limits: each program stays under 150 lines and under 8000 characters, and never repeats the
same statement more than twice. A reply that runs past the token budget is discarded whole, so
finish every program you start.

Emit the program source as a single JSON string with escaped newlines; do not wrap it in
markdown fences.'''


CMR_EXPLORATION_AXES: tuple[str, ...] = (
    "the adaptation objective itself -- something other than plain prediction entropy "
    "(class-balanced or marginal-regularized entropy, sharpness-aware minimization, "
    "a contrastive or ranking loss built from the batch's own similarity matrix)",
    "which queries are adapted on at all -- reliability selection and per-sample "
    "weighting from confidence, margin, entropy quantiles or agreement across steps",
    "the modality gap -- explicitly aligning the query-feature distribution to the "
    "fixed gallery text features (mean/covariance shift, prototype pull, OT coupling)",
    "the geometry of the query features -- uniformity, whitening, hubness correction "
    "or a momentum anchor that keeps the adapted encoder near the source one",
    "the similarity matrix as an output to calibrate -- temperature, Sinkhorn or "
    "doubly-stochastic normalization, query-side and gallery-side score centering",
    "state carried across batches -- a feature/prototype bank, EMA of the adapted "
    "LayerNorm parameters, or a restore rule that undoes a harmful update",
    "consistency across views -- augmentation, dropout or multi-crop agreement on the "
    "same query, used as the adaptation signal instead of a single-view loss",
    "the control policy -- adaptive learning rate, early exit on unreliable batches, "
    "gradient clipping or step count decided per batch from a reliability statistic",
)
