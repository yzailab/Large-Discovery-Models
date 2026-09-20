"""continual-learning 的接线。"""
from __future__ import annotations

import os
import shutil
import sys

from .base import TaskProfile


class ContinualLearningProfile(TaskProfile):
    def grade_argv(self, workspace, slot, summary_path, extra):
        # grade.py rewrites the markdown it is pointed at; aim it at a scratch
        # copy so the description the proposal prompt reads stays pristine.
        scratch_md = summary_path.parent / "task_description.md"
        source_md = workspace / "task_description.md"
        if source_md.exists():
            shutil.copy(source_md, scratch_md)
        else:
            scratch_md.write_text("", encoding="utf-8")
        return [
            sys.executable,
            str(workspace / "grading" / "grade.py"),
            "--logs", str(workspace / "logs"),
            "--json-out", str(summary_path),
            "--md", str(scratch_md),
            "--table", os.environ.get("RG_CL_TABLE", "cifar100"),
            *extra,
        ]

    def scalarize(self, summary):
        """Flatten grade.py's table payload; Acc is the headline, AAA a diagnostic.

        ``{"tables": [{"datasets": [{"dataset", "num_tasks", "num_runs",
        "metrics": {"acc_mean", "aaa_mean", ...}}]}]}``. A cell with no runs
        carries ``metrics: null`` and is dropped, so an empty logs directory
        yields no objective rather than a plausible-looking zero.
        """
        metrics: dict[str, float] = {}
        accuracies: list[float] = []
        for table in summary.get("tables") or []:
            for entry in table.get("datasets") or []:
                stats = entry.get("metrics")
                if not isinstance(stats, dict):
                    continue
                cell = f"{entry.get('dataset')}.N{entry.get('num_tasks')}"
                acc = stats.get("acc_mean")
                if isinstance(acc, (int, float)) and not isinstance(acc, bool):
                    metrics[f"{cell}.acc"] = float(acc)
                    accuracies.append(float(acc))
                    runs = entry.get("num_runs")
                    metrics[f"{cell}.acc.count"] = float(runs or 0)
                for name in ("aaa_mean", "acc_se", "aaa_se"):
                    value = stats.get(name)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"diag_{cell}.{name}"] = float(value)
        if accuracies:
            metrics["official_score"] = sum(accuracies) / len(accuracies)
        return metrics


CL_SYSTEM_PROMPT = '''You design continual-learning methods for a pre-trained ViT-B/16 backbone. \
Reply with JSON only: {"candidates": [{"program": "<python source>"}]}. Each program is the full \
source of one module that defines a top-level `class Learner(BaseLearner)`, imported as \
`from models.ldm_method import Learner` and constructed as `Learner(args)`.

Contract, enforced by the task's own trainer.py:
- `from models.base import BaseLearner` and subclass it. The trainer calls, once per task:
  `model.incremental_train(data_manager)`, then `model.eval_task()`, then `model.after_task()`.
- `incremental_train` must bump `self._cur_task`, set `self._total_classes =
  self._known_classes + data_manager.get_task_size(self._cur_task)`, call
  `self._network.update_fc(self._total_classes)`, and assign `self.test_loader`
  (BaseLearner.eval_task reads it). `after_task` must set
  `self._known_classes = self._total_classes`.
- `self._network(x)` returns a DICT, not a tensor: {"logits": ..., "features": ...}. Use
  `self._network(x)["logits"]` in a loss, and `self._network.extract_vector(x)` for features.
- The backbone is already on `self._device`. Any module, parameter or buffer you add is
  created on CPU, so move it: build it with `.to(self._device)` and call
  `self._network.to(self._device)` after you modify the network, or the first forward
  dies with "Expected all tensors to be on the same device".
- DataLoader batches are `(index, inputs, targets)` TRIPLES, not pairs. Targets are absolute
  class indices, so subtract `self._known_classes` when training only the current head.
- `args` is the config dict: nb_classes, nb_tasks, tuned_epoch, init_lr, batch_size,
  weight_decay, min_lr, device (list), backbone_type, dataset.
- Networks live in `utils.inc_net` (SimpleVitNet, SimpleCosineIncrementalNet,
  IncrementalNet). torch, timm, numpy, tqdm are importable. No network access.
- `SimpleCosineIncrementalNet`'s head is a `CosineLinear`. It has `weight`, a scalar
  `sigma` and `out_features` -- and NO `bias`. Reaching for `self._network.fc.bias` is
  the single most common way a candidate dies here: it killed 9 of 15 evaluations in one
  campaign. Its forward L2-normalizes both the feature and the weight, so the logits are
  cosine similarities scaled by `sigma`, and `update_fc` copies the old rows into the new
  head -- class prototypes you write into `fc.weight.data` survive the next task.

Reference: the task's own SimpleCIL baseline, which your program must match in interface:
```python
{simplecil}
```

What is scored: top-1 accuracy over EVERY class seen so far, measured after each of the
10 tasks. The campaign maximizes Acc -- accuracy after the final task, over all 100 classes --
and reports AAA, the mean of those 10 measurements, alongside; both are averaged over seeds
1992-1994. Nothing is ever scored on one task in isolation: a learner that fits each task's
own 10 classes and is then asked to separate all 100 lands near 25 Acc however well it fits.
The SimpleCIL baseline above reaches roughly 80 Acc / 87 AAA with a frozen backbone.

Rules that decide whether a candidate is scored at all:
- The method must be REHEARSAL-FREE (no replay buffer, `memory_size` is 0) and must not
  grow a prompt/LoRA pool with the task count -- that is the limitation the task asks you
  to beat. Inference cost must stay constant in the number of tasks.
- It runs on CIFAR-100 split into 10 tasks of 10 classes, three seeds. A run that crashes
  or does not finish all 10 tasks is rejected, not merely scored lower.
- Wall clock is the binding constraint. Build DataLoaders with `num_workers=8,
  pin_memory=True`; keep `tuned_epoch` small; fine-tune a small parameter subset
  (LoRA/adapter/SSF-style on the first task only) rather than the whole backbone.
- Do not hardcode a class count, a task count or a dataset name.

Hard limits: each program stays under 150 lines and under 8000 characters, and never
repeats the same statement more than twice. A reply that runs past the token budget is
discarded whole, so finish every program you start.

Emit the program source as a single JSON string with escaped newlines; do not wrap it in
markdown fences.'''


CL_EXPLORATION_AXES: tuple[str, ...] = (
    "what is trained at all -- keep the backbone frozen and build the classifier from "
    "class-mean prototypes or a cosine head, instead of gradient-training a head per task",
    "first-task-only parameter-efficient adaptation (SSF / adapter / LoRA-style) followed by "
    "a frozen prototype classifier for every later task",
    "the loss over the FULL head -- train logits over all classes seen so far, protecting the "
    "old-class logits with masking, a margin or a distillation term instead of dropping them",
    "closed-form classifiers over frozen features -- ridge regression, Mahalanobis or a "
    "class-covariance head recomputed as each task arrives",
    "prototype drift -- estimate and compensate the shift of old-class prototypes after a "
    "task changes the features, rather than retraining the old classes",
    "logit calibration across tasks -- per-task normalization, temperature or bias correction "
    "that equalizes old-class and new-class scores at inference",
    "replay-free regularization -- importance penalties (EWC/MAS-style) or projecting updates "
    "onto the null space of the old-class features",
    "merging per-task parameter deltas into one constant-cost network (weight averaging or "
    "task arithmetic), so inference cost does not grow with the task count",
)
