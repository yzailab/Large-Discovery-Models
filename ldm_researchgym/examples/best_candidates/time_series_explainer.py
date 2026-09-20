"""LDM candidate explainer (variant 15).

Contract, mirroring the baseline blocks in ``real/main.py``:
``attribute`` consumes the trained classifier plus the task's test loader and
returns one attribution tensor shaped like the stacked inputs (N, T, F).
"""

import torch as th
from captum._utils.common import _run_forward


def attribute(classifier, test_loader, timesteps, device,
              n_steps=8, segments=8,
              baseline="zero", smooth=3, direction_aware=False):
    outputs = []
    for batch in test_loader:
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        steps = timesteps[: x_batch.shape[0], :]

        with th.autograd.set_grad_enabled(False):
            logits = _run_forward(
                classifier, x_batch, additional_forward_args=(data_mask, steps, False)
            )
        target = th.argmax(logits, -1)

        ref = th.zeros_like(x_batch) if baseline == "zero" else x_batch.mean(0, keepdim=True)
        total = th.zeros_like(x_batch)
        for step in range(1, n_steps + 1):
            alpha = float(step) / n_steps
            point = (ref + alpha * (x_batch - ref)).requires_grad_(True)
            score = _run_forward(
                classifier, point, additional_forward_args=(data_mask, steps, False)
            ).gather(-1, target.unsqueeze(-1)).sum()
            grad = th.autograd.grad(score, point, retain_graph=False)[0]
            total = total + grad

        saliency = (x_batch - ref) * total / float(n_steps)
        if smooth > 0:
            saliency = _temporal_smooth(saliency, smooth)
        outputs.append((saliency if direction_aware else saliency.abs()).detach().cpu())
    return th.cat(outputs, dim=0)


def _temporal_smooth(values, window):
    kernel = th.ones(1, 1, window, device=values.device) / float(window)
    shape = values.shape
    flat = values.permute(0, 2, 1).reshape(-1, 1, shape[1])
    smoothed = th.nn.functional.conv1d(flat, kernel, padding=window // 2)[..., : shape[1]]
    return smoothed.reshape(shape[0], shape[2], shape[1]).permute(0, 2, 1)