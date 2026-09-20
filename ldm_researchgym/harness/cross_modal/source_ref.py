import torch
import torch.nn as nn


class SourceRef(nn.Module):
    """No adaptation at all: the frozen source model under the same data protocol.

    Gives the `BLIP ViT-B/16` row of Table 1 on whatever subset the configs point at,
    so a candidate's number can be read against "did adaptation help" rather than only
    against the other baselines.
    """

    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.module.forward_output(x, device, args)
        # main.py averages these meters afterwards; a meter with zero updates divides by zero.
        metric_logger.update(loss_total=0.0)
        metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        return outputs
