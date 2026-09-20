import torch
import torch.nn as nn
import torch.nn.functional as F

class LDMTTA(nn.Module):
    def __init__(self, model, optimizer, steps=3):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        if not if_adapt:
            self.model.eval()
            with torch.no_grad():
                return self.model.module.forward_output(x, device, args)
        for _ in range(self.steps):
            outputs = self._adapt(x, device, args, metric_logger)
        return outputs

    @torch.enable_grad()
    def _adapt(self, x, device, args, metric_logger):
        model = self.model.module
        img_feat = model.encode_image(x)
        text_feat = model.text_features
        sims = img_feat @ text_feat.T / args.temperature
        # Sinkhorn normalization on the similarity matrix
        sims_norm = self._sinkhorn(sims, 3)
        log_probs = F.log_softmax(sims_norm, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)
        # Reliability selection: keep samples with entropy below median
        threshold = entropy.median()
        mask = entropy <= threshold
        if mask.sum() == 0:
            mask[0] = True
        loss = entropy[mask].mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        metric_logger.update(loss_total=loss.item())
        metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        return sims.detach()

    def _sinkhorn(self, sims, num_iters):
        sims = sims.exp()
        sims = sims / sims.sum(dim=1, keepdim=True)
        for _ in range(num_iters):
            sims = sims / sims.sum(dim=1, keepdim=True)
            sims = sims / sims.sum(dim=0, keepdim=True)
        return sims