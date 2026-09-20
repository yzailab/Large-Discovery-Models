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
            with torch.no_grad():
                return self.model.module.forward_output(x, device, args)
        for _ in range(self.steps):
            outputs = self._adapt_step(x, device, args, metric_logger)
        return outputs

    @torch.enable_grad()
    def _adapt_step(self, x, device, args, metric_logger):
        model = self.model.module
        # Forward pass to get image features and similarity matrix
        img_feat = model.encode_image(x)  # B x D, L2-normalized
        text_feat = model.text_features  # N x D, L2-normalized
        sims = img_feat @ text_feat.t() / args.temperature

        # Compute softmax probabilities over gallery
        probs = sims.softmax(dim=1)
        log_probs = sims.log_softmax(dim=1)
        entropy = -(probs * log_probs).sum(dim=1)  # B

        # Reliability weighting: lower entropy -> higher weight
        weights = torch.exp(-entropy).clamp(min=1e-6)
        weights = weights / weights.sum()

        # Entropy loss weighted by reliability
        loss = (weights * entropy).sum()

        # Add a contrastive loss to encourage uniformity of query features
        # Use a simple InfoNCE-like loss with the batch itself as negatives
        # Normalize features again for cosine similarity
        q_norm = F.normalize(img_feat, dim=1)
        sim_matrix = q_norm @ q_norm.t() / args.temperature
        # Exclude diagonal (self-similarity) by masking
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)
        # Contrastive loss: maximize similarity to a pseudo-positive? We don't have labels.
        # Instead, we minimize the average similarity to other queries to push uniformity.
        # Use a margin-based loss: encourage off-diagonal similarities to be low.
        margin = 0.1
        contrastive_loss = torch.relu(sim_matrix + margin).mean()

        total_loss = loss + 0.1 * contrastive_loss

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        metric_logger.update(loss_total=total_loss.item())
        metric_logger.update(lr=self.optimizer.param_groups[0]['lr'])

        # Return the similarity matrix (post-adaptation)
        return sims.detach()