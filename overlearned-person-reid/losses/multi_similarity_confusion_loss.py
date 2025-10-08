import torch
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

class MultiSimilarityConfusionLoss(MultiSimilarityLoss):
    """
    Multi-Similarity Loss with an added confusion (de-identification) term for a forbidden class.

    For the forbidden class (or classes) specified by forbidden_label, this loss adds a penalty
    whenever the cosine similarity between any two forbidden samples is higher than forbidden_margin.

    Args:
        forbidden_label: int or list of ints specifying the class(es) to be de-identified.
        forbidden_margin: The similarity threshold above which forbidden pairs incur a penalty.
                          (Default: 0.5)
        confusion_weight: Weighting factor for the confusion loss term. (Default: 1.0)
        confusion_gamma: Exponential weight for the confusion term (similar in spirit to alpha/beta). (Default: 2.0)
        alpha, beta, base: Parameters passed to the standard Multi-Similarity Loss.
    """
    def __init__(self, forbidden_label, forbidden_margin=0.5, confusion_weight=1.0, confusion_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self._forbidden_label = forbidden_label
        self.forbidden_margin = forbidden_margin
        self.confusion_weight = confusion_weight
        self.confusion_gamma = confusion_gamma
        # Metrics to track
        self.forbidden_pairs_fraction = 0.0
        # We'll store the average MS loss and confusion loss for logging
        self.ms_loss_value = 0.0
        self.conf_loss_value_scalar = 0.0

    @property
    def forbidden_label(self):
        return self._forbidden_label

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # Instead of directly using super().compute_loss, we'll intercept and modify it
        # to avoid competing objectives between MS Loss and confusion loss
        
        # Convert indices_tuple for pair-based losses:
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()

        # Compute the distance matrix
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indices_tuple

        # Create masks for positive and negative pairs
        pos_mask = torch.zeros_like(mat)
        neg_mask = torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1

        # Identify forbidden label samples
        if isinstance(self.forbidden_label, (list, tuple)):
            forbidden_mask_a = torch.zeros_like(labels, dtype=torch.bool)
            for fl in self.forbidden_label:
                forbidden_mask_a |= (labels == fl)
        else:
            forbidden_mask_a = (labels == self.forbidden_label)

        # Zero out positive pairs where anchors have forbidden labels
        # This ensures MS Loss only pushes forbidden samples away from other classes
        # but doesn't pull them closer to anything
        for i, anchor_idx in enumerate(a1):
            if forbidden_mask_a[anchor_idx]:
                pos_mask[anchor_idx, p[i]] = 0

        # Compute modified MS loss
        ms_loss_dict = super()._compute_loss(mat, pos_mask, neg_mask)
        loss_dict = {"loss": ms_loss_dict["loss"]}
        
        # Extract the original MS loss
        original_losses = loss_dict["loss"]["losses"]
        if isinstance(original_losses, torch.Tensor):
            if original_losses.ndim > 0:  # per-element losses
                self.ms_loss_value = original_losses.mean().item()
            else:
                self.ms_loss_value = original_losses.item()
        else:
            # If it's a python float or int
            self.ms_loss_value = float(original_losses)

        # If no labels, we cannot compute confusion loss
        if labels is None:
            self.conf_loss_value_scalar = 0.0
            return loss_dict

        # We already have the forbidden mask from above
        forbidden_mask = forbidden_mask_a

        # If fewer than two forbidden samples, no confusion penalty
        if forbidden_mask.sum() < 2:
            self.forbidden_pairs_fraction = 0.0
            self.conf_loss_value_scalar = 0.0
            return loss_dict

        # Compute confusion loss
        forbidden_emb = embeddings[forbidden_mask]
        # Compute pairwise similarities among forbidden samples using the configured distance (e.g., cosine)
        sim_matrix = self.distance(forbidden_emb, forbidden_emb)

        # Create an upper-triangular mask (excluding the diagonal) to avoid duplicate and self comparisons.
        tri_mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        # Also consider only pairs with similarity above the forbidden_margin.
        sim_mask = sim_matrix > self.forbidden_margin
        final_mask = tri_mask & sim_mask

        total_pairs = tri_mask.sum().item()
        exceeded_pairs = final_mask.sum().item()
        self.forbidden_pairs_fraction = exceeded_pairs / total_pairs if total_pairs > 0 else 0.0

        confusion_loss_value = torch.zeros(1, device=embeddings.device, requires_grad=True)
        if exceeded_pairs > 0:
            diff = sim_matrix - self.forbidden_margin
            # Use a log-sum-exp formulation (with add_one=True for numerical stability)
            # Make sure this returns a scalar
            confusion_term = lmu.logsumexp(
                self.confusion_gamma * diff, keep_mask=final_mask, add_one=True
            )
            confusion_loss_value = (1.0 / self.confusion_gamma) * confusion_term

        # Scale the confusion term by the provided weight.
        confusion_loss_value = self.confusion_weight * confusion_loss_value
        # Store average confusion loss for logging
        self.conf_loss_value_scalar = confusion_loss_value.mean().item()

        # Now add the confusion penalty to the base MS loss
        if isinstance(original_losses, torch.Tensor):
            if len(original_losses.shape) > 0:  
                # If it's a per-element loss tensor, distribute confusion across the batch
                batch_size = original_losses.size(0)

                # Check if confusion_loss_value is a scalar tensor or a multi-element tensor
                if confusion_loss_value.numel() == 1:
                    # Add a fraction of the confusion loss to each element
                    # Using unsqueeze and expand to maintain gradient flow
                    expanded_conf = (confusion_loss_value / batch_size).expand_as(original_losses)
                    loss_dict["loss"]["losses"] = original_losses + expanded_conf
                else:
                    # For a multi-element tensor, use the mean and expand
                    mean_conf = confusion_loss_value.mean()
                    expanded_conf = (mean_conf / batch_size).expand_as(original_losses)
                    loss_dict["loss"]["losses"] = original_losses + expanded_conf
            else:
                # If it's a scalar tensor
                # Check if confusion_loss_value is a scalar tensor or a multi-element tensor
                if confusion_loss_value.numel() == 1:
                    loss_dict["loss"]["losses"] = original_losses + confusion_loss_value
                else:
                    # For a multi-element tensor, use the mean
                    loss_dict["loss"]["losses"] = original_losses + confusion_loss_value.mean()
        else:
            # If it's a Python scalar, convert to tensor while preserving gradients
            if confusion_loss_value.numel() == 1:
                loss_dict["loss"]["losses"] = confusion_loss_value + original_losses
            else:
                # For a multi-element tensor, use the mean
                loss_dict["loss"]["losses"] = confusion_loss_value.mean() + original_losses

        return loss_dict
