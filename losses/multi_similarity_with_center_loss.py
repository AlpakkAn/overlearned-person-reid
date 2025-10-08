import torch
import torch.nn as nn
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class MultiSimilarityWithCenterLoss(MultiSimilarityLoss):
    """
    An extension of MultiSimilarityLoss that adds a center-based collapse penalty
    for forbidden classes, pushing all forbidden embeddings toward a single center.

    Args:
        alpha, beta, base (float): Same as in MultiSimilarityLoss
        forbidden_labels (list or set or None): Labels of forbidden classes.
            If None or empty, then no center penalty is applied.
        center_loss_weight (float): Weight for the center-based penalty term.
        embedding_size (int): The dimensionality of your embedding (needed
            to initialize the center).
        use_trainable_center (bool): If True, the center is a learnable parameter.
                                     If False, the center is a fixed buffer.
        center_lr (float): Learning rate for the center when manual center
                         updates are enabled (ignored if not use_manual_center)
        use_manual_center (bool): If True, manually update center in compute_loss
                                 with center_lr instead of using optimizer.
        init_center (torch.Tensor or None): Optional initial value for the center
            (shape = [embedding_size]). If None, defaults to zeros.
        kwargs: Other args for the parent MultiSimilarityLoss constructor.
    """

    def __init__(
        self,
        alpha=2,
        beta=50,
        base=0.5,
        forbidden_labels=None,
        center_loss_weight=0.01,
        embedding_size=128,
        use_trainable_center=False,
        center_lr=0.1,
        use_manual_center=False,
        init_center=None,
        **kwargs
    ):
        super().__init__(alpha=alpha, beta=beta, base=base, **kwargs)

        # Convert to set for quick "in" checks if user passed list/tuple.
        if forbidden_labels is None:
            self.forbidden_labels = set()
        else:
            self.forbidden_labels = set(forbidden_labels)

        self.center_loss_weight = center_loss_weight
        self.use_trainable_center = use_trainable_center
        self.center_lr = center_lr
        self.use_manual_center = use_manual_center
        
        if init_center is None:
            init_center = torch.zeros(embedding_size, dtype=torch.float)

        # Register the center as either a trainable parameter or a fixed buffer.
        if self.use_trainable_center:
            # Learnable parameter
            self.center = nn.Parameter(init_center)
        else:
            # Fixed buffer
            self.register_buffer("center", init_center)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        Overrides compute_loss to allow access to embeddings and labels, so we can
        add the center-based penalty for the forbidden classes. The multi-similarity
        loss is computed via the parent class, and then we add an extra term.
        """
        # Convert indices_tuple for pair-based losses:
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()

        # Compute the standard multi-similarity portion
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indices_tuple

        pos_mask = torch.zeros_like(mat)
        neg_mask = torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1

        # Identify forbidden label samples
        forbidden_mask_a = torch.zeros_like(labels, dtype=torch.bool)
        forbidden_mask_ref = torch.zeros_like(ref_labels, dtype=torch.bool)
        
        for lb in self.forbidden_labels:
            forbidden_mask_a |= (labels == lb)
            forbidden_mask_ref |= (ref_labels == lb)
            
        # Zero out positive pairs where anchors have forbidden labels
        # This ensures MS Loss only pushes forbidden samples away from other classes
        # but doesn't pull them closer to anything
        for i, anchor_idx in enumerate(a1):
            if forbidden_mask_a[anchor_idx]:
                pos_mask[anchor_idx, p[i]] = 0

        # Compute modified MS loss
        ms_loss_dict = super()._compute_loss(mat, pos_mask, neg_mask)
        ms_loss_val = ms_loss_dict["loss"]["losses"]

        # Add the center-based regularization if forbidden_labels is defined
        reg_loss = 0.0
        if self.forbidden_labels:
            # Identify which samples belong to any of the forbidden labels
            # We already have forbidden_mask_a from above
            if forbidden_mask_a.any():
                forbidden_emb = embeddings[forbidden_mask_a]
                # Move center to the same device as embeddings if needed
                center = self.center.to(embeddings.device)
                # L2 distance to the single shared center
                reg_loss = (forbidden_emb - center).pow(2).sum(dim=1).mean()
                
                # If using manual center updates, update the center directly
                if self.use_manual_center and self.training:
                    # Compute gradient for center update
                    with torch.enable_grad():
                        if not center.requires_grad:
                            center.requires_grad = True
                            center_grad = torch.autograd.grad(
                                reg_loss, center, retain_graph=True
                            )[0]
                            center.requires_grad = False
                        else:
                            center_grad = torch.autograd.grad(
                                reg_loss, center, retain_graph=True
                            )[0]
                        
                        # Update center with custom learning rate
                        if self.use_trainable_center:
                            with torch.no_grad():
                                # Make sure to use the same device as self.center
                                center_grad = center_grad.to(self.center.device)
                                self.center.data = self.center.data - self.center_lr * center_grad
                        else:
                            # Make sure to use the same device
                            center_grad = center_grad.to(self.center.device)
                            self.center.copy_(self.center - self.center_lr * center_grad)

        total_loss = ms_loss_val + self.center_loss_weight * reg_loss
        ms_loss_dict["loss"]["losses"] = total_loss
        return ms_loss_dict
