import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): Margin for triplet loss. Default is 1.0.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin=None):
        """
        Args:
            anchor (Tensor): Anchor embeddings of shape (batch_size, embedding_dim)
            positive (Tensor): Positive embeddings of shape (batch_size, embedding_dim)
            negative (Tensor): Negative embeddings of shape (batch_size, embedding_dim)
            margin : The renew margin
        Returns:
            Tensor: Computed triplet loss
        """
        # Use provided margin or fallback to the instance margin
        margin = margin if margin is not None else self.margin
        
        # Reshape input tensors if necessary
        if anchor.ndim > 2:
            anchor = anchor.view(-1, anchor.size(-1))  # Flatten to (batch_size, embedding_dim)
            positive = positive.view(-1, positive.size(-1))
            negative = negative.view(-1, negative.size(-1))

        # Compute pairwise distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # L2 norm
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Compute triplet loss
        loss = pos_dist - neg_dist + margin
        loss = F.relu(loss)
        return loss.mean()
