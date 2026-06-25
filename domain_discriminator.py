import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class UnifiedDomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.domain_head = nn.Linear(hidden_dim, 2)

    def forward(self, feature):
        x = self.grl(feature)
        x = self.shared(x)
        return self.domain_head(x)




