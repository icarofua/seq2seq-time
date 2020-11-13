import torch
from torch import nn
from torch.nn import functional as F
from routing_transformer import RoutingTransformerLM

class RoutingTransfomer(nn.Module):
    """
    A single RoutingTransformer
    """
    def __init__(self, 
        x_dim, 
        y_dim,
        num_tokens, dim, depth, max_seq_len, 
        **kwargs):
        super().__init__()
        self._min_std = min_std
        self.nan_value = nan_value
        enc_x_dim = x_dim + y_dim

        self.enc_emb = nn.Linear(enc_x_dim, hidden_size)
        encoder_norm = nn.LayerNorm(hidden_size)

        self.encoder = RoutingTransformerLM(
                        num_tokens=num_tokens, 
                        max_seq_len=max_seq_len, 
                        dim=dim, 
                        depth=depth, 
                        **kwargs)

        self.mean = nn.Linear(hidden_size, y_dim)
        self.std = nn.Linear(hidden_size, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        B, S, _ = future_x.shape
        future_y_fake = past_y[:, -1:, :].repeat(1, S, 1).to(device)
        context = torch.cat([past_x, past_y], -1).detach()
        target = torch.cat([future_x, future_y_fake], -1).detach()
        x = torch.cat([context, target * 1], 1).detach()

        x = self.enc_emb(x).permute(1, 0, 2)
        outputs = self.encoder(x).permute(1, 0, 2)

        # Seems to help a little, especially with extrapolating out of bounds
        steps = past_y.shape[1]
        mean = self.mean(outputs)[:, steps:, :]
        log_sigma = self.std(outputs)[:, steps:, :]
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}