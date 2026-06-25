import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


def safe_project(x, eps=1e-5):
    norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)
    scale = torch.clamp(1 - eps, max=1.0) / norm
    scale = torch.minimum(scale, torch.ones_like(scale))
    return x * scale

# === CBA ===
class CBA(nn.Module):
    def __init__(self, dim, num_bands=5, dropout=0.2):
        super().__init__()
        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_bands)])
        self.k_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_bands)])
        self.v_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_bands)])
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xs):
        out = []
        for i in range(len(xs)):
            q = self.q_proj[i](xs[i])
            cross = 0
            for j in range(len(xs)):
                if i == j:
                    continue
                k = self.k_proj[j](xs[j])
                v = self.v_proj[j](xs[j])
                attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5))
                cross += torch.bmm(attn, v)
            cross = cross / (len(xs) - 1)
            out.append(self.dropout(cross + xs[i]))
        return out

class HGAA(nn.Module):
    def __init__(self, dim, c=1.0, num_heads=4):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.linear = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, band_feats):  # list of [B, Seq, D]
        out = []
        for x in band_feats:
            x_euc = self.linear(x)
            x_hyp = self.manifold.expmap0(x_euc)
            try:
                x_hyp = self.manifold.projx(x_hyp)
            except:
                x_hyp = safe_project(x_hyp)
            x_tan = self.manifold.logmap0(x_hyp)
            attn_out, _ = self.attn(x_hyp,x_hyp, x_hyp)
            residual = self.norm(x_hyp + attn_out)
            gate = self.gate(torch.cat([residual, x], dim=-1))
            out.append(gate * residual + (1 - gate) * x)
        return out

# === Separable Conv ===
class SeparableConvEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class RSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x

class RSA_Group(nn.Module):
    def __init__(self, dim, depth=2, num_heads=4):
        super().__init__()
        self.blocks = nn.Sequential(*[RSA(dim) for _ in range(depth)])

    def forward(self, x, x_size):
        return self.blocks(x)

# === Frequency GRSAFormer分支 ===
class FrequencyGRSAFormerBranch(nn.Module):
    def __init__(self, in_channels=62, num_bands=5, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.embed = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, num_bands, embed_dim))
        self.encoder = RSA_Group(embed_dim, depth, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Conv1d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1) + self.pos_embed
        x = self.encoder(x, x_size=(1, x.size(1)))
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.out_proj(x)
        return F.relu(x)

# === 可学习位置编码 ===
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# === Gated Residual Fusion ===
class GatedResidualFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.res_proj = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU())
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x1, x2):
        x_cat = torch.cat([x1, x2], dim=1)
        gate = self.gate(x_cat)
        fused = gate * x1 + (1 - gate) * x2
        residual = self.res_proj(torch.cat([fused, x1 + x2], dim=1))
        return self.norm(residual + fused)

# === 主模型 V4 ===
class BandWiseHyperbolicTransformer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, seq_len=62, num_classes=3,
                 dropout=0.3, num_bands=5, in_channels=62):
        super().__init__()
        self.num_bands = num_bands

        # === 加入输入批归一化层 ===
        self.input_bn = nn.BatchNorm2d(1)

        self.band_processor = FrequencyGRSAFormerBranch(in_channels, num_bands)
        self.pre_encoder = nn.ModuleList([
            SeparableConvEncoder(1, embed_dim) for _ in range(num_bands)
        ])
        self.embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout))
            for _ in range(num_bands)
        ])
        self.position = LearnablePositionalEncoding(embed_dim, max_len=seq_len)
        self.cbsa = CBA(embed_dim, num_bands, dropout)
        self.hyper_enhance = HGAA(embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_proj1 = nn.Sequential(nn.Linear(in_channels * num_bands, 200), nn.BatchNorm1d(200), nn.ReLU())
        self.feature_proj2 = nn.Sequential(nn.Linear(embed_dim, 200), nn.BatchNorm1d(200), nn.ReLU())
        self.fusion = GatedResidualFusion(200)
        self.classifier = nn.Sequential(
            nn.Linear(200, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.input_bn(x)
        x = x.squeeze(1)

        x1 = self.band_processor(x)

        band_features = []
        for i in range(self.num_bands):
            band_input = x[:, :, i].unsqueeze(1)
            band_input = self.pre_encoder[i](band_input)
            band_input = band_input.transpose(1, 2)
            emb = self.embedding[i](band_input)
            emb = self.position(emb)
            band_features.append(emb)

        band_features = self.cbsa(band_features)
        band_features = self.hyper_enhance(band_features)

        pooled_features = [self.pool(f.transpose(1, 2)).squeeze(-1) for f in band_features]
        x2 = torch.stack(pooled_features, dim=-1)  # [B, D, M]

        x_f1 = self.feature_proj1(self.flatten(x1))
        x_f2 = self.feature_proj2(x2.mean(dim=-1))
        fused = self.fusion(x_f1, x_f2)
        out = self.classifier(fused)
        return out, fused, x1, x2


class BandContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, num_bands=5, use_band_attention=True,
                 in_dim1=62, in_dim2=64, proj_dim=64):
        super().__init__()
        self.temperature = temperature
        self.num_bands = num_bands
        self.use_band_attention = use_band_attention
        self.proj1 = nn.Linear(in_dim1, proj_dim)
        self.proj2 = nn.Linear(in_dim2, proj_dim)

        if use_band_attention:
            self.band_weight = nn.Parameter(torch.ones(num_bands, num_bands) * 0.5)
            nn.init.eye_(self.band_weight.data)

    def forward(self, feat1, feat2, labels):
        device = feat1.device
        labels = labels.to(device)

        self.proj1 = self.proj1.to(device)
        self.proj2 = self.proj2.to(device)
        if self.use_band_attention:
            self.band_weight.data = self.band_weight.data.to(device)

        if feat1.dim() == 2:
            feat1 = feat1.unsqueeze(-1)
        if feat2.dim() == 2:
            feat2 = feat2.unsqueeze(-1)

        B, C1, M = feat1.shape
        _, C2, _ = feat2.shape
        loss = 0.0

        for i in range(M):
            for j in range(M):
                z1 = F.normalize(self.proj1(feat1[:, :, i]), dim=1)
                z2 = F.normalize(self.proj2(feat2[:, :, j]), dim=1)

                sim = torch.matmul(z1, z2.T)
                logits = sim / self.temperature

                label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
                pos_mask = label_mask.float()
                neg_mask = 1.0 - pos_mask

                exp_logits = torch.exp(logits)

                weight = torch.sigmoid(self.band_weight[i, j]) if self.use_band_attention else 1.0

                pos = exp_logits * pos_mask * weight
                neg = exp_logits * neg_mask * (2.0 - weight)

                pos_sum = pos.sum(dim=1) + 1e-6
                denom = pos_sum + neg.sum(dim=1) + 1e-6

                loss += -torch.log(pos_sum / denom).mean()

        return loss / (M * M)