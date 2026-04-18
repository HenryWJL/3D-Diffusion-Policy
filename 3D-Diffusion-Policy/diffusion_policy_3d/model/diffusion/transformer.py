import torch
import torch.nn as nn
from copy import deepcopy
from diffusion_policy_3d.model.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from diffusion_policy_3d.model.common.utils import SinusoidalPosEmb, init_weights


class TransformerNoisePredictionNet(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        global_cond_dim: int,
        timestep_embed_dim: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.input_len = input_len

        # Input encoder and decoder
        hidden_dim = int(max(input_dim, embed_dim) * mlp_ratio)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # Model components
        self.pos_embed = nn.Parameter(
            torch.empty(1, input_len, embed_dim).normal_(std=0.02)
        )
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, sample, timestep, global_cond):
        # Encode input
        embed = self.input_encoder(sample)

        # Encode timestep
        if len(timestep.shape) == 0:
            timestep = timestep.expand(sample.shape[0]).to(
                dtype=torch.long, device=sample.device
            )
        temb = self.timestep_encoder(timestep)

        # Concatenate timestep and condition along the sequence dimension
        x = embed + self.pos_embed
        cond = torch.cat([global_cond, temb], dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Decode output
        out = self.output_decoder(x)
        return out


class TransformerNoisePredictionNetwIndex(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        global_cond_dim: int,
        timestep_embed_dim: int = 256,
        index_embed_dim: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.input_len = input_len

        # Input encoder and decoder
        hidden_dim = int(max(input_dim, embed_dim) * mlp_ratio)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # Index encoder
        self.index_encoder = nn.Sequential(
            SinusoidalPosEmb(index_embed_dim),
            nn.Linear(index_embed_dim, index_embed_dim * 4),
            nn.Mish(),
            nn.Linear(index_embed_dim * 4, index_embed_dim),
        )

        # Model components
        self.pos_embed = nn.Parameter(
            torch.empty(1, input_len, embed_dim).normal_(std=0.02)
        )
        cond_dim = global_cond_dim + timestep_embed_dim + index_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, sample, timestep, index, global_cond):
        # Encode input
        embed = self.input_encoder(sample)

        # Encode timestep
        if len(timestep.shape) == 0:
            timestep = timestep.expand(sample.shape[0]).to(
                dtype=torch.long, device=sample.device
            )
        temb = self.timestep_encoder(timestep)

        # Encode index
        indices = index
        if not torch.is_tensor(indices):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            indices = torch.tensor([indices], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(indices) and len(indices.shape) == 0:
            indices = indices[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        indices = indices.expand(sample.shape[0])
        index_embed = self.index_encoder(indices)

        # Concatenate timestep and condition along the sequence dimension
        x = embed + self.pos_embed
        cond = torch.cat([global_cond, temb, index_embed], dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Decode output
        out = self.output_decoder(x)
        return out


class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)


class DualNoisePredictionNet(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        horizon: int,
        action_dim: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True
    ):
        super().__init__()

        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        # Low-frequency encoder and decoder
        self.low_freq_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.low_freq_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )
        # High-frequency encoder and decoder
        self.high_freq_encoder = deepcopy(self.low_freq_encoder)
        self.high_freq_decoder = deepcopy(self.low_freq_decoder)

        # Timestep embedding
        self.timestep_embedding = DualTimestepEncoder(timestep_embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.empty(1, horizon, embed_dim).normal_(std=0.02)
        )

        # DiT blocks
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        self.horizon = horizon

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, global_cond, low_freq, low_freq_t, high_freq, high_freq_t):
        # Encode inputs
        low_freq_embed = self.low_freq_encoder(low_freq)
        high_freq_embed = self.high_freq_encoder(high_freq)

        # Expand and encode timesteps
        if len(low_freq_t.shape) == 0:
            low_freq_t = low_freq_t.expand(low_freq.shape[0]).to(
                dtype=torch.long, device=low_freq.device
            )
        if len(high_freq_t.shape) == 0:
            high_freq_t = high_freq_t.expand(high_freq.shape[0]).to(
                dtype=torch.long, device=high_freq.device
            )
        temb = self.timestep_embedding(low_freq_t, high_freq_t)

        # Forward through model
        x = torch.cat((low_freq_embed, high_freq_embed), dim=1)
        x = x + self.pos_embed
        cond = torch.cat((global_cond, temb), dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Extract low- and high-frequency predictions
        low_freq_pred = x[:, :self.horizon/2]
        high_freq_pred = x[:, self.horizon/2:]

        # Decode outputs
        low_freq_pred = self.low_freq_decoder(low_freq_pred)
        high_freq_pred = self.high_freq_decoder(high_freq_pred)
        return low_freq_pred, high_freq_pred