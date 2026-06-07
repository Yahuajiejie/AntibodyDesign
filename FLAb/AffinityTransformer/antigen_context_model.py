"""
antigen_context_model.py — v3 抗原上下文模型

这个模块定义 v3 模型结构，但不会替换 AffinityMLP(v1/v2)。

核心模型 AffinityTransformer 把不同来源的向量看成 modality tokens：

  [CLS], antibody, antigen_single, antigen_msa, flags

然后用 TransformerEncoder 做融合。缺失的 antigen_single / antigen_msa token
通过 key_padding_mask 屏蔽，而不是让模型把缺失零向量误认为真实生物信号。
"""

from __future__ import annotations


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - 轻量质检环境可能没有 torch
    torch = None
    nn = None


if nn is not None:

    class AntigenContextProjector(nn.Module):
        """
        把 antigen context 投影到统一维度。

        用途：
          v3.1 中 single_esm2 [1280]、msa_esm1b [768]、type/flags 维度不同，
          先投影成统一 antigen_context，再和 antibody feature 融合。
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int = 512,
            hidden_dim: int | None = None,
            dropout: float = 0.2,
        ):
            super().__init__()
            hidden_dim = hidden_dim or max(output_dim, min(1024, input_dim))
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
            )

        def forward(self, antigen_features: torch.Tensor) -> torch.Tensor:
            """返回投影后的 antigen context。"""
            return self.net(antigen_features)


    class AntigenContextMLP(nn.Module):
        """
        v3 抗体-抗原上下文打分模型。

        输入：
          antibody_features: [batch, antibody_dim]
          antigen_features:  [batch, antigen_dim]

        输出：
          score: [batch]，值越大表示预测亲和力越强。
        """

        def __init__(
            self,
            antibody_dim: int,
            antigen_dim: int,
            antigen_projection_dim: int = 512,
            hidden_dim: int = 256,
            dropout: float = 0.2,
            project_antigen: bool = True,
        ):
            super().__init__()
            self.antibody_dim = int(antibody_dim)
            self.antigen_dim = int(antigen_dim)
            self.project_antigen = bool(project_antigen)

            if self.project_antigen:
                self.antigen_projector = AntigenContextProjector(
                    input_dim=self.antigen_dim,
                    output_dim=antigen_projection_dim,
                    dropout=dropout,
                )
                fused_dim = self.antibody_dim + antigen_projection_dim
            else:
                self.antigen_projector = nn.Identity()
                fused_dim = self.antibody_dim + self.antigen_dim

            self.net = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            antibody_features: torch.Tensor,
            antigen_features: torch.Tensor,
        ) -> torch.Tensor:
            """计算 affinity score。"""
            antigen_context = self.antigen_projector(antigen_features)
            fused = torch.cat([antibody_features, antigen_context], dim=1)
            return self.net(fused).squeeze(-1)

        def score_from_concat(self, features: torch.Tensor) -> torch.Tensor:
            """
            从已经拼接好的 [antibody, antigen] 特征中切分并打分。

            这个方法方便和 numpy feature matrix / DataLoader 对接。
            """
            antibody = features[:, :self.antibody_dim]
            antigen = features[:, self.antibody_dim:]
            if antigen.shape[1] != self.antigen_dim:
                raise ValueError(
                    f"antigen feature dim={antigen.shape[1]}，"
                    f"期望 {self.antigen_dim}"
                )
            return self.forward(antibody, antigen)


    class AffinityTransformer(nn.Module):
        """
        v3 modality-token Transformer。

        输入：
          antibody_features:       [batch, antibody_dim]
          antigen_single_features: [batch, antigen_single_dim]
          antigen_msa_features:    [batch, antigen_msa_dim]
          flag_features:           [batch, flag_dim]

        输出：
          score: [batch]，值越大表示预测亲和力越强。

        设计要点：
          - 官方有抗原序列时，single 和 MSA 两个 token 都参与注意力；
          - 官方没有抗原序列时，single token 可传零向量，并用 single_available=False
            mask 掉；模型主要依赖 MSA token；
          - flags token 始终参与，让模型知道当前样本采用哪种上下文策略。
        """

        def __init__(
            self,
            antibody_dim: int,
            antigen_single_dim: int = 1280,
            antigen_msa_dim: int = 768,
            flag_dim: int = 0,
            token_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            feedforward_dim: int = 512,
            dropout: float = 0.2,
            use_flags: bool = True,
        ):
            super().__init__()
            self.antibody_dim = int(antibody_dim)
            self.antigen_single_dim = int(antigen_single_dim)
            self.antigen_msa_dim = int(antigen_msa_dim)
            self.flag_dim = int(flag_dim)
            self.token_dim = int(token_dim)
            self.use_flags = bool(use_flags and flag_dim > 0)

            self.antibody_proj = nn.Linear(self.antibody_dim, self.token_dim)
            self.single_proj = nn.Linear(self.antigen_single_dim, self.token_dim)
            self.msa_proj = nn.Linear(self.antigen_msa_dim, self.token_dim)
            self.flag_proj = (
                nn.Linear(self.flag_dim, self.token_dim)
                if self.use_flags else None
            )

            # token order: cls, antibody, single, msa, flags(optional)
            n_tokens = 4 + (1 if self.use_flags else 0)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
            self.modality_embedding = nn.Parameter(torch.zeros(1, n_tokens, self.token_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.output = nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, self.token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.token_dim, 1),
            )

            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.modality_embedding, std=0.02)

        @staticmethod
        def _infer_available(features: torch.Tensor) -> torch.Tensor:
            """
            根据特征是否全零推断 token 是否存在。

            主路径建议显式传 single_available/msa_available；这个函数只是兜底。
            """
            return features.abs().sum(dim=1) > 0

        def forward(
            self,
            antibody_features: torch.Tensor,
            antigen_single_features: torch.Tensor,
            antigen_msa_features: torch.Tensor,
            flag_features: torch.Tensor | None = None,
            single_available: torch.Tensor | None = None,
            msa_available: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """计算 affinity score。"""
            batch_size = antibody_features.shape[0]
            if single_available is None:
                single_available = self._infer_available(antigen_single_features)
            if msa_available is None:
                msa_available = self._infer_available(antigen_msa_features)

            tokens = [
                self.cls_token.expand(batch_size, -1, -1),
                self.antibody_proj(antibody_features).unsqueeze(1),
                self.single_proj(antigen_single_features).unsqueeze(1),
                self.msa_proj(antigen_msa_features).unsqueeze(1),
            ]
            key_padding_parts = [
                torch.zeros(batch_size, dtype=torch.bool, device=antibody_features.device),
                torch.zeros(batch_size, dtype=torch.bool, device=antibody_features.device),
                ~single_available.bool(),
                ~msa_available.bool(),
            ]

            if self.use_flags:
                if flag_features is None:
                    raise ValueError("use_flags=True 时必须传 flag_features")
                tokens.append(self.flag_proj(flag_features).unsqueeze(1))
                key_padding_parts.append(
                    torch.zeros(batch_size, dtype=torch.bool, device=antibody_features.device)
                )

            token_tensor = torch.cat(tokens, dim=1)
            token_tensor = token_tensor + self.modality_embedding[:, :token_tensor.shape[1], :]
            key_padding_mask = torch.stack(key_padding_parts, dim=1)

            encoded = self.encoder(
                token_tensor,
                src_key_padding_mask=key_padding_mask,
            )
            cls = encoded[:, 0, :]
            return self.output(cls).squeeze(-1)

        def score_from_concat(self, features: torch.Tensor) -> torch.Tensor:
            """
            从拼接好的 v3 feature matrix 中切分并打分。

            拼接顺序必须为：
              antibody, antigen_single, antigen_msa, flags(optional)
            """
            start = 0
            antibody = features[:, start:start + self.antibody_dim]
            start += self.antibody_dim
            single = features[:, start:start + self.antigen_single_dim]
            start += self.antigen_single_dim
            msa = features[:, start:start + self.antigen_msa_dim]
            start += self.antigen_msa_dim
            flags = None
            if self.use_flags:
                flags = features[:, start:start + self.flag_dim]
                if flags.shape[1] != self.flag_dim:
                    raise ValueError(
                        f"flag dim={flags.shape[1]}，期望 {self.flag_dim}"
                    )
            return self.forward(
                antibody_features=antibody,
                antigen_single_features=single,
                antigen_msa_features=msa,
                flag_features=flags,
            )

else:

    class AntigenContextProjector:  # type: ignore[no-redef]
        """PyTorch 未安装时的占位类。"""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("AntigenContextProjector 需要安装 torch")


    class AntigenContextMLP:  # type: ignore[no-redef]
        """PyTorch 未安装时的占位类。"""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("AntigenContextMLP 需要安装 torch")


    class AffinityTransformer:  # type: ignore[no-redef]
        """PyTorch 未安装时的占位类。"""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("AffinityTransformer 需要安装 torch")
