"""
model.py - DrugCLIP-inspired two-tower ranking model.

DrugCLIP treats virtual screening as dense retrieval: encode the target
pocket as a query, encode each molecule as a key, and rank candidates by
query-key similarity.  AffinityCLIP applies the same retrieval idea to
antibody affinity ranking:

  antigen context -> query Transformer tower
  antibody chains -> key Transformer tower
  cosine(query, key) * temperature -> ranking logit

The model deliberately returns a full similarity matrix.  Pair scores are the
diagonal when each row contains the matched antigen context and antibody.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import cfg


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - lightweight doc env
    torch = None
    nn = None
    F = None


if nn is not None:

    class TransformerTower(nn.Module):
        """
        Encode a list of modality vectors into one normalized retrieval vector.

        Parameters:
          feature_dims:     input dimension of each modality token.
          token_dim:        Transformer hidden dimension.
          projection_dim:   final shared embedding dimension.
          num_layers:       number of TransformerEncoder layers.
          num_heads:        number of attention heads.
          feedforward_dim:  feed-forward dimension inside Transformer layers.
          dropout:          dropout probability.

        Returns:
          forward(...) returns a tensor of shape [batch, projection_dim],
          L2-normalized along the last dimension.

        Implementation:
          Each input vector is linearly projected to token_dim.  A learned CLS
          token is prepended, modality embeddings are added, masked missing
          tokens are ignored by src_key_padding_mask, and the encoded CLS state
          is projected into the shared retrieval space.
        """

        def __init__(
            self,
            feature_dims: Sequence[int],
            token_dim: int = cfg.token_dim,
            projection_dim: int = cfg.projection_dim,
            num_layers: int = cfg.num_layers,
            num_heads: int = cfg.num_heads,
            feedforward_dim: int = cfg.feedforward_dim,
            dropout: float = cfg.dropout,
        ):
            super().__init__()
            if len(feature_dims) == 0:
                raise ValueError("feature_dims 不能为空")

            self.feature_dims = tuple(int(dim) for dim in feature_dims)
            self.token_dim = int(token_dim)
            self.projection_dim = int(projection_dim)

            self.projections = nn.ModuleList([
                nn.Linear(dim, self.token_dim)
                for dim in self.feature_dims
            ])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
            self.modality_embedding = nn.Parameter(
                torch.zeros(1, len(self.feature_dims) + 1, self.token_dim)
            )

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
                nn.Linear(self.token_dim, self.projection_dim),
            )

            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.modality_embedding, std=0.02)

        @staticmethod
        def _all_available(
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor:
            """Return a [batch] bool tensor marking every token available."""
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        def _validate_features(
            self,
            features: Sequence[torch.Tensor],
        ) -> tuple[int, torch.device]:
            """Validate feature count/rank/dim and return batch size/device."""
            if len(features) != len(self.feature_dims):
                raise ValueError(
                    f"收到 {len(features)} 个 feature，期望 {len(self.feature_dims)} 个"
                )

            batch_size = int(features[0].shape[0])
            device = features[0].device
            for idx, (feature, expected_dim) in enumerate(zip(features, self.feature_dims)):
                if feature.ndim != 2:
                    raise ValueError(f"feature[{idx}] 必须是二维：[batch, dim]")
                if int(feature.shape[1]) != expected_dim:
                    raise ValueError(
                        f"feature[{idx}] dim={feature.shape[1]}，期望 {expected_dim}"
                    )
                if int(feature.shape[0]) != batch_size:
                    raise ValueError("所有 feature 的 batch size 必须一致")
                if feature.device != device:
                    raise ValueError("所有 feature 必须在同一个 device 上")
            return batch_size, device

        def forward(
            self,
            features: Sequence[torch.Tensor],
            available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """
            Encode modality features.

            Parameters:
              features:  list of [batch, dim] tensors.
              available: optional list of [batch] bool tensors.  False means
                         this modality token is missing and should be masked.

            Returns:
              [batch, projection_dim] normalized retrieval vectors.
            """
            batch_size, device = self._validate_features(features)
            if available is None:
                available = [
                    self._all_available(batch_size, device)
                    for _ in self.feature_dims
                ]
            if len(available) != len(self.feature_dims):
                raise ValueError("available 数量必须和 features 数量一致")

            tokens = [self.cls_token.expand(batch_size, -1, -1)]
            padding_parts = [
                torch.zeros(batch_size, dtype=torch.bool, device=device)
            ]

            for feature, is_available, projection in zip(
                features,
                available,
                self.projections,
            ):
                if is_available.shape[0] != batch_size:
                    raise ValueError("available mask 的 batch size 不匹配")
                tokens.append(projection(feature).unsqueeze(1))
                padding_parts.append(~is_available.to(device=device).bool())

            token_tensor = torch.cat(tokens, dim=1)
            token_tensor = token_tensor + self.modality_embedding[:, :token_tensor.shape[1], :]
            key_padding_mask = torch.stack(padding_parts, dim=1)

            encoded = self.encoder(
                token_tensor,
                src_key_padding_mask=key_padding_mask,
            )
            cls_state = encoded[:, 0, :]
            return F.normalize(self.output(cls_state), p=2, dim=-1)


    class AffinityCLIP(nn.Module):
        """
        v4 antibody-antigen retrieval model.

        Parameters:
          antibody_feature_dims: modality dimensions for the antibody tower.
                                Default is [heavy, light].
          antigen_feature_dims:  modality dimensions for the antigen tower.
                                Default is [single antigen, MSA, flags].
          token_dim/projection_dim/num_layers/num_heads/feedforward_dim/dropout:
                                tower architecture parameters.
          logit_temperature:     initial softmax temperature; lower means
                                sharper similarity distribution.
          logit_scale_max:       max clamp for exp(logit_scale).

        Returns:
          forward(...) returns logits of shape [num_antigens, num_antibodies].
          Larger logits mean stronger predicted affinity.

        Implementation:
          Antigen vectors are queries and antibody vectors are keys.  Both are
          L2-normalized, so matrix multiplication is cosine similarity.  A
          learned temperature rescales the logits for contrastive training.
        """

        def __init__(
            self,
            antibody_feature_dims: Sequence[int] = cfg.antibody_feature_dims,
            antigen_feature_dims: Sequence[int] = cfg.antigen_feature_dims,
            token_dim: int = cfg.token_dim,
            projection_dim: int = cfg.projection_dim,
            num_layers: int = cfg.num_layers,
            num_heads: int = cfg.num_heads,
            feedforward_dim: int = cfg.feedforward_dim,
            dropout: float = cfg.dropout,
            logit_temperature: float = cfg.logit_temperature,
            logit_scale_max: float = cfg.logit_scale_max,
        ):
            super().__init__()
            if logit_temperature <= 0:
                raise ValueError("logit_temperature 必须大于 0")

            tower_kwargs = {
                "token_dim": token_dim,
                "projection_dim": projection_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "feedforward_dim": feedforward_dim,
                "dropout": dropout,
            }
            self.antibody_tower = TransformerTower(
                antibody_feature_dims,
                **tower_kwargs,
            )
            self.antigen_tower = TransformerTower(
                antigen_feature_dims,
                **tower_kwargs,
            )
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / logit_temperature), dtype=torch.float32)
            )
            self.logit_scale_max = float(logit_scale_max)

        @staticmethod
        def _as_feature_list(features: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
            """Normalize tensor-or-list input into a list of tensors."""
            if isinstance(features, torch.Tensor):
                return [features]
            return list(features)

        def encode_antibody(
            self,
            antibody_features: torch.Tensor | Sequence[torch.Tensor],
            antibody_available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """Return normalized antibody key vectors."""
            return self.antibody_tower(
                self._as_feature_list(antibody_features),
                available=antibody_available,
            )

        def encode_antigen(
            self,
            antigen_features: torch.Tensor | Sequence[torch.Tensor],
            antigen_available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """Return normalized antigen query vectors."""
            return self.antigen_tower(
                self._as_feature_list(antigen_features),
                available=antigen_available,
            )

        def similarity_matrix(
            self,
            antibody_features: torch.Tensor | Sequence[torch.Tensor],
            antigen_features: torch.Tensor | Sequence[torch.Tensor],
            antibody_available: Sequence[torch.Tensor] | None = None,
            antigen_available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """
            Compute antigen-query by antibody-key logits.

            Parameters:
              antibody_features:   antibody modality tensors.
              antigen_features:    antigen modality tensors.
              antibody_available:  optional availability masks for antibody tokens.
              antigen_available:   optional availability masks for antigen tokens.

            Returns:
              logits: [num_antigens, num_antibodies].
            """
            antibody_keys = self.encode_antibody(
                antibody_features,
                antibody_available=antibody_available,
            )
            antigen_queries = self.encode_antigen(
                antigen_features,
                antigen_available=antigen_available,
            )
            scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
            return scale * antigen_queries @ antibody_keys.t()

        def forward(
            self,
            antibody_features: torch.Tensor | Sequence[torch.Tensor],
            antigen_features: torch.Tensor | Sequence[torch.Tensor],
            antibody_available: Sequence[torch.Tensor] | None = None,
            antigen_available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """Alias for similarity_matrix(...)."""
            return self.similarity_matrix(
                antibody_features=antibody_features,
                antigen_features=antigen_features,
                antibody_available=antibody_available,
                antigen_available=antigen_available,
            )

        def score_pairs(
            self,
            antibody_features: torch.Tensor | Sequence[torch.Tensor],
            antigen_features: torch.Tensor | Sequence[torch.Tensor],
            antibody_available: Sequence[torch.Tensor] | None = None,
            antigen_available: Sequence[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """
            Return matched pair scores from the diagonal of the similarity matrix.

            This is the score vector used by RankNet when each batch row contains
            one observed antibody-antigen measurement.
            """
            logits = self.similarity_matrix(
                antibody_features=antibody_features,
                antigen_features=antigen_features,
                antibody_available=antibody_available,
                antigen_available=antigen_available,
            )
            if logits.shape[0] != logits.shape[1]:
                raise ValueError("score_pairs 要求 antibody 和 antigen batch size 相同")
            return logits.diagonal()

else:

    class TransformerTower:  # type: ignore[no-redef]
        """Placeholder used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("TransformerTower 需要安装 torch")


    class AffinityCLIP:  # type: ignore[no-redef]
        """Placeholder used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("AffinityCLIP 需要安装 torch")
