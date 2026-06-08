"""
losses.py - losses for AffinityClip(v4).

The raw DrugCLIP objective uses one positive diagonal pair per batch row and
treats other in-batch pairs as negatives.  FLAb affinity ranking is different:
one antigen group can contain many antibodies, and labels are continuous
ranking labels rather than binary bind/non-bind labels.

Therefore v4 uses:
  1. group-aware soft CLIP loss:
     for each antigen query, antibodies from the same compatible_group form a
     soft target distribution weighted by affinity labels.
  2. RankNet loss:
     the diagonal matched-pair scores are compared only within each group.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .config import cfg


try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - lightweight doc env
    torch = None
    F = None


if torch is not None:

    def _group_list(group_ids: Sequence[object] | torch.Tensor) -> list[str]:
        """Convert Python or tensor group ids to a list of strings."""
        if isinstance(group_ids, torch.Tensor):
            values = group_ids.detach().cpu().tolist()
            return [str(value) for value in values]
        return [str(value) for value in group_ids]


    def _indices_by_group(group_ids: Sequence[str]) -> dict[str, list[int]]:
        """Return group -> row indices mapping."""
        mapping: dict[str, list[int]] = defaultdict(list)
        for idx, group in enumerate(group_ids):
            mapping[group].append(idx)
        return dict(mapping)


    def ranknet_loss_from_scores(
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_ids: Sequence[object] | torch.Tensor,
        min_label_diff: float = cfg.min_label_diff,
    ) -> torch.Tensor:
        """
        Compute within-group RankNet loss from matched pair scores.

        Parameters:
          scores:         [batch] matched antibody-antigen scores.  Larger means
                          stronger predicted affinity.
          labels:         [batch] unified ranking labels.  Larger means stronger
                          measured affinity.
          group_ids:      compatible_group for each row.
          min_label_diff: ignore label pairs whose absolute gap is too small.

        Returns:
          Scalar tensor.  If no valid pair exists, returns a differentiable zero.

        Implementation:
          For each group, all pairs with label_i > label_j are constructed.
          RankNet minimizes softplus(score_j - score_i), i.e. it asks the
          higher-affinity sample to receive the higher score.
        """
        scores = scores.reshape(-1)
        labels = labels.to(device=scores.device, dtype=scores.dtype).reshape(-1)
        groups = _group_list(group_ids)
        if len(groups) != int(scores.shape[0]):
            raise ValueError("group_ids 长度必须等于 scores 长度")

        losses: list[torch.Tensor] = []
        for indices in _indices_by_group(groups).values():
            if len(indices) < 2:
                continue
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores.index_select(0, idx)
            group_labels = labels.index_select(0, idx)

            label_diff = group_labels[:, None] - group_labels[None, :]
            valid = label_diff > float(min_label_diff)
            if not torch.any(valid):
                continue
            score_diff = group_scores[:, None] - group_scores[None, :]
            losses.append(F.softplus(-score_diff[valid]).mean())

        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()


    def group_soft_clip_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        group_ids: Sequence[object] | torch.Tensor,
        label_temperature: float = cfg.label_temperature,
        average_by_group: bool = True,
    ) -> torch.Tensor:
        """
        Compute group-aware soft CLIP loss.

        Parameters:
          logits:            [batch, batch] antigen_query x antibody_key logits.
          labels:            [batch] unified ranking labels.
          group_ids:         compatible_group for each row/column.
          label_temperature: softmax temperature used to convert labels into
                             target probabilities inside one group.
          average_by_group:  if True, each compatible_group has equal weight
                             regardless of its row count.

        Returns:
          Scalar tensor.

        Implementation:
          For row i, only antibodies j with group_j == group_i receive non-zero
          target probability.  Their target weights are softmax(label_j / tau).
          All other antibodies remain negatives through the log_softmax
          denominator.  This adapts DrugCLIP's one-hot diagonal target to the
          many-antibodies-per-antigen FLAb setting.
        """
        if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
            raise ValueError("logits 必须是 [batch, batch] 方阵")
        if label_temperature <= 0:
            raise ValueError("label_temperature 必须大于 0")

        batch_size = int(logits.shape[0])
        labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        groups = _group_list(group_ids)
        if len(groups) != batch_size or int(labels.shape[0]) != batch_size:
            raise ValueError("labels/group_ids 长度必须等于 logits batch size")

        log_probs = F.log_softmax(logits, dim=1)
        group_to_losses: dict[str, list[torch.Tensor]] = defaultdict(list)
        group_to_indices = _indices_by_group(groups)

        for row_idx, group in enumerate(groups):
            candidate_indices = group_to_indices[group]
            idx = torch.tensor(candidate_indices, device=logits.device, dtype=torch.long)
            target = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
            target_weights = F.softmax(labels.index_select(0, idx) / label_temperature, dim=0)
            target.index_copy_(0, idx, target_weights)
            row_loss = -(target * log_probs[row_idx]).sum()
            group_to_losses[group].append(row_loss)

        if average_by_group:
            group_losses = [
                torch.stack(losses).mean()
                for losses in group_to_losses.values()
            ]
            return torch.stack(group_losses).mean()
        all_losses = [
            loss
            for losses in group_to_losses.values()
            for loss in losses
        ]
        return torch.stack(all_losses).mean()


    def affinity_clip_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        group_ids: Sequence[object] | torch.Tensor,
        clip_weight: float = cfg.clip_weight,
        ranknet_weight: float = cfg.ranknet_weight,
        label_temperature: float = cfg.label_temperature,
        min_label_diff: float = cfg.min_label_diff,
        average_clip_by_group: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Combine group-aware soft CLIP loss and RankNet loss.

        Parameters:
          logits:                [batch, batch] output of AffinityCLIP.
          labels:                [batch] unified ranking labels.
          group_ids:             compatible_group for each row.
          clip_weight:           coefficient for group_soft_clip_loss.
          ranknet_weight:        coefficient for ranknet_loss_from_scores.
          label_temperature:     target-distribution softness for CLIP loss.
          min_label_diff:        gap threshold for RankNet pair construction.
          average_clip_by_group: give each group equal weight in CLIP loss.

        Returns:
          dict containing:
            loss:    weighted sum used for backpropagation.
            clip:    unweighted group-aware soft CLIP loss.
            ranknet: unweighted RankNet loss.

        Implementation:
          The diagonal of logits is the matched observed pair score.  RankNet
          only compares those diagonal scores inside each compatible_group.
        """
        clip = group_soft_clip_loss(
            logits=logits,
            labels=labels,
            group_ids=group_ids,
            label_temperature=label_temperature,
            average_by_group=average_clip_by_group,
        )
        ranknet = ranknet_loss_from_scores(
            scores=logits.diagonal(),
            labels=labels,
            group_ids=group_ids,
            min_label_diff=min_label_diff,
        )
        loss = float(clip_weight) * clip + float(ranknet_weight) * ranknet
        return {
            "loss": loss,
            "clip": clip,
            "ranknet": ranknet,
        }

else:

    def ranknet_loss_from_scores(*args, **kwargs):  # type: ignore[no-redef]
        """Placeholder used when PyTorch is not installed."""
        raise ModuleNotFoundError("ranknet_loss_from_scores 需要安装 torch")


    def group_soft_clip_loss(*args, **kwargs):  # type: ignore[no-redef]
        """Placeholder used when PyTorch is not installed."""
        raise ModuleNotFoundError("group_soft_clip_loss 需要安装 torch")


    def affinity_clip_loss(*args, **kwargs):  # type: ignore[no-redef]
        """Placeholder used when PyTorch is not installed."""
        raise ModuleNotFoundError("affinity_clip_loss 需要安装 torch")
