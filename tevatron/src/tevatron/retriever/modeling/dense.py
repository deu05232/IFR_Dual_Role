import torch
import logging

from typing import Dict, Optional
from transformers import PreTrainedModel

from .encoder import EncoderModel, EncoderOutput
import torch.distributed as dist

from torch import nn, Tensor
from transformers import AutoTokenizer
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


class DenseJointLHModel(DenseModel):
    """JointLH loss: average NLL over all K positive passages per query.

    Group layout in passage tensor (per query):
        [pos_0, pos_1, ..., pos_{K-1}, neg_0, ..., neg_{G-K-1}]

    Supports mixed batches where some queries have K positives (q-type)
    and others have 1 positive (q_inst-type). The mask-based formulation
    handles both uniformly: when |D+|=1, JointLH reduces to standard InfoNCE.
    """

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        instruct_flag: Optional[Tensor] = None,
        num_positives: int = 1,
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if instruct_flag is not None:
                    instruct_flag = self._dist_gather_tensor(instruct_flag)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            loss = self._joint_lh_loss(
                scores / self.temperature,
                num_positives=num_positives,
                instruct_flag=instruct_flag,
            )

            if self.is_ddp:
                loss = loss * self.world_size
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def _joint_lh_loss(
        self,
        scores: Tensor,
        num_positives: int,
        instruct_flag: Optional[Tensor] = None,
    ) -> Tensor:
        """
        scores: [B, B*G]
        instruct_flag: [B] or None
            - None or 0 (q-type):      num_positives positives
            - 1 (q_inst-type):         1 positive
        """
        B = scores.size(0)
        G = scores.size(1) // B
        device = scores.device

        if instruct_flag is None:
            num_pos = torch.full((B,), num_positives, device=device, dtype=torch.long)
        else:
            num_pos = torch.where(instruct_flag == 0, num_positives, 1)
            # num_pos = torch.where(instruct_flag == 0,
            #                       torch.tensor(num_positives, device=device),
            #                       torch.tensor(1, device=device))  # [B]

        # pos_mask[i, j] = 1 iff passage j is a positive for query i
        group_pos = torch.arange(G, device=device).unsqueeze(0)       # [1, G]
        in_group_pos = group_pos < num_pos.unsqueeze(1)                # [B, G]

        pos_mask = torch.zeros(B, B * G, device=device)
        group_start = (torch.arange(B, device=device) * G).unsqueeze(1)  # [B, 1]
        col_idx = group_start + group_pos                               # [B, G]
        pos_mask.scatter_(1, col_idx, in_group_pos.float())

        log_probs = F.log_softmax(scores, dim=-1)                      # [B, B*G]
        pos_log_probs = (log_probs * pos_mask).sum(dim=-1)             # [B]
        loss_per_sample = -pos_log_probs / num_pos.float()

        joint_loss = loss_per_sample.mean()

        if self.process_rank == 0:
            with torch.no_grad():
                print("scores[0, :16]:", scores[0, :16].detach().float().cpu())
                print("num_pos[0]:", num_pos[0].detach().cpu().item())
                print("joint_loss:", joint_loss.detach().cpu().item())

        return joint_loss




class DenseLSEPairModel(DenseModel):
    """LSEPair loss: log-sum-exp over all (positive, negative) score differences.

    L_LSEPair = log(1 + sum_{d+ in D+} sum_{d- in D-} exp(s(q,d-) - s(q,d+)))

    Group layout in passage tensor (per query):
        [pos_0, pos_1, ..., pos_{K-1}, neg_0, ..., neg_{G-K-1}]

    Supports mixed batches where some queries have K positives (q-type)
    and others have 1 positive (q_inst-type). When |D+|=1, LSEPair reduces
    to standard InfoNCE (Wang et al., Section 3.3).
    """

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        instruct_flag: Optional[Tensor] = None,
        num_positives: int = 1,
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if instruct_flag is not None:
                    instruct_flag = self._dist_gather_tensor(instruct_flag)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            loss = self._lsepair_loss(
                scores / self.temperature,
                num_positives=num_positives,
                instruct_flag=instruct_flag,
            )

            if self.is_ddp:
                loss = loss * self.world_size
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def _lsepair_loss(
        self,
        scores: Tensor,
        num_positives: int,
        instruct_flag: Optional[Tensor] = None,
    ) -> Tensor:
        """
        scores: [B, B*G]
        instruct_flag: [B] or None
            - None or 0 (q-type):      num_positives positives
            - 1 (q_inst-type):         1 positive
        """
        B = scores.size(0)
        N = scores.size(1)
        G = N // B
        device = scores.device

        # Per-query positive count
        if instruct_flag is None:
            num_pos = torch.full((B,), num_positives, device=device, dtype=torch.long)
        else:
            num_pos = torch.where(instruct_flag == 0, num_positives, 1)

        # Build positive and negative masks over [B, B*G]
        # Positive: query i's first num_pos[i] slots in its own group
        # Negative: everything else in the full candidate space
        group_pos = torch.arange(G, device=device).unsqueeze(0)          # [1, G]
        in_group_pos = group_pos < num_pos.unsqueeze(1)                   # [B, G]

        pos_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        group_start = (torch.arange(B, device=device) * G).unsqueeze(1)   # [B, 1]
        col_idx = group_start + group_pos                                  # [B, G]
        pos_mask.scatter_(1, col_idx, in_group_pos)

        # Negative mask: not positive
        neg_mask = ~pos_mask                                               # [B, N]
        
        # neg에 대한 logsumexp
        scores_neg = scores.masked_fill(~neg_mask, float('-inf'))         # [B, N]
        logsumexp_neg = torch.logsumexp(scores_neg, dim=-1)               # [B]
        
        # pos에 대한 logsumexp of -s_i
        scores_pos_neg = (-scores).masked_fill(~pos_mask, float('-inf'))  # [B, N]
        logsumexp_pos = torch.logsumexp(scores_pos_neg, dim=-1)           # [B]
        
        # log(sum_{i,j} exp(s_j - s_i)) = logsumexp_neg + logsumexp_pos
        log_pair_sum = logsumexp_neg + logsumexp_pos                       # [B]

        # Final: log(1 + sum_{i,j} exp(s_j - s_i)) = logsumexp([0, log_pair_sum])
        zeros = torch.zeros_like(log_pair_sum)
        loss_per_sample = torch.logsumexp(
            torch.stack([zeros, log_pair_sum], dim=-1), dim=-1
        )                                                                  # [B]

        return loss_per_sample.mean()


class DenseSubsetJointLHModel(DenseModel):
    """JointLH loss with *dynamic* per-query positive count.

    Difference from DenseJointLHModel:
      - num_positives is a Tensor [B] passed in from the collator (not a scalar).
      - No instruct_flag branching; the per-query count fully determines the mask.

    group layout per query: [pos_0, ..., pos_{K_i-1}, neg_0, ..., neg_{G-K_i-1}]
    """

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        num_positives: Optional[Tensor] = None,
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if num_positives is not None:
                    num_positives = self._dist_gather_tensor(num_positives)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            loss = self._joint_lh_loss(scores / self.temperature, num_positives=num_positives)

            if self.is_ddp:
                loss = loss * self.world_size
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def _joint_lh_loss(self, scores: Tensor, num_positives: Tensor) -> Tensor:
        """
        scores:        [B, B*G]
        num_positives: [B] long tensor, per-query positive count K_i (1..G).
        """
        B = scores.size(0)
        G = scores.size(1) // B
        device = scores.device

        num_pos = num_positives.to(device=device, dtype=torch.long)
        assert num_pos.shape == (B,), f"num_positives shape {num_pos.shape} != ({B},)"
        assert (num_pos >= 1).all() and (num_pos <= G).all(), \
            f"num_positives out of range [1, {G}]: {num_pos.tolist()}"

        # pos_mask[i, j] = 1 iff passage j is a positive for query i
        group_pos = torch.arange(G, device=device).unsqueeze(0)            # [1, G]
        in_group_pos = group_pos < num_pos.unsqueeze(1)                     # [B, G]

        pos_mask = torch.zeros(B, B * G, device=device)
        group_start = (torch.arange(B, device=device) * G).unsqueeze(1)     # [B, 1]
        col_idx = group_start + group_pos                                    # [B, G]
        pos_mask.scatter_(1, col_idx, in_group_pos.float())

        log_probs = F.log_softmax(scores, dim=-1)                            # [B, B*G]
        pos_log_probs = (log_probs * pos_mask).sum(dim=-1)                   # [B]
        loss_per_sample = -pos_log_probs / num_pos.float()

        if self.is_ddp and self.process_rank == 0:
            with torch.no_grad():
                print("scores[0, :16]:", scores[0, :16].detach().float().cpu())
                print("num_pos:", num_pos.detach().cpu().tolist())

        return loss_per_sample.mean()


class DenseSubsetLSEPairModel(DenseModel):
    """LSEPair loss with *dynamic* per-query positive count.

    Difference from DenseLSEPairModel:
      - num_positives is a Tensor [B] passed in from the collator (not a scalar).
      - No instruct_flag branching; the per-query count fully determines the mask.
    """

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        num_positives: Optional[Tensor] = None,
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if num_positives is not None:
                    num_positives = self._dist_gather_tensor(num_positives)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            loss = self._lsepair_loss(scores / self.temperature, num_positives=num_positives)

            if self.is_ddp:
                loss = loss * self.world_size
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def _lsepair_loss(self, scores: Tensor, num_positives: Tensor) -> Tensor:
        """
        scores:        [B, B*G]
        num_positives: [B] long tensor, per-query positive count K_i (1..G).
        """
        B = scores.size(0)
        N = scores.size(1)
        G = N // B
        device = scores.device

        num_pos = num_positives.to(device=device, dtype=torch.long)
        assert num_pos.shape == (B,), f"num_positives shape {num_pos.shape} != ({B},)"
        assert (num_pos >= 1).all() and (num_pos <= G).all(), \
            f"num_positives out of range [1, {G}]: {num_pos.tolist()}"

        group_pos = torch.arange(G, device=device).unsqueeze(0)              # [1, G]
        in_group_pos = group_pos < num_pos.unsqueeze(1)                       # [B, G]

        pos_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        group_start = (torch.arange(B, device=device) * G).unsqueeze(1)       # [B, 1]
        col_idx = group_start + group_pos                                      # [B, G]
        pos_mask.scatter_(1, col_idx, in_group_pos)

        neg_mask = ~pos_mask                                                   # [B, N]

        scores_neg = scores.masked_fill(~neg_mask, float('-inf'))             # [B, N]
        logsumexp_neg = torch.logsumexp(scores_neg, dim=-1)                   # [B]

        scores_pos_neg = (-scores).masked_fill(~pos_mask, float('-inf'))      # [B, N]
        logsumexp_pos = torch.logsumexp(scores_pos_neg, dim=-1)               # [B]

        log_pair_sum = logsumexp_neg + logsumexp_pos                           # [B]

        zeros = torch.zeros_like(log_pair_sum)
        loss_per_sample = torch.logsumexp(
            torch.stack([zeros, log_pair_sum], dim=-1), dim=-1
        )                                                                      # [B]

        return loss_per_sample.mean()
