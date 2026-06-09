import collections
import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import math

import re
from tevatron.retriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)
print_once = False


def format_query(query: str, prefix: str = '', prompt: str = '') -> str:
    global print_once
    if prompt.strip() != '':
        query_ends_in_punct = query.strip()[-1] in ['.', '?', '!']
        added_q = "" if query_ends_in_punct else "?"
        ret_str = f'{prefix.strip()} {query.strip()}{added_q} {prompt.strip()}'.strip()
    else:
        query_ends_in_punct = query.strip()[-1] in ['.', '?', '!']
        added_q = "" if query_ends_in_punct else "?"
        ret_str = f'{prefix.strip()} {query.strip()}{added_q}'.strip()

    if not print_once:
        logger.info(f'Prompt: `{prompt}`')
        logger.info(f'Query: {ret_str}')
        print_once = True

    return ret_str

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    if title == "-":
        title = ""
    return f'{prefix.strip()} {title.strip()} {text.strip()}'.strip()



class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_n = min(self.data_args.negatives_first_n, len(group["new_negatives"]), negative_size)
            first_negs = group["new_negatives"][:first_n]
            remaining_to_select = negative_size - first_n
            # logger.info(f"first_n: {first_n}, remaining_to_select: {remaining_to_select}")
            
            if remaining_to_select > 0:
                _offset = epoch * remaining_to_select % len(group_negatives)
                shuffled_negs = [x for x in group_negatives]
                random.Random(_hashed_seed).shuffle(shuffled_negs)
                shuffled_negs = shuffled_negs * 2
                selected_negs = shuffled_negs[_offset: _offset + remaining_to_select]
            else:
                selected_negs = []

            negs = first_negs + selected_negs
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages
    
    
class FixedOriginalTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * (math.ceil(remaining_to_select / len(neg_passages)))
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * (math.ceil(remaining_to_select / len(neg_passages)))
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n == 0:
            negs = group_negatives[:negative_size]
            assert len(negs) == negative_size
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages


class JointLHTrainDataset(Dataset):
    """Dataset for JointLH training.

    Returns num_positives positives first, then negatives.
    group layout per query: [pos_0, ..., pos_{K-1}, neg_0, ..., neg_{G-K-1}]

    __getitem__ returns (query, passages, instruct_flag)
      instruct_flag=0 -> q-type  (JointLH, K positives)
      instruct_flag=1 -> q_inst-type (InfoNCE, 1 positive)
    """

    def __init__(self, data_args: DataArguments, trainer=None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        # q-type uses K positives; q_inst-type uses 1 positive (standard InfoNCE)
        self.num_positives = data_args.num_positives

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str], int]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']
        instruct_flag = 1 if group['has_instruction'] else 0 

        formatted_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formatted_passages = []

        # positive shuffle 적용 x
        if not group['has_instruction']:
            assert len(group_positives) >= self.num_positives, f"Not enough positives for query {item}"
            for one in group_positives[:self.num_positives]:
                formatted_passages.append(format_passage(one['text'], one.get('title', ''), self.data_args.passage_prefix))
        else:
            pos_psg = group_positives[0]
            formatted_passages.append(format_passage(pos_psg['text'], pos_psg.get('title', ''), self.data_args.passage_prefix))

        # Select negatives — same logic as FixedOriginalTrainDataset
        negative_size = self.data_args.train_group_size - len(formatted_passages)

        if negative_size == 0:
            negs = []
        elif len(group_negatives) < negative_size:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining = negative_size - len(first_negs)
            if remaining > 0:
                neg_pool = group_negatives * math.ceil(remaining / len(group_negatives))
                negs = first_negs + neg_pool[:remaining]
            else:
                negs = first_negs
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining = negative_size - len(first_negs)
            if remaining > 0:
                neg_pool = group_negatives * math.ceil(remaining / len(group_negatives))
                negs = first_negs + neg_pool[:remaining]
            else:
                negs = first_negs[:negative_size]
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n == 0:
            negs = group_negatives[:negative_size]
            assert len(negs) == negative_size
        else:
            assert False

        for neg_psg in negs:
            formatted_passages.append(
                format_passage(neg_psg['text'], neg_psg.get('title', ''), self.data_args.passage_prefix)
            )

        return formatted_query, formatted_passages, instruct_flag


class JointLHMarginTrainDataset(JointLHTrainDataset):
    """JointLH training with an auxiliary instruction-margin loss.

    Same selection logic as JointLHTrainDataset. Additionally returns
        - margin_flag  : from group['is_margin'] (1 iff the four-element
                         tuple q_inst / q / d+inst / d-inst is available).
        - formatted_only_query : instruction-stripped query from
                         group['only_query'], returned for every sample so the
                         collator emits a fixed-shape [B, ...] tensor. This is
                         required for DDP all-gather of only_q_reps; the model
                         selects margin-flagged samples inside the margin-loss
                         computation. For non-instruction samples we fall back
                         to the regular query.

    Returns (formatted_query, formatted_passages, instruct_flag, margin_flag,
             formatted_only_query).
    """

    def __getitem__(self, item):
        formatted_query, formatted_passages, instruct_flag = super().__getitem__(item)
        group = self.train_data[item]

        margin_flag = 1 if group['is_margin'] else 0
        only_q_text = group.get('only_query', None) or group['query']
        formatted_only_query = format_query(
            only_q_text, self.data_args.query_prefix, self.data_args.prompt
        )

        return formatted_query, formatted_passages, instruct_flag, margin_flag, formatted_only_query


class SubsetJointLHTrainDataset(Dataset):
    """Dataset for subset JointLH / LSEPair training with *dynamic* positive count.

    Difference from JointLHTrainDataset:
      - num_positives is decided per-query: min(len(positive_passages), max_positives).
      - No has_instruction branching; the per-query positive count fully describes the group.

    group layout per query: [pos_0, ..., pos_{K_i-1}, neg_0, ..., neg_{G-K_i-1}]
    where K_i = num_positives for query i (1..max_positives).

    __getitem__ returns (query, passages, num_positives_i)
    """

    def __init__(self, data_args: DataArguments, trainer=None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        self.max_positives = data_args.num_positives

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str], int]:
        group = self.train_data[item]

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        assert len(group_positives) >= 1, f"No positives for query {item}"
        num_positives = min(len(group_positives), self.max_positives)

        formatted_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formatted_passages = []

        for one in group_positives[:num_positives]:
            formatted_passages.append(format_passage(one['text'], one['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - len(formatted_passages)

        if negative_size == 0:
            negs = []
        elif len(group_negatives) < negative_size:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining = negative_size - len(first_negs)
            if remaining > 0:
                neg_pool = group_negatives * math.ceil(remaining / len(group_negatives))
                negs = first_negs + neg_pool[:remaining]
            else:
                negs = first_negs
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining = negative_size - len(first_negs)
            if remaining > 0:
                neg_pool = group_negatives * math.ceil(remaining / len(group_negatives))
                negs = first_negs + neg_pool[:remaining]
            else:
                negs = first_negs[:negative_size]
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n == 0:
            negs = group_negatives[:negative_size]
            assert len(negs) == negative_size
        else:
            assert False

        for neg_psg in negs:
            formatted_passages.append(
                format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix)
            )

        assert len(formatted_passages) == self.data_args.train_group_size, \
            f"Group size mismatch: {len(formatted_passages)} vs {self.data_args.train_group_size}"

        return formatted_query, formatted_passages, num_positives


def _selected_neg_docids(group, data_args, negative_size: int) -> set:
    """Docids of the negatives that ``__getitem__`` will actually emit for a
    group, given ``negative_size`` (= train_group_size - #positives used).

    Mirrors the negative-selection branches of FixedOriginalTrainDataset /
    JointLHTrainDataset.__getitem__. Negative selection there is deterministic
    (no epoch/seed dependence), so this can be precomputed once. Keep in sync
    with those __getitem__ methods.
    """
    if negative_size <= 0:
        return set()
    group_negatives = group['negative_passages']
    nfn = data_args.negatives_first_n
    if len(group_negatives) < negative_size or (nfn and nfn > 0):
        first_negs = group['new_negatives'][:nfn]
        remaining = negative_size - len(first_negs)
        if remaining > 0:
            pool = group_negatives * math.ceil(remaining / len(group_negatives))
            negs = first_negs + pool[:remaining]
        else:
            negs = first_negs[:negative_size]
    else:  # negatives_first_n == 0
        negs = group_negatives[:negative_size]
    return {d['docid'] for d in negs}


def _build_distinct_batch_order(
    base_qids: List[str],
    effective_batch_size: int,
    seed: int,
    pos_docids: List[set] = None,
    neg_docids: List[set] = None,
) -> List[int]:
    """Return an index permutation such that any window of
    `effective_batch_size` consecutive items has all-distinct base qids
    (relaxed only when the remaining bucket count makes the constraint
    impossible). Greedy round-robin with a sliding window of size
    (effective_batch_size - 1) of recently emitted base qids; ties on
    bucket length are broken randomly so base-qid order does not lock
    into a fixed cycle within an epoch.

    When both `pos_docids` and `neg_docids` are given (one set per sample,
    indexed by original sample index), an *additional* constraint is enforced
    inside each window: a sample's positive document(s) must not be used as a
    negative by any other sample in the window, and vice versa. This avoids
    in-batch false negatives (a doc that is gold for one query being pushed
    away as a negative for another). Like the base-qid constraint, it is
    best-effort: if no candidate satisfies it at a given step, the constraint
    is relaxed for that pick rather than crashing.
    """
    rng = random.Random(seed)
    use_docs = pos_docids is not None and neg_docids is not None

    buckets: dict = collections.defaultdict(list)
    for idx, bqid in enumerate(base_qids):
        buckets[bqid].append(idx)
    for ids in buckets.values():
        rng.shuffle(ids)

    window = max(effective_batch_size - 1, 0)
    recent = collections.deque(maxlen=window) if window > 0 else None
    recent_count: dict = collections.Counter()  # base qid -> count in window
    win_pos: dict = collections.Counter()        # positive docid -> count in window
    win_neg: dict = collections.Counter()        # negative docid -> count in window

    def doc_ok(sample_idx: int) -> bool:
        # sample's positives not used as a negative in-window, and its
        # negatives not used as a positive in-window.
        return (pos_docids[sample_idx].isdisjoint(win_neg)
                and neg_docids[sample_idx].isdisjoint(win_pos))

    live = [b for b, lst in buckets.items() if lst]
    rng.shuffle(live)

    out: List[int] = []
    while live:
        allowed = [b for b in live if recent_count.get(b, 0) == 0]
        if not allowed:
            allowed = list(live)
        rng.shuffle(allowed)  # random tie-break for equal-length buckets
        if use_docs and window > 0:
            clean = [b for b in allowed if doc_ok(buckets[b][-1])]
            chosen = max(clean or allowed, key=lambda b: len(buckets[b]))
        else:
            chosen = max(allowed, key=lambda b: len(buckets[b]))
        s = buckets[chosen].pop()
        out.append(s)
        if not buckets[chosen]:
            live = [b for b in live if b != chosen]
        if recent is not None:
            if len(recent) == recent.maxlen:
                evicted = recent[0]  # auto-dropped by deque on append below
                eb = base_qids[evicted]
                recent_count[eb] -= 1
                if recent_count[eb] == 0:
                    del recent_count[eb]
                if use_docs:
                    for d in pos_docids[evicted]:
                        win_pos[d] -= 1
                        if win_pos[d] == 0:
                            del win_pos[d]
                    for d in neg_docids[evicted]:
                        win_neg[d] -= 1
                        if win_neg[d] == 0:
                            del win_neg[d]
            recent.append(s)
            recent_count[base_qids[s]] += 1
            if use_docs:
                for d in pos_docids[s]:
                    win_pos[d] += 1
                for d in neg_docids[s]:
                    win_neg[d] += 1

    assert len(out) == len(base_qids)
    return out


class JointLHDistinctBatchTrainDataset(JointLHTrainDataset):
    """JointLH dataset that reorders samples so a mini-batch contains no two
    samples sharing the same *base* query_id.

    Base query_id = first underscore-separated token of `query_id`:
        '123_0' -> '123', '123' -> '123'.

    A fresh order is built each epoch via greedy round-robin over per-base-qid
    buckets, with a sliding window of size (effective_batch_size - 1) recording
    recently emitted base qids. When more samples share the same base qid than
    the window can absorb (e.g. very few distinct base qids), the constraint is
    relaxed for the remaining picks rather than crashing.

    Usage:
      - MUST be paired with `dont_shuffle=True` on the trainer, otherwise the
        HF Trainer's default RandomSampler reshuffles indices and defeats the
        ordering. With dont_shuffle=True, TevatronTrainer uses SequentialSampler
        which walks the dataset in our intended order.
      - `effective_batch_size` should be the per-step batch size the model
        actually sees: per_device_train_batch_size * world_size
        (do not include gradient accumulation).
    """

    def __init__(
        self,
        data_args: DataArguments,
        effective_batch_size: int,
        base_qid_field: str = "query_id",
        trainer=None,
    ):
        super().__init__(data_args, trainer=trainer)
        assert effective_batch_size >= 1
        self.effective_batch_size = effective_batch_size
        self.base_qid_field = base_qid_field

        # One pass over metadata: base qid + pos/neg docid sets. The pos/neg
        # in-batch non-overlap constraint is always on for this dataset.
        self._no_overlap = True
        self._base_qids: List[str] = []
        self._pos_docids: List[set] = []
        self._neg_docids: List[set] = []
        for i in range(len(self.train_data)):
            g = self.train_data[i]
            self._base_qids.append(self._extract_base_qid(g[base_qid_field]))
            if self._no_overlap:
                if g['has_instruction']:
                    pos = {g['positive_passages'][0]['docid']}
                    num_pos_used = 1
                else:
                    pos = {p['docid'] for p in g['positive_passages'][:self.num_positives]}
                    num_pos_used = self.num_positives
                negative_size = data_args.train_group_size - num_pos_used
                self._pos_docids.append(pos)
                self._neg_docids.append(_selected_neg_docids(g, data_args, negative_size))

        self._order: List[int] = list(range(len(self.train_data)))
        self._order_epoch: int = -1

    @staticmethod
    def _extract_base_qid(qid) -> str:
        # '123_0' -> '123';  '123' -> '123';  also handles int qids.
        return str(qid).split("_", 1)[0]

    def _maybe_rebuild_order(self):
        # Same seed across DDP ranks → identical ordering → consistent batches.
        if self.trainer is not None:
            epoch = int(self.trainer.state.epoch)
            base_seed = int(self.trainer.args.seed)
        else:
            epoch = 0
            base_seed = 42
        if epoch != self._order_epoch:
            self._order = _build_distinct_batch_order(
                self._base_qids, self.effective_batch_size,
                seed=base_seed * 1000003 + epoch,
                pos_docids=self._pos_docids if self._no_overlap else None,
                neg_docids=self._neg_docids if self._no_overlap else None,
            )
            self._order_epoch = epoch

    def __getitem__(self, item):
        self._maybe_rebuild_order()
        return super().__getitem__(self._order[item])


class FixedOriginalDistinctBatchTrainDataset(FixedOriginalTrainDataset):
    """FixedOriginalTrainDataset with same-base-qid batch separation.

    Wraps FixedOriginalTrainDataset and reorders samples so no two samples
    sharing the same base query_id appear in the same mini-batch. See
    JointLHDistinctBatchTrainDataset for algorithm and usage constraints
    (MUST be paired with `dont_shuffle=True`).
    """

    def __init__(
        self,
        data_args: DataArguments,
        effective_batch_size: int,
        base_qid_field: str = "query_id",
        trainer=None,
    ):
        super().__init__(data_args, trainer=trainer)
        assert effective_batch_size >= 1
        self.effective_batch_size = effective_batch_size
        self.base_qid_field = base_qid_field

        # One pass over metadata: base qid + pos/neg docid sets. The pos/neg
        # in-batch non-overlap constraint is always on for this dataset.
        self._no_overlap = True
        negative_size = data_args.train_group_size - 1
        self._base_qids: List[str] = []
        self._pos_docids: List[set] = []
        self._neg_docids: List[set] = []
        for i in range(len(self.train_data)):
            g = self.train_data[i]
            self._base_qids.append(str(g[base_qid_field]).split("_", 1)[0])
            if self._no_overlap:
                # one positive is used per epoch (rotated); guard all of them.
                self._pos_docids.append({p['docid'] for p in g['positive_passages']})
                self._neg_docids.append(_selected_neg_docids(g, data_args, negative_size))

        self._order: List[int] = list(range(len(self.train_data)))
        self._order_epoch: int = -1

    def _maybe_rebuild_order(self):
        if self.trainer is not None:
            epoch = int(self.trainer.state.epoch)
            base_seed = int(self.trainer.args.seed)
        else:
            epoch = 0
            base_seed = 42
        if epoch != self._order_epoch:
            self._order = _build_distinct_batch_order(
                self._base_qids, self.effective_batch_size,
                seed=base_seed * 1000003 + epoch,
                pos_docids=self._pos_docids if self._no_overlap else None,
                neg_docids=self._neg_docids if self._no_overlap else None,
            )
            self._order_epoch = epoch

    def __getitem__(self, item):
        self._maybe_rebuild_order()
        return super().__getitem__(self._order[item])


# 수정 필요
class RandLHDistinctBatchTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages'][:3]
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * (math.ceil(remaining_to_select / len(neg_passages)))
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group['new_negatives'][:self.data_args.negatives_first_n]
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * (math.ceil(remaining_to_select / len(neg_passages)))
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
        elif self.data_args.negatives_first_n == 0:
            negs = group_negatives[:negative_size]
            assert len(negs) == negative_size
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages

class MultiposValidDataset(Dataset):
    "instruct flag 있어야 함, jointLH거 그대로 써도 될듯? positive 개수만 고정하면 될듯?"
    pass