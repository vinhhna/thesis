import importlib.util
import pathlib
import unittest

import torch


SCORE_PATH = pathlib.Path(__file__).resolve().parents[1] / "llava" / "model" / "language_model" / "score.py"
SPEC = importlib.util.spec_from_file_location("sparse_score", SCORE_PATH)
sparse_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sparse_score)


def relation_scores(num_tokens):
    return torch.arange(num_tokens, 0, -1, dtype=torch.float32).unsqueeze(0)


def identity_states(num_tokens):
    return torch.eye(num_tokens, dtype=torch.float32).unsqueeze(0)


def same_states(num_tokens, dim=None):
    dim = dim or num_tokens
    states = torch.zeros(1, num_tokens, dim, dtype=torch.float32)
    states[:, :, 0] = 1
    return states


def candidate_same_outside_unique_states(num_tokens, candidate_count):
    states = torch.zeros(1, num_tokens, num_tokens, dtype=torch.float32)
    states[:, :candidate_count, 0] = 1
    for idx in range(candidate_count, num_tokens):
        states[0, idx, idx] = 1
    return states


def attention_from_relation(relation):
    batch, num_tokens = relation.shape
    seq_len = num_tokens + 1
    attn = torch.zeros(batch, 1, seq_len, seq_len, dtype=relation.dtype)
    attn[:, 0, num_tokens, :num_tokens] = relation
    text_idx = torch.where(torch.ones(batch, 1, dtype=torch.bool))
    return attn, text_idx


class SparseSelectionTests(unittest.TestCase):
    def test_topk_dispatch_matches_original_importance_topk(self):
        num_tokens = 40
        relation = relation_scores(num_tokens)
        attn, text_idx = attention_from_relation(relation)

        mask, _, relation_vis, stats = sparse_score.attn_postprocess_select(
            "topk",
            attn,
            same_states(num_tokens),
            0,
            num_tokens,
            num_tokens,
            text_idx,
            layer_idx=15,
            retained_tokens=64,
        )

        keep_num = sparse_score._get_keep_budget(15, 64, num_tokens)
        expected = torch.topk(relation_vis, keep_num, dim=1).indices[0].tolist()
        actual = torch.where(mask[0])[0].tolist()

        self.assertEqual(set(actual), set(expected))
        self.assertEqual(stats["selection_method"], "topk")
        self.assertEqual(stats["threshold_selected_count"], 0)
        self.assertEqual(stats["backfill_remaining_importance_count"], 0)

    def test_threshold_fixed_backfills_outside_pool_before_rejected_candidates(self):
        num_tokens = 40
        keep_num = sparse_score._get_keep_budget(15, 64, num_tokens)
        candidate_count = sparse_score._get_candidate_num(keep_num, 2, num_tokens)
        relation = relation_scores(num_tokens)
        states = candidate_same_outside_unique_states(num_tokens, candidate_count)

        mask, _, stats = sparse_score.threshold_postprocess_from_relation(
            relation,
            states,
            layer_idx=15,
            retained_tokens=64,
            threshold_tau=0.85,
            candidate_pool_factor=2,
            adaptive=False,
        )

        self.assertEqual(int(mask.sum().item()), keep_num)
        self.assertEqual(stats["threshold_selected_count"], 1)
        self.assertEqual(stats["backfill_outside_pool_threshold_count"], num_tokens - candidate_count)
        self.assertEqual(stats["backfill_outside_pool_importance_count"], 0)
        self.assertEqual(
            stats["backfill_remaining_importance_count"],
            keep_num - 1 - (num_tokens - candidate_count),
        )

    def test_threshold_fixed_importance_backfill_runs_before_last_resort(self):
        num_tokens = 40
        keep_num = sparse_score._get_keep_budget(15, 64, num_tokens)
        candidate_count = sparse_score._get_candidate_num(keep_num, 2, num_tokens)
        relation = relation_scores(num_tokens)

        mask, _, stats = sparse_score.threshold_postprocess_from_relation(
            relation,
            same_states(num_tokens),
            layer_idx=15,
            retained_tokens=64,
            threshold_tau=0.85,
            candidate_pool_factor=2,
            adaptive=False,
        )

        self.assertEqual(int(mask.sum().item()), keep_num)
        self.assertEqual(stats["threshold_selected_count"], 1)
        self.assertEqual(stats["backfill_outside_pool_threshold_count"], 0)
        self.assertEqual(stats["backfill_outside_pool_importance_count"], num_tokens - candidate_count)
        self.assertEqual(
            stats["backfill_remaining_importance_count"],
            keep_num - 1 - (num_tokens - candidate_count),
        )

    def test_threshold_adaptive_is_restricted_to_candidate_pool_without_backfill(self):
        num_tokens = 40
        keep_num = sparse_score._get_keep_budget(15, 64, num_tokens)
        candidate_count = sparse_score._get_candidate_num(keep_num, 2, num_tokens)

        mask, _, stats = sparse_score.threshold_postprocess_from_relation(
            relation_scores(num_tokens),
            identity_states(num_tokens),
            layer_idx=15,
            retained_tokens=128,
            threshold_tau=0.85,
            candidate_pool_factor=2,
            adaptive=True,
        )

        self.assertEqual(candidate_count, 34)
        self.assertEqual(int(mask.sum().item()), candidate_count)
        self.assertEqual(torch.where(mask[0])[0].tolist(), list(range(candidate_count)))
        self.assertEqual(stats["candidate_pool_size"], candidate_count)
        self.assertEqual(stats["backfill_remaining_importance_count"], 0)

    def test_threshold_fixed_caps_budget_when_current_visual_tokens_are_small(self):
        num_tokens = 5
        keep_num = sparse_score._get_keep_budget(15, 64, num_tokens)

        mask, _, stats = sparse_score.threshold_postprocess_from_relation(
            relation_scores(num_tokens),
            identity_states(num_tokens),
            layer_idx=15,
            retained_tokens=64,
            threshold_tau=0.85,
            candidate_pool_factor=2,
            adaptive=False,
        )

        self.assertEqual(keep_num, 4)
        self.assertEqual(int(mask.sum().item()), keep_num)
        self.assertEqual(stats["current_visual_token_count"], num_tokens)
        self.assertEqual(stats["per_layer_budget"], keep_num)


if __name__ == "__main__":
    unittest.main()
