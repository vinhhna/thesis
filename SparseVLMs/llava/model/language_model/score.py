import torch
import torch.nn.functional as F


layer_dict = {2: 0, 6: 1, 15: 2}

sparse_token_list_192 = [300, 200, 110]
sparse_token_list_128 = [303, 110, 36]
sparse_token_list_64 = [66, 30, 17]

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64: sparse_token_list_64,
}

SELECTION_METHODS = {"topk", "mmr", "threshold_fixed", "threshold_adaptive"}
THRESHOLD_ADAPTIVE_ANCHOR_RETAINED_TOKENS = 64


def _to_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _get_keep_budget(layer_idx, retained_tokens, v_token_num):
    sparse_token_list = sparse_token_dict[retained_tokens]
    return min(sparse_token_list[layer_dict[layer_idx]], max(_to_int(v_token_num) - 1, 0))


def _get_candidate_num(keep_num, candidate_pool_factor, v_token_num):
    keep_num = _to_int(keep_num)
    v_token_num = _to_int(v_token_num)
    candidate_pool_factor = max(int(candidate_pool_factor), 1)
    if keep_num <= 0 or v_token_num <= 0:
        return 0
    return min(max(keep_num * candidate_pool_factor, keep_num), v_token_num)


def _compute_relation_vis_text(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx):
    v_token_num = _to_int(v_token_num)
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]
    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start + v_token_num]
    relation_vis_text = relation_vis_text.mean(1)
    return relation_vis_text


def _selected_indices_from_mask(mask):
    return [torch.where(mask[batch])[0].detach().cpu().tolist() for batch in range(mask.shape[0])]


def _base_stats(
    selection_method,
    layer_idx,
    retained_tokens,
    v_token_num,
    keep_num,
    candidate_pool_size,
    mask,
):
    selected_token_indices = _selected_indices_from_mask(mask)
    return {
        "selection_method": selection_method,
        "layer_idx": int(layer_idx),
        "retained_tokens": int(retained_tokens),
        "current_visual_token_count": int(v_token_num),
        "per_layer_budget": int(keep_num),
        "candidate_pool_size": int(candidate_pool_size),
        "selected_count": int(mask.sum().item()),
        "selected_token_indices": selected_token_indices,
        "threshold_selected_count": 0,
        "backfill_outside_pool_threshold_count": 0,
        "backfill_outside_pool_importance_count": 0,
        "backfill_remaining_importance_count": 0,
    }


def _compute_redundancy_sim(visual_states, v_token_num, dtype):
    visual_states = visual_states[:, :v_token_num, :]
    visual_states = F.normalize(visual_states.float(), p=2, dim=-1)
    redundancy_sim = torch.matmul(visual_states, visual_states.transpose(1, 2))
    return redundancy_sim.to(dtype=dtype)


def _selection_pairwise_similarity_stats(visual_states, mask, v_token_num):
    """
    Compute diagnostic selected-token similarity aggregates without storing vectors.

    The similarities are computed over current visual-token hidden states at the
    pruning layer. They are not based on original CLIP patch IDs, so they remain
    separate from spatial/Jaccard metrics that use selected_original_token_indices.
    """
    v_token_num = _to_int(v_token_num)
    selected_counts = [int(mask[batch].sum().item()) for batch in range(mask.shape[0])]
    stats = {
        "pairwise_similarity_available": False,
        "pairwise_similarity_token_count": int(sum(selected_counts)),
        "mean_pairwise_similarity": None,
        "median_pairwise_similarity": None,
        "max_pairwise_similarity": None,
        "p90_pairwise_similarity": None,
        "similarity_above_0.80_ratio": None,
        "similarity_above_0.85_ratio": None,
        "similarity_above_0.90_ratio": None,
    }
    if v_token_num <= 0:
        return stats

    normalized = F.normalize(visual_states[:, :v_token_num, :].float(), p=2, dim=-1)
    values = []
    for batch in range(mask.shape[0]):
        selected = torch.where(mask[batch])[0]
        if selected.numel() < 2:
            continue
        selected_states = normalized[batch, selected, :]
        similarity = torch.matmul(selected_states, selected_states.transpose(0, 1))
        off_diagonal = similarity[~torch.eye(selected.numel(), dtype=torch.bool, device=similarity.device)]
        values.append(off_diagonal.detach().float().cpu())

    if not values:
        return stats

    similarities = torch.cat(values)
    stats.update({
        "pairwise_similarity_available": True,
        "mean_pairwise_similarity": float(similarities.mean().item()),
        "median_pairwise_similarity": float(similarities.median().item()),
        "max_pairwise_similarity": float(similarities.max().item()),
        "p90_pairwise_similarity": float(torch.quantile(similarities, 0.90).item()),
        "similarity_above_0.80_ratio": float((similarities > 0.80).float().mean().item()),
        "similarity_above_0.85_ratio": float((similarities > 0.85).float().mean().item()),
        "similarity_above_0.90_ratio": float((similarities > 0.90).float().mean().item()),
    })
    return stats


def _attach_pairwise_similarity_stats(stats, visual_states, mask, v_token_num, record_selection_similarity):
    if record_selection_similarity:
        stats.update(_selection_pairwise_similarity_stats(visual_states, mask, v_token_num))
    return stats


def _passes_similarity_threshold(redundancy_sim, batch_idx, token_idx, selected_indices, threshold_tau):
    if len(selected_indices) == 0:
        return True
    selected = torch.tensor(selected_indices, device=redundancy_sim.device, dtype=torch.long)
    max_similarity = redundancy_sim[batch_idx, token_idx, selected].max()
    return bool(max_similarity.item() <= threshold_tau)


def attn_postprocess_topk(
    self_attn_weights,
    v_token_start,
    v_token_num,
    text_token_start,
    t_token_idx,
    layer_idx,
    retained_tokens,
):
    """
    Original SparseVLM importance-based top-k selection.

    self_attn_weights: [B, H, L, L]
    """
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )

    relation_vis = relation_vis_text
    s_flag = True
    v_token_num = _to_int(v_token_num)

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        keep_num = _get_keep_budget(layer_idx, retained_tokens, v_token_num)
        if keep_num > 0:
            _, indices = torch.topk(relation_vis, keep_num, dim=1)
            mask.scatter_(1, indices, True)
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text


def attn_postprocess_mmr(
    self_attn_weights,
    visual_states,
    v_token_start,
    v_token_num,
    text_token_start,
    t_token_idx,
    layer_idx,
    retained_tokens,
    lambda_relevance=0.8,
    candidate_pool_factor=2,
):
    """
    Select visual tokens with SparseVLM relevance plus an MMR redundancy penalty.

    self_attn_weights: [B, H, L, L]
    visual_states: [B, V, C], current hidden states for the visual-token block
    """
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )

    relation_vis = relation_vis_text
    s_flag = True
    v_token_num = _to_int(v_token_num)

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        keep_num = _get_keep_budget(layer_idx, retained_tokens, v_token_num)

        if keep_num > 0:
            candidate_num = _get_candidate_num(keep_num, candidate_pool_factor, v_token_num)
            _, candidate_idx = torch.topk(relation_vis, candidate_num, dim=1)

            rel_min = relation_vis.min(dim=1, keepdim=True).values
            rel_max = relation_vis.max(dim=1, keepdim=True).values
            normalized_relevance = (relation_vis - rel_min) / (rel_max - rel_min + 1e-6)

            redundancy_sim = _compute_redundancy_sim(
                visual_states,
                v_token_num,
                normalized_relevance.dtype,
            ).clamp_min(0)

            selected_idx = mmr_select(
                normalized_relevance,
                redundancy_sim,
                candidate_idx,
                keep_num,
                lambda_relevance,
            )
            mask.scatter_(1, selected_idx, True)
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text


def attn_postprocess_threshold_fixed(
    self_attn_weights,
    visual_states,
    v_token_start,
    v_token_num,
    text_token_start,
    t_token_idx,
    layer_idx,
    retained_tokens,
    threshold_tau=0.85,
    candidate_pool_factor=2,
):
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )
    mask, s_flag, stats = threshold_postprocess_from_relation(
        relation_vis_text,
        visual_states,
        layer_idx,
        retained_tokens,
        threshold_tau=threshold_tau,
        candidate_pool_factor=candidate_pool_factor,
        adaptive=False,
    )
    return mask, s_flag, relation_vis_text, stats


def attn_postprocess_threshold_adaptive(
    self_attn_weights,
    visual_states,
    v_token_start,
    v_token_num,
    text_token_start,
    t_token_idx,
    layer_idx,
    retained_tokens,
    threshold_tau=0.85,
    candidate_pool_factor=2,
    adaptive_anchor_retained_tokens=THRESHOLD_ADAPTIVE_ANCHOR_RETAINED_TOKENS,
):
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )
    mask, s_flag, stats = threshold_postprocess_from_relation(
        relation_vis_text,
        visual_states,
        layer_idx,
        retained_tokens,
        threshold_tau=threshold_tau,
        candidate_pool_factor=candidate_pool_factor,
        adaptive=True,
        adaptive_anchor_retained_tokens=adaptive_anchor_retained_tokens,
    )
    return mask, s_flag, relation_vis_text, stats


def threshold_postprocess_from_relation(
    relation_vis,
    visual_states,
    layer_idx,
    retained_tokens,
    threshold_tau=0.85,
    candidate_pool_factor=2,
    adaptive=False,
    adaptive_anchor_retained_tokens=THRESHOLD_ADAPTIVE_ANCHOR_RETAINED_TOKENS,
):
    """
    Threshold filtering from a top-k candidate pool.

    Returns:
        mask, s_flag, stats
    """
    v_token_num = int(relation_vis.shape[1])
    s_flag = v_token_num != 0
    mask = torch.zeros_like(relation_vis, dtype=bool)

    budget_retained_tokens = adaptive_anchor_retained_tokens if adaptive else retained_tokens
    keep_num = _get_keep_budget(layer_idx, budget_retained_tokens, v_token_num) if v_token_num != 0 else 0
    candidate_num = _get_candidate_num(keep_num, candidate_pool_factor, v_token_num)
    method = "threshold_adaptive" if adaptive else "threshold_fixed"

    stats = _base_stats(
        method,
        layer_idx,
        retained_tokens,
        v_token_num,
        keep_num,
        candidate_num,
        mask,
    )
    stats["threshold_tau"] = float(threshold_tau)
    if adaptive:
        stats["adaptive_anchor_retained_tokens"] = int(adaptive_anchor_retained_tokens)

    if v_token_num == 0:
        s_flag = False
        mask = torch.ones_like(relation_vis, dtype=bool)
        stats["selected_token_indices"] = _selected_indices_from_mask(mask)
        return mask, s_flag, stats

    if keep_num <= 0 or candidate_num <= 0:
        return mask, s_flag, stats

    _, candidate_idx = torch.topk(relation_vis, candidate_num, dim=1)
    importance_order = torch.argsort(relation_vis, dim=1, descending=True)
    redundancy_sim = _compute_redundancy_sim(visual_states, v_token_num, relation_vis.dtype)

    threshold_selected_count = 0
    outside_threshold_count = 0
    outside_importance_count = 0
    remaining_importance_count = 0

    for batch_idx in range(relation_vis.shape[0]):
        selected = []
        candidate_list = candidate_idx[batch_idx].detach().cpu().tolist()
        candidate_set = set(candidate_list)

        for token_idx in candidate_list:
            if _passes_similarity_threshold(redundancy_sim, batch_idx, token_idx, selected, threshold_tau):
                selected.append(token_idx)
                threshold_selected_count += 1
                if not adaptive and len(selected) >= keep_num:
                    break

        if not adaptive and len(selected) < keep_num:
            outside_order = [
                int(idx)
                for idx in importance_order[batch_idx].detach().cpu().tolist()
                if int(idx) not in candidate_set and int(idx) not in selected
            ]

            for token_idx in outside_order:
                if _passes_similarity_threshold(redundancy_sim, batch_idx, token_idx, selected, threshold_tau):
                    selected.append(token_idx)
                    outside_threshold_count += 1
                    if len(selected) >= keep_num:
                        break

            if len(selected) < keep_num:
                for token_idx in outside_order:
                    if token_idx not in selected:
                        selected.append(token_idx)
                        outside_importance_count += 1
                        if len(selected) >= keep_num:
                            break

            if len(selected) < keep_num:
                for token_idx in importance_order[batch_idx].detach().cpu().tolist():
                    token_idx = int(token_idx)
                    if token_idx not in selected:
                        selected.append(token_idx)
                        remaining_importance_count += 1
                        if len(selected) >= keep_num:
                            break

        if len(selected) > 0:
            selected_idx = torch.tensor(selected, device=relation_vis.device, dtype=torch.long)
            mask[batch_idx, selected_idx] = True

    stats.update({
        "selected_count": int(mask.sum().item()),
        "selected_token_indices": _selected_indices_from_mask(mask),
        "threshold_selected_count": int(threshold_selected_count),
        "backfill_outside_pool_threshold_count": int(outside_threshold_count),
        "backfill_outside_pool_importance_count": int(outside_importance_count),
        "backfill_remaining_importance_count": int(remaining_importance_count),
    })
    return mask, s_flag, stats


def attn_postprocess_select(
    selection_method,
    self_attn_weights,
    visual_states,
    v_token_start,
    v_token_num,
    text_token_start,
    t_token_idx,
    layer_idx,
    retained_tokens,
    threshold_tau=0.85,
    candidate_pool_factor=2,
    lambda_relevance=0.8,
    record_selection_similarity=False,
):
    selection_method = selection_method.lower()
    if selection_method not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection_method '{selection_method}'. Expected one of {sorted(SELECTION_METHODS)}.")

    v_token_num_int = _to_int(v_token_num)

    if selection_method == "topk":
        mask, s_flag, relation_vis_text = attn_postprocess_topk(
            self_attn_weights,
            v_token_start,
            v_token_num,
            text_token_start,
            t_token_idx,
            layer_idx,
            retained_tokens,
        )
        keep_num = _get_keep_budget(layer_idx, retained_tokens, v_token_num_int) if v_token_num_int != 0 else 0
        stats = _base_stats("topk", layer_idx, retained_tokens, v_token_num_int, keep_num, keep_num, mask)
        stats = _attach_pairwise_similarity_stats(
            stats, visual_states, mask, v_token_num_int, record_selection_similarity
        )
        return mask, s_flag, relation_vis_text, stats

    if selection_method == "mmr":
        mask, s_flag, relation_vis_text = attn_postprocess_mmr(
            self_attn_weights,
            visual_states,
            v_token_start,
            v_token_num,
            text_token_start,
            t_token_idx,
            layer_idx,
            retained_tokens,
            lambda_relevance=lambda_relevance,
            candidate_pool_factor=candidate_pool_factor,
        )
        keep_num = _get_keep_budget(layer_idx, retained_tokens, v_token_num_int) if v_token_num_int != 0 else 0
        candidate_num = _get_candidate_num(keep_num, candidate_pool_factor, v_token_num_int)
        stats = _base_stats("mmr", layer_idx, retained_tokens, v_token_num_int, keep_num, candidate_num, mask)
        stats["lambda_relevance"] = float(lambda_relevance)
        stats = _attach_pairwise_similarity_stats(
            stats, visual_states, mask, v_token_num_int, record_selection_similarity
        )
        return mask, s_flag, relation_vis_text, stats

    if selection_method == "threshold_fixed":
        mask, s_flag, relation_vis_text, stats = attn_postprocess_threshold_fixed(
            self_attn_weights,
            visual_states,
            v_token_start,
            v_token_num,
            text_token_start,
            t_token_idx,
            layer_idx,
            retained_tokens,
            threshold_tau=threshold_tau,
            candidate_pool_factor=candidate_pool_factor,
        )
        stats = _attach_pairwise_similarity_stats(
            stats, visual_states, mask, v_token_num_int, record_selection_similarity
        )
        return mask, s_flag, relation_vis_text, stats

    mask, s_flag, relation_vis_text, stats = attn_postprocess_threshold_adaptive(
        self_attn_weights,
        visual_states,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
        layer_idx,
        retained_tokens,
        threshold_tau=threshold_tau,
        candidate_pool_factor=candidate_pool_factor,
    )
    stats = _attach_pairwise_similarity_stats(
        stats, visual_states, mask, v_token_num_int, record_selection_similarity
    )
    return mask, s_flag, relation_vis_text, stats


def mmr_select(relevance, redundancy_sim, candidate_idx, keep_num, lambda_relevance=0.8):
    """
    Greedily select final tokens from a relevance candidate pool.

    relevance: [B, V], normalized relevance score
    redundancy_sim: [B, V, V], cosine similarity between visual tokens
    candidate_idx: [B, P], candidate token indices from original top-k relevance
    """
    batch_size, pool_num = candidate_idx.shape
    lambda_relevance = float(lambda_relevance)
    lambda_relevance = min(max(lambda_relevance, 0.0), 1.0)

    candidate_relevance = relevance.gather(1, candidate_idx)
    available = torch.ones(batch_size, pool_num, dtype=torch.bool, device=relevance.device)
    selected = []
    batch_idx = torch.arange(batch_size, device=relevance.device)

    for _ in range(keep_num):
        if len(selected) == 0:
            mmr_score = candidate_relevance
        else:
            selected_idx = torch.stack(selected, dim=1)
            redundancy = torch.zeros(batch_size, pool_num, dtype=relevance.dtype, device=relevance.device)
            for batch in range(batch_size):
                candidate_similarity = redundancy_sim[batch, candidate_idx[batch]]
                redundancy[batch] = candidate_similarity[:, selected_idx[batch]].max(dim=1).values
            mmr_score = lambda_relevance * candidate_relevance - (1 - lambda_relevance) * redundancy

        mmr_score = mmr_score.masked_fill(~available, torch.finfo(mmr_score.dtype).min)
        selected_pool_idx = mmr_score.argmax(dim=1)
        selected_token_idx = candidate_idx[batch_idx, selected_pool_idx]
        selected.append(selected_token_idx)
        available[batch_idx, selected_pool_idx] = False

    return torch.stack(selected, dim=1)


if __name__ == "__main__":
    batch_size, num_heads, v_token_start, v_token_num, text_token_num = 1, 16, 36, 576, 53
    text_token_start = v_token_start + v_token_num
    seq_len = text_token_start + text_token_num
    self_attn_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
    visual_states = torch.rand(batch_size, v_token_num, 4096)
    t_token_idx = torch.where(torch.ones(batch_size, text_token_num, dtype=torch.bool))
    mask, _, _, stats = attn_postprocess_select(
        "mmr",
        self_attn_weights,
        visual_states,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
        layer_idx=15,
        retained_tokens=64,
    )
    print(mask.shape)
    print(stats)
