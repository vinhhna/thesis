import torch
import torch.nn as nn
import torch.nn.functional as F

layer_dict = {2:0,6:1,15:2}     # 

sparse_token_list_192 = [300,200,110]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303,110,36]
sparse_token_list_64 = [66,30,17]          

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64 : sparse_token_list_64
}

def _to_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _get_keep_budget(layer_idx, retained_tokens, v_token_num):
    sparse_token_list = sparse_token_dict[retained_tokens]
    return min(sparse_token_list[layer_dict[layer_idx]], max(_to_int(v_token_num) - 1, 0))


def _compute_relation_vis_text(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx):
    v_token_num = _to_int(v_token_num)
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]
    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num] # B, L2, L1
    relation_vis_text = relation_vis_text.mean(1) # B, L1
    return relation_vis_text


def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx,retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.
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
    '''
    Select visual tokens with SparseVLM relevance plus an MMR redundancy penalty.

    self_attn_weights: [B, H, L, L]
    visual_states: [B, V, C], current hidden states for the visual-token block
    '''
    relation_vis_text = _compute_relation_vis_text(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
    )

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.
    v_token_num = _to_int(v_token_num)

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        keep_num = _get_keep_budget(layer_idx, retained_tokens, v_token_num)

        if keep_num > 0:
            candidate_pool_factor = max(int(candidate_pool_factor), 1)
            candidate_num = min(max(keep_num * candidate_pool_factor, keep_num), v_token_num)
            _, candidate_idx = torch.topk(relation_vis, candidate_num, dim=1)

            rel_min = relation_vis.min(dim=1, keepdim=True).values
            rel_max = relation_vis.max(dim=1, keepdim=True).values
            normalized_relevance = (relation_vis - rel_min) / (rel_max - rel_min + 1e-6)

            visual_states = visual_states[:, :v_token_num, :]
            visual_states = F.normalize(visual_states.float(), p=2, dim=-1)
            redundancy_sim = torch.matmul(visual_states, visual_states.transpose(1, 2)).clamp_min(0)
            redundancy_sim = redundancy_sim.to(dtype=normalized_relevance.dtype)

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


def mmr_select(relevance, redundancy_sim, candidate_idx, keep_num, lambda_relevance=0.8):
    '''
    Greedily select final tokens from a relevance candidate pool.

    relevance: [B, V], normalized relevance score
    redundancy_sim: [B, V, V], cosine similarity between visual tokens
    candidate_idx: [B, P], candidate token indices from original top-k relevance
    '''
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
    mask, _, _ = attn_postprocess_mmr(
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
