# SparseVLM Rebuild Plan

## Baseline choice

- Use `SparseVLMs` as the reference repository.
- Treat `v1.5` as the baseline for reproducing the original SparseVLM paper.
- Treat `main` / `v2.0` as the follow-up improvement baseline (`SparseVLM+`).

## What to rebuild first

1. Baseline LLaVA inference path.
2. Multimodal input packing with image tokens inserted into the text stream.
3. Sparse decoder path that prunes visual tokens during prefill.
4. Token scoring and retention schedule.
5. Evaluation scripts for the same benchmarks used by the paper.

## Key code map

- Repo usage and branch note: `SparseVLMs/README.md`
- Model loader switch for sparse inference: `SparseVLMs/llava/model/builder.py`
- Multimodal embedding assembly: `SparseVLMs/llava/model/llava_arch.py`
- Sparse model wrapper: `SparseVLMs/llava/model/language_model/sparse_llava_llama.py`
- Sparse decoder implementation: `SparseVLMs/llava/model/language_model/modelling_sparse_llama.py`
- Token retention logic: `SparseVLMs/llava/model/language_model/score.py`
- Evaluation entry scripts: `SparseVLMs/scripts/v1_5/eval/`

## Recommended execution order

1. Pin the exact baseline branch or commit to reproduce.
2. Create a clean environment and install dependencies.
3. Run one normal LLaVA inference path to verify the base stack.
4. Trace the sparse path end to end on a single image-question example.
5. Confirm retained-token schedules for `192 / 128 / 96 / 64`.
6. Reproduce one benchmark script before attempting modifications.
7. Only after reproduction, introduce your improvements behind a config flag.

## Immediate next tasks

- Check out the `v1.5` branch in the cloned repo for paper-faithful reproduction.
- Document exact model checkpoints and datasets required for the chosen benchmark.
- Add a minimal debug script that logs:
  - image token count before pruning
  - selected text tokens
  - retained visual tokens per pruning layer
  - final generation output
