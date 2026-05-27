## 1. Paper Information

- Paper name: SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

- Main method: 
  Training-free visual token sparsification for vision-language models. 
  The method estimates the importance of visual tokens using text-guided attention, keeps only a subset of important visual tokens, and reduces redundant visual information during VLM inference.

- Baseline model:
  The main baseline is the original full-token VLM setting, referred to as Dense / Vanilla. 
  In the current thesis, SparseVLM-Original is treated as the main baseline method to compare against the proposed improvement.

- Main purpose of the experiment:
  The original experiment aims to show that a VLM can use fewer visual tokens during inference while preserving most of its performance on standard multimodal benchmarks. 
  For this thesis, the purpose is to use the original SparseVLM experiment as the reference point, then evaluate whether the proposed token selection method can reduce redundancy and preserve or improve performance compared with SparseVLM-Original.

## 2. Backbone / Model Setting

The main image-understanding experiment in the paper uses SparseVLM on LLaVA-1.5, referred to in the paper as SparseLLaVA. In the local SparseVLM code used for this thesis, the evaluation scripts load `liuhaotian/llava-v1.5-13b`, so this is the main backbone setting for the current experiment.

- Backbone:
  LLaVA-1.5 / SparseLLaVA. The paper also reports compatibility experiments on Mini-Gemini (MGM) and Qwen2-VL, but the primary table for the image benchmarks is based on LLaVA.

- Vision encoder:
  CLIP-pretrained ViT-L visual tower. In the local implementation, the corresponding Hugging Face vision tower is `openai/clip-vit-large-patch14-336`, which processes images at 336 x 336 resolution.

- Language model:
  Vicuna / LLaMA-based autoregressive language model. In the local experiment setting, this is `llava-v1.5-13b`, which uses the Vicuna-style conversation mode (`vicuna_v1`).

- Original number of visual tokens:
  576 visual tokens. This corresponds to a 24 x 24 patch grid from the CLIP ViT-L/14 336 px image encoder.

- Sparse token settings:
  The paper evaluates retained-token budgets of 192, 128, and 64 visual tokens for LLaVA. In the local code, these budgets are implemented as progressive layer-wise keep schedules at LLaMA decoder layers 2, 6, and 15:
  - `retained_tokens = 192`: keep 300 tokens at layer 2, 200 at layer 6, and 110 at layer 15.
  - `retained_tokens = 128`: keep 303 tokens at layer 2, 110 at layer 6, and 36 at layer 15.
  - `retained_tokens = 64`: keep 66 tokens at layer 2, 30 at layer 6, and 17 at layer 15.

  Therefore, the paper-level setting should be described as retaining 192 / 128 / 64 visual-token budgets from an original 576-token LLaVA input, while the implementation performs this through progressive pruning inside the language-model layers.


## 3. Benchmark Datasets

The original SparseVLM paper benchmarks the method mainly on image-understanding datasets, then additionally reports video question-answering results to show that the same sparsification idea also works beyond single-image inputs.

| Dataset | Task type | Why it matters |
| --- | --- | --- |
| GQA | Visual question answering with scene-graph-style reasoning | Tests whether pruning preserves object, attribute, and relation information needed for compositional image reasoning. |
| MMBench | General multimodal understanding benchmark | Evaluates broad VLM capability across perception, reasoning, and commonsense-style multimodal questions. |
| MME | Multimodal evaluation benchmark | Measures both perception and cognition ability, making it useful for checking whether SparseVLM keeps general multimodal competence after token reduction. |
| POPE | Object hallucination evaluation | Tests whether pruning visual tokens increases or reduces hallucinated object claims. This is important because aggressive sparsification can remove visual evidence. |
| ScienceQA (SQA) | Science question answering with multimodal context | Evaluates reasoning over images, diagrams, and question text, not only simple visual recognition. |
| SEED-Bench | Generative multimodal comprehension benchmark | Tests image and text understanding across diverse categories, so it helps measure robustness beyond one narrow VQA format. |
| TextVQA / VQAText | Text-centric visual question answering | Checks whether retained visual tokens still preserve small text regions and OCR-related evidence, which are easy to lose during pruning. |
| MMVet | Integrated multimodal capability benchmark | Evaluates higher-level capabilities such as recognition, OCR, spatial understanding, knowledge, and reasoning in a combined setting. |

The paper also evaluates SparseVLM on video-understanding benchmarks:

| Dataset | Task type | Why it matters |
| --- | --- | --- |
| TGIF-QA | Video question answering | Tests whether sparse visual-token selection can handle temporal visual content instead of only static images. |
| MSVD-QA | Video question answering | Measures open-ended QA on short video clips, requiring event and action understanding. |
| MSRVTT-QA | Video question answering | Evaluates video-language understanding on more diverse web videos. |
| ActivityNet-QA | Video question answering | Tests longer activity-level understanding, where important evidence may be spread across time. |

For this thesis, the focus is on the image-understanding benchmarks because the current experiment follows the LLaVA-1.5 / SparseLLaVA single-image setting. Due to the project scope, available hardware, and higher compute/storage cost of video inputs, video-understanding tasks are documented here only for completeness and are not included in the current evaluation.

## 4. Token Settings

| Setting | Number of visual tokens | Meaning |
| --- | --- | --- |
| Dense / Vanilla | 576 | Original LLaVA-1.5 setting where all visual tokens from the CLIP vision encoder are kept. This gives the reference accuracy and highest visual-token compute cost. |
| SparseVLM-192 | 192 | Light sparsification setting. It keeps a relatively large visual-token budget and is expected to stay closest to dense-model performance. |
| SparseVLM-128 | 128 | Medium sparsification setting. It gives a stronger efficiency trade-off while still preserving more visual evidence than the most aggressive setting. |
| SparseVLM-64 | 64 | Aggressive sparsification setting. It has the largest token reduction and is useful for testing whether the selection method can still preserve task-relevant image evidence under a tight budget. |
| SparseVLM-96 | 96 | Optional setting supported by repo. It can be useful as an extra budget between 128 and 64 if time allows. |

For my thesis, the minimum token settings to reproduce should be 128 and 64. If time allows, I will also evaluate 192 and 96.

## 5. Compared Methods

| Method | Role |
| --- | --- |
| Dense / Vanilla | Full-token upper baseline. This is the original LLaVA-1.5 setting where all 576 visual tokens are used, so it provides the reference performance before sparsification. |
| SparseVLM-Original | Main baseline from the original paper. It uses text-guided attention scores to select visual tokens without the proposed redundancy-aware modification. |
| Ours | My proposed token selection method. It keeps the SparseVLM inference framework but changes the token ranking/selection logic to reduce redundant selected patches while preserving question-relevant visual evidence. |
| Threshold Filtering | Additional redundancy-aware baseline suggested by advisor. It uses a simpler filtering rule to remove highly similar or redundant visual tokens, providing a comparison point against the proposed method. |

## 6. Reported Metrics

In the original SparseVLM image-understanding table, the paper reports one main performance number for each benchmark, plus an overall relative accuracy percentage, FLOPs, and latency. The table caption describes the benchmark numbers as raw benchmark performance, but some datasets use their own official scoring format rather than simple classification accuracy.

| Dataset | Metric reported in the paper | Notes |
| --- | --- | --- |
| GQA | Accuracy (%) | Matches the evaluation setup. The repo documentation uses the official GQA evaluation script, which reports answer accuracy. |
| MMBench | Accuracy (%) | Matches the evaluation setup. The local code formats predictions for MMBench, while the final score is obtained from the official/OpenCompass evaluation server. |
| MME | MME total score | Matches the evaluation setup. The paper reports values such as 1864, so this is not a percentage accuracy. The local script calls the official MME `calculation.py` tool after converting answers. |
| POPE | POPE score, reported in the paper as an accuracy-style benchmark number | Partially mismatched in naming. The local `eval_pope.py` prints Accuracy, Precision, Recall, F1, and Yes ratio, but the returned/final averaged value is F1. When reproducing, the exact POPE value should be taken consistently from the same evaluator output used by the paper. |
| ScienceQA (SQA) | Accuracy (%) | Mostly matches. The local ScienceQA evaluator prints both total Accuracy and IMG-Accuracy, while the paper reports a single SQA number. For this thesis, the selected SQA metric should be stated explicitly if ScienceQA is reproduced. |
| SEED-Bench | Accuracy (%) | Matches if using the image split. The local conversion/evaluation script reports total accuracy and also image/video accuracy separately; since this thesis focuses on image understanding, the image accuracy is the relevant one. |
| TextVQA / VQAText | TextVQA accuracy (%) | Matches. The local evaluator uses `TextVQAAccuracyEvaluator`, which computes the standard soft TextVQA accuracy over human answers. |
| MMVet | MM-Vet score (%) | Conceptually matches, but the local repo delegates evaluation to the official MM-Vet notebook/evaluator rather than a fully local metric script. The score should therefore be reproduced using the official MM-Vet evaluation pipeline. |

For the video-understanding benchmarks, the paper reports both Accuracy and GPT evaluation Score for TGIF-QA, MSVD-QA, MSRVTT-QA, and ActivityNet-QA. These metrics are not part of the current thesis evaluation because video understanding is outside the project scope.

## 7. Key Findings from the Original Paper

- SparseVLM shows that many visual tokens in VLM inference are redundant.
  The original LLaVA-1.5 setting uses 576 visual tokens, but the paper shows that performance can be mostly preserved after reducing the visual-token budget to much smaller settings such as 192, 128, or 64.

- Text-guided token selection is more effective than text-agnostic pruning.
  SparseVLM selects visual tokens based on the question/prompt, so different questions can preserve different image regions. This is important because the useful visual evidence depends on what the user asks.

- The method improves inference efficiency without retraining the full model.
  SparseVLM is a training-free inference-time method. It reduces the number of visual tokens processed by later language-model layers, which lowers computation and latency compared with the dense full-token setting.

- Moderate sparsification gives the best accuracy-efficiency trade-off.
  Larger retained-token budgets such as 192 and 128 generally stay closer to dense-model performance, while 64 gives stronger efficiency improvement but has higher risk of losing fine-grained visual evidence.

- SparseVLM is broadly compatible across VLM backbones.
  Although the main image experiments are based on LLaVA-1.5 / SparseLLaVA, the paper also reports experiments on other VLMs such as Mini-Gemini and Qwen2-VL, suggesting that the idea is not limited to one model family.

- The original method does not explicitly optimize diversity among selected visual tokens.
  SparseVLM ranks tokens by text-guided importance, but selected high-scoring tokens can still be visually similar or redundant. This leaves room for this thesis to test a redundancy-aware token selection method that preserves relevance while improving coverage of distinct visual evidence.

## 8. What I Will Reproduce in My Thesis

This thesis reproduces a focused subset of the original SparseVLM experimental setting instead of reproducing all benchmarks from the original paper. The selected protocol is designed to evaluate both general benchmark performance and targeted failure cases related to redundant or missing visual information.

### 8.1 Selected Evaluation Protocol

The official benchmark evaluation includes:

- GQA
- POPE

GQA is selected as the main benchmark because it evaluates visual reasoning, object relations, and spatial relations. These abilities are directly related to whether the model can preserve the necessary visual evidence after token sparsification.

POPE is selected as the secondary benchmark because it evaluates object-level hallucination. This is useful for examining whether different token selection methods preserve reliable object information and reduce incorrect visual grounding.

In addition to the official benchmarks, this thesis also evaluates a failure mining set. This set is used for failure-driven analysis, focusing on cases where SparseVLM-Original fails because of redundant token selection or missing important visual evidence.

### 8.2 Compared Methods

The thesis evaluates the following four methods:

| Method | Role |
| --- | --- |
| Dense / Vanilla | Full-token reference setting |
| SparseVLM-Original | Main baseline from the original paper |
| Ours | Proposed token selection method |
| Threshold Filtering | Additional redundancy-aware baseline |

Dense / Vanilla is included as the full-token reference. SparseVLM-Original is the main baseline that the proposed method directly improves upon. Threshold Filtering is included as an additional redundancy-aware baseline to test whether a simpler similarity-based filtering strategy can address the same problem.

### 8.3 Token Settings

The thesis evaluates the following retain-token settings:

- 128 visual tokens
- 64 visual tokens

The 128-token setting represents a practical sparse setting where the model still keeps a moderate amount of visual information. The 64-token setting represents a more aggressive sparsification setting, where differences between token selection strategies are expected to become clearer.

### 8.4 Metrics

For GQA and POPE, the thesis follows the official metrics used by the corresponding benchmark or by the SparseVLM evaluation scripts.

For the failure mining set, the thesis reports whether each method answers correctly and analyzes the failure pattern qualitatively. The failure analysis focuses on why SparseVLM-Original fails, whether Ours recovers the failure, and how the selected visual tokens differ across methods.

### 8.5 Reproduction Scope

This thesis does not aim to fully reproduce every benchmark from the original SparseVLM paper. Instead, it reproduces a focused experimental subset that is sufficient for the thesis objective.

The reproduction scope includes:

- official benchmark evaluation on GQA and POPE;
- failure-driven analysis on the failure mining set;
- comparison among Dense / Vanilla, SparseVLM-Original, Ours, and Threshold Filtering;
- evaluation under 128-token and 64-token settings.

This protocol supports the central claim of the thesis: the proposed method improves visual token selection by reducing redundancy and preserving more useful visual information, while maintaining performance on standard multimodal benchmarks.
