# Refusal is mediated by a single direction: a small reproduction

A from-scratch reproduction of Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)), on **Qwen2.5-0.5B-Instruct**.

## Method
1. Cache last-token activations on harmful prompts (AdvBench, Zou et al. 2023) and harmless prompts (Alpaca).
2. At each layer, the refusal direction is the difference of the two means, normalised.
3. Choose the layer whose direction most reduces refusal when it is ablated. The search runs over the middle layers, on a validation slice.
4. **Ablate** that direction at every layer and position on harmful prompts, which should remove refusal.
5. **Add** it at the chosen layer on harmless prompts, which should induce refusal.

Refusal is scored with the standard AdvBench/GCG refusal-phrase check on the first ~64 tokens.

## Results (`results_fixed.json`, 24 evaluation prompts per set; best layer 11 of 24; addition strength 8)

| | Refusal rate |
|---|---|
| Harmful prompts, baseline | 83% |
| Harmful prompts, direction ablated | **8%** |
| Harmless prompts, baseline | 4% |
| Harmless prompts, direction added | **96%** |

Limits: one small model, 24 prompts per set, and a phrase-based refusal check. These numbers show the effect is reproducible; they are not a precise measurement.

## Run
```
python run.py --model Qwen/Qwen2.5-0.5B-Instruct --n-eval 24
```
Files: `data.py` (prompt loading), `refusal.py` (directions, interventions, refusal check), `run.py` (pipeline).
