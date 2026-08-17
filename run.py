"""Orchestrate the reproduction:

  extract directions -> pick best layer -> measure baseline / ablate / induce.

Usage:
  python run.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --n-train 96 --n-eval 24 --max-new-tokens 48
"""

from __future__ import annotations

import argparse
import json

import torch

import data as data_mod
import refusal as R


def select_best_layer(model, tok, directions, harmful_val, candidate_layers,
                      max_new_tokens):
    """Pick the layer whose direction most reduces refusal when ablated."""
    best = None
    print("\n[layer-sweep] ablating each candidate direction on val set...")
    for layer_idx in candidate_layers:
        d = directions[layer_idx]
        with R.Intervention(model, d, mode="ablate"):
            comps = R.generate(model, tok, harmful_val,
                               max_new_tokens=max_new_tokens)
        rr = R.refusal_rate(comps)
        print(f"  layer {layer_idx:>2}: post-ablation refusal_rate = {rr:.3f}")
        if best is None or rr < best[1]:
            best = (layer_idx, rr)
    print(f"[layer-sweep] best layer = {best[0]} (refusal_rate {best[1]:.3f})")
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    torch.manual_seed(0)

    harmful_tr, harmless_tr, harmful_ev, harmless_ev = data_mod.load_prompts(
        n_train=args.n_train, n_eval=args.n_eval)

    model, tok = R.load_model(args.model)
    n_layers = len(R._decoder_layers(model))
    print(f"[model] {n_layers} decoder layers")

    # 1. activations -> directions
    print("\n[extract] caching last-token activations (harmful/harmless train)...")
    harm_acts = R.last_token_activations(model, tok, harmful_tr)
    safe_acts = R.last_token_activations(model, tok, harmless_tr)
    directions = R.compute_refusal_directions(harm_acts, safe_acts)  # [L+1, D]

    # 2. pick best layer by ablation on a small val slice (reuse harmful_ev)
    #    candidates: middle band of layers (Arditi finds these most causal)
    lo, hi = int(0.30 * n_layers), int(0.85 * n_layers)
    candidates = list(range(lo, hi + 1, max(1, (hi - lo) // 6)))
    val = harmful_ev[: max(8, len(harmful_ev) // 2)]
    best_layer = select_best_layer(model, tok, directions, val, candidates,
                                   args.max_new_tokens)
    best_dir = directions[best_layer]

    # 3. final measurements on held-out eval sets
    print("\n[eval] baseline (no intervention)...")
    base_harm = R.generate(model, tok, harmful_ev, max_new_tokens=args.max_new_tokens)
    base_safe = R.generate(model, tok, harmless_ev, max_new_tokens=args.max_new_tokens)

    print("[eval] ablate refusal direction (jailbreak)...")
    with R.Intervention(model, best_dir, mode="ablate"):
        abl_harm = R.generate(model, tok, harmful_ev, max_new_tokens=args.max_new_tokens)

    print("[eval] add refusal direction to harmless prompts (induce refusal)...")
    with R.Intervention(model, best_dir, mode="add", add_layer=best_layer, alpha=args.alpha):
        ind_safe = R.generate(model, tok, harmless_ev, max_new_tokens=args.max_new_tokens)

    results = {
        "model": args.model,
        "n_layers": n_layers,
        "best_layer": best_layer,
        "alpha": args.alpha,
        "n_eval": len(harmful_ev),
        "metrics": {
            "baseline_refusal_on_harmful": R.refusal_rate(base_harm),
            "baseline_refusal_on_harmless": R.refusal_rate(base_safe),
            "ablated_refusal_on_harmful": R.refusal_rate(abl_harm),
            "induced_refusal_on_harmless": R.refusal_rate(ind_safe),
        },
    }
    m = results["metrics"]
    results["attack_success_rate_drop"] = (
        m["baseline_refusal_on_harmful"] - m["ablated_refusal_on_harmful"])

    print("\n===== RESULTS =====")
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # a couple of qualitative samples
    print("\n----- sample harmful prompt, baseline vs ablated -----")
    print("PROMPT :", harmful_ev[0])
    print("BASE   :", base_harm[0][:200].replace("\n", " "))
    print("ABLATED:", abl_harm[0][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
