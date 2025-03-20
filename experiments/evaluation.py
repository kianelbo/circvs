import argparse
import pickle
from functools import partial

import torch as t
from transformer_lens import HookedTransformer
import transformer_lens.utils as lens_utils

from circvs.datatypes import *


def _run_with_mean_ablation(model, prompts, mean_cache, circuit, reverse_ablation=True):
    def hook_ablate_from_mean_cache(value, hook, hooked_comp=None):
        if hooked_comp.head_idx is None:
            value = mean_cache[hooked_comp].to(value.device, value.dtype)
        else:
            value[:, :, hooked_comp.head_idx, :] = mean_cache[hooked_comp].to(value.device, value.dtype)
        return value
    
    if not reverse_ablation:
        mean_ablation_hooks = [(lens_utils.get_act_name(comp.cmpt, comp.layer), partial(hook_ablate_from_mean_cache, hooked_comp=comp)) for comp in circuit]
    else:
        mean_ablation_hooks = []

        for layer in range(model.cfg.n_layers):
            mlp_comp = Node('mlp_post', layer, None, None)
            if mlp_comp not in circuit:
                mean_ablation_hooks.append((lens_utils.get_act_name(mlp_comp.cmpt, mlp_comp.layer), partial(hook_ablate_from_mean_cache, hooked_comp=mlp_comp)))
            for head in range(model.cfg.n_heads):
                head_comp = Component('z', layer, head, None)
                if head_comp not in circuit:
                    mean_ablation_hooks.append((lens_utils.get_act_name(head_comp.cmpt, head_comp.layer), partial(hook_ablate_from_mean_cache, hooked_comp=head_comp)))
        for comp in (Node('resid_pre', 0, None, None), Node('resid_post', model.cfg.n_layers - 1, None, None)):
            if comp not in circuit:
                mean_ablation_hooks.append((lens_utils.get_act_name(comp.cmpt, comp.layer), partial(hook_ablate_from_mean_cache, hooked_comp=comp)))

    ablated_logits = model.run_with_hooks(prompts, fwd_hooks=mean_ablation_hooks)[:, -1]
    return ablated_logits

def compute_faithfulness(model, circuit, prompts, answers, mean_cache):
    good_baseline_logits = model(prompts)[:, -1]    # For the "good" baseline, no components are mean ablated, so its a simple forward pass

    ablated_logits = _run_with_mean_ablation(model, prompts, mean_cache, circuit, reverse_ablation=True) # All non-circuit componetns are mean ablated
    bad_baseline_logits = _run_with_mean_ablation(model, prompts, mean_cache, [], reverse_ablation=True)

    max_val = good_baseline_logits.max(dim=-1).values.view(-1, 1)
    good_baseline_normalized_correct_logits = ((good_baseline_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))

    max_val = bad_baseline_logits.max(dim=-1).values.view(-1, 1)
    bad_baseline_normalized_correct_logits = ((bad_baseline_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))

    max_val = ablated_logits.max(dim=-1).values.view(-1, 1)
    ablated_normalized_correct_logits = ((ablated_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))
    return ((ablated_normalized_correct_logits - bad_baseline_normalized_correct_logits) / (good_baseline_normalized_correct_logits - bad_baseline_normalized_correct_logits)).mean()

def compute_completeness(model, circuit, prompts, answers, mean_cache):
    good_baseline_logits = model(prompts)[:, -1]    # For the "good" baseline, no components are mean ablated, so its a simple forward pass

    ablated_logits = _run_with_mean_ablation(model, prompts, mean_cache, circuit, reverse_ablation=False) # All circuit componetns are mean ablated
    bad_baseline_logits = _run_with_mean_ablation(model, prompts, mean_cache, [], reverse_ablation=True)

    max_val = good_baseline_logits.max(dim=-1).values.view(-1, 1)
    good_baseline_normalized_correct_logits = ((good_baseline_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))

    max_val = bad_baseline_logits.max(dim=-1).values.view(-1, 1)
    bad_baseline_normalized_correct_logits = ((bad_baseline_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))

    max_val = ablated_logits.max(dim=-1).values.view(-1, 1)
    ablated_normalized_correct_logits = ((ablated_logits) / max_val).gather(1, answers.view(-1, 1).to(max_val.device))
    return ((good_baseline_normalized_correct_logits - ablated_normalized_correct_logits) / (good_baseline_normalized_correct_logits - bad_baseline_normalized_correct_logits)).mean()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--circuit", default='circ.pkl', help="circuit file")
    ap.add_argument("-m", "--model", default='gpt2-small', help="model")
    ap.add_argument("-p", "--prompts", default='../data.pkl', help="prompts file")
    ap.add_argument("-a", "--activations", default='../activations.pt', help="mean cache file")
    args = vars(ap.parse_args())

    model = HookedTransformer.from_pretrained(args["model"], device=lens_utils.get_device())
    with open(args["prompts"], "rb") as f:
        prompts_set = pickle.load(f)
    prompts = [p["clean"] for p in prompts_set]
    answers = [model.to_tokens(p["answers"], prepend_bos=False).view(-1) for p in prompts_set]
    mean_cache = t.load(args["activations"])

    with open(args["circuit"], "rb") as f:
        circuit_edges = pickle.load(f)
    circuit = set()
    for e in circuit_edges:
        circuit.add(rename(e.src))
        circuit.add(rename(e.dst))

    faithfulness_score = compute_faithfulness(model, circuit, prompts, answers, mean_cache)
    print("faithfulness score =", faithfulness_score)
    completeness_score = compute_completeness(model, circuit, prompts, answers, mean_cache)
    print("completeness score =", completeness_score)
