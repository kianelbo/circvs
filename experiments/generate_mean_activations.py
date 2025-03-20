import argparse
import pickle

import torch as t
from transformer_lens import HookedTransformer
import transformer_lens.utils as lens_utils
from tqdm import tqdm

from circvs.datatypes import Node


def _get_hook_dim(model, hook_name):
    return {
        lens_utils.get_act_name('z', 0): model.cfg.d_head,

        lens_utils.get_act_name('mlp_in', 0): model.cfg.d_model,
        lens_utils.get_act_name('mlp_out', 0): model.cfg.d_model,

        lens_utils.get_act_name('resid_pre', 0): model.cfg.d_model,
        lens_utils.get_act_name('resid_post', 0): model.cfg.d_model,        
    }[lens_utils.get_act_name(hook_name, 0)]


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", type=int, default=32, help="batch size")
    ap.add_argument("-p", "--pos", type=int, default=-1, help="position")
    ap.add_argument("-m", "--model", default='gpt2-small', help="model")
    ap.add_argument("-p", "--prompts", default='../data.pkl', help="prompts file")
    ap.add_argument("-o", "--output", default='../activations.pt', help="output file")
    args = vars(ap.parse_args())

    model = HookedTransformer.from_pretrained(args["model"], device=lens_utils.get_device())
    pos = args["pos"]
    with open(args["prompts"], "rb") as f:
        prompts = pickle.load(f)

    all_heads = [(l, h) for h in range(model.cfg.n_heads) for l in range(model.cfg.n_layers)]
    components = [Node('z', l, h, None) for (l, h) in all_heads] + \
                     [Node('mlp_post', l, None, None) for l in range(model.cfg.n_layers)] + \
                     [Node('mlp_in', l, None, None) for l in range(model.cfg.n_layers)]
    components.append(Node('resid_pre', 0, None, None))
    components.append(Node('resid_post', model.cfg.n_layers - 1, None, None))

    activations = [t.zeros(_get_hook_dim(model, component.cmpt)) for component in components]

    # Run batched forward passes through the model, saving the activations of the requested components for each prompt (or sum across prompts in case of mean reduction)
    dataloader = t.utils.data.DataLoader(prompts, batch_size=args["batch_size"], shuffle=False)
    for _, batch in enumerate(tqdm(dataloader)):
        _, cache = model.run_with_cache(batch)
        for j, component in enumerate(components):
            if component.head_idx is None:  # MLP
                activations[j] += cache[lens_utils.get_act_name(component.cmpt, component.layer)][:, pos, :].sum(dim=0).to(activations[j].device)
            else:   # SA
                activations[j] += cache[lens_utils.get_act_name(component.cmpt, component.layer)][:, pos, component.head_idx, :].sum(dim=0).to(activations[j].device)
        
        del cache
        t.cuda.empty_cache()

    for k in range(len(activations)):
        activations[k] = activations[k] / len(prompts)

    cached_activations = {c: a[None, ...].to(device='cpu') for c, a in zip(components, activations)}
    t.save(cached_activations, args["output"])
