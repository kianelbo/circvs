import argparse
import pickle
from time import time

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
import torch as t

from circvs.datatypes import *


def make_node(node, pos=None):
        layer = int(node.module_name.split(".")[1])
        head_idx = None
        if node.head_idx is not None:
            head_idx = node.head_idx
            cmpt = Component.SA
        elif node.name.startswith("MLP"):
            cmpt = Component.MLP
        elif node.name.startswith("Resid End"):
            cmpt = Component.LMH
        elif node.name.startswith("Resid Start"):
            cmpt = Component.EMB
        else:
            raise TypeError("Invalid component type")
        return Node(cmpt, layer, head_idx, pos)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algorithm", type=str, default='eap', choices=["acdc", "snp", "eap"], help="discovery algorithm")
    ap.add_argument("-t", "--tau", type=float, default=3, help="pruning threshold")
    ap.add_argument("-m", "--model", default='gpt2-small', help="model")
    ap.add_argument("-p", "--prompts", default='../data.json', help="prompts file")
    ap.add_argument("-o", "--output", default='circ.pkl', help="output file")
    args = vars(ap.parse_args())


    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = load_tl_model(args["model"], device)

    path = repo_path_to_abs_path(args["prompts"])
    train_loader, test_loader = load_datasets_from_json(
        model=model,
        path=path,
        device=device,
        prepend_bos=True,
        batch_size=8,
    )

    model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=False,
        device=device,
    )

    t0 = time()

    if args["algorithm"] == "eap":
        attribution_scores = mask_gradient_prune_scores(
            model=model,
            dataloader=train_loader,
            official_edges=None,
            grad_function="logit",
            answer_function="avg_diff",
            mask_val=0.0,
        )
    elif args["algorithm"] == "acdc":
        attribution_scores = acdc_prune_scores(
            model=model,
            dataloader=train_loader,
            official_edges=None,
            tao_exps=[-3, -2],
            tao_bases=[1],
        )
    else:
        attribution_scores = subnetwork_probing_prune_scores(
            model=model,
            dataloader=train_loader,
            official_edges=None,
            learning_rate=0.05,
            epochs=30,
            tree_optimisation=True,
        )

    elapsed = time() - t0
    print("Exec time:", elapsed)

    edges = []

    for pos in model.edge_dict:
        for e in model.edge_dict[pos]:
            edge_score = attribution_scores[e.dest.module_name][e.patch_idx].item()
            if abs(edge_score) < args["tau"]:
                continue
            edges.append(Edge(
                make_node(e.src, pos),
                make_node(e.dest, pos),
                abs(edge_score)
            ))

    with open(args["output"], "wb") as f:
        pickle.dump(edges, f)
