from dataclasses import dataclass
from enum import Enum


class Component(Enum):
    EMB = 1
    MLP = 2
    SA = 3
    LMH = 4

@dataclass(frozen=True, unsafe_hash=True)
class Node:
    cmpt: Component | str
    layer: int
    head_idx: int | None
    position: int | None

@dataclass(frozen=True, unsafe_hash=True)
class Edge:
    src: Node
    dst: Node
    weight: float

__cmpt2hook = {
    Component.EMB: "resid_pre",
    Component.MLP: "mlp_out",
    Component.SA: "z",
    Component.LMH: "resid_post",
}

def rename(n: Node):
    return Node(__cmpt2hook[n.cmpt], n.layer, n.head_idx, n.position)
