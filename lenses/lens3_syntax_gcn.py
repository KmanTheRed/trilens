#!/usr/bin/env python3
"""
Lens-3: Syntax-GCN with POS, dependency and spaCy-vector node features.

*   `doc_to_graph(text)`  → PyG `Data` object
*   `SyntaxGCN`           → tiny GCN that outputs logits
*   `score(text)`         → sigmoid-prob using a (lazy-loaded) default model
"""

import torch
import torch.nn as nn
import spacy
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# --------------------------------------------------------------------------- #
# spaCy model (vectors on CPU)                                                #
# --------------------------------------------------------------------------- #
_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
_MEAN_VEC = _nlp.vocab.vectors.data.mean(0) if _nlp.vocab.vectors else None

# --------------------------------------------------------------------------- #
# Tag mappings                                                                #
# --------------------------------------------------------------------------- #
POS = ["ADJ","ADP","ADV","AUX","CONJ","CCONJ","DET","INTJ","NOUN","NUM",
       "PART","PRON","PROPN","PUNCT","SCONJ","SYM","X"]
POS_MAPPING = {tag:i for i,tag in enumerate(POS)}
POS_MAPPING["UNK"] = len(POS_MAPPING)

DEP = ["acl","acomp","advcl","advmod","agent","amod","appos","attr","aux",
       "auxpass","case","cc","ccomp","compound","conj","cop","csubj",
       "csubjpass","dative","dep","det","dobj","expl","intj","mark","neg",
       "nmod","npmod","nsubj","nsubjpass","nummod","obj","obl","parataxis",
       "pcomp","pobj","poss","preconj","predet","prep","prt","punct",
       "quantmod","relcl","xcomp"]
DEP_MAPPING = {tag:i for i,tag in enumerate(DEP)}
DEP_MAPPING["UNK"] = len(DEP_MAPPING)

VECTOR_DIM = 50                 # slice spaCy’s 300-D vector to this many dims

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def one_hot(idx: int, size: int):
    vec = [0]*size
    if 0 <= idx < size:
        vec[idx] = 1
    return vec

def get_token_feature(tok):
    # POS one-hot
    pos_idx = POS_MAPPING.get(tok.pos_, POS_MAPPING["UNK"])
    pos_vec = one_hot(pos_idx, len(POS_MAPPING))
    # dep one-hot
    dep_idx = DEP_MAPPING.get(tok.dep_, DEP_MAPPING["UNK"])
    dep_vec = one_hot(dep_idx, len(DEP_MAPPING))
    # truncated vector
    if tok.has_vector:
        vec = tok.vector[:VECTOR_DIM]
    else:
        vec = (_MEAN_VEC[:VECTOR_DIM] if _MEAN_VEC is not None
               else [0.0]*VECTOR_DIM)
    return pos_vec + dep_vec + vec.tolist()

# --------------------------------------------------------------------------- #
# Text ➜ graph                                                                #
# --------------------------------------------------------------------------- #
def doc_to_graph(text: str) -> Data:
    doc = _nlp(text)
    src, dst = [], []

    # dependency edges (bidirectional)
    for tok in doc:
        if tok.i != tok.head.i:
            src += [tok.i, tok.head.i]
            dst += [tok.head.i, tok.i]

    # sequential edges
    for i in range(len(doc)-1):
        src += [i, i+1]
        dst += [i+1, i]

    # self-loops
    for i in range(len(doc)):
        src.append(i); dst.append(i)

    x = torch.tensor([get_token_feature(t) for t in doc], dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class SyntaxGCN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid)])
        self.convs.extend(GCNConv(hid, hid) for _ in range(num_layers-1))
        self.do   = nn.Dropout(dropout)
        self.lin  = nn.Linear(hid, 1)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.do(x)
        h = global_mean_pool(x, batch)
        return self.lin(h).view(-1)          # logits

# --------------------------------------------------------------------------- #
# Tiny convenience scorer (lazy model load)                                   #
# --------------------------------------------------------------------------- #
_default_model = None
def _load_default(device="cpu"):
    global _default_model
    if _default_model is None:
        in_dim =  len(POS_MAPPING)+len(DEP_MAPPING) + VECTOR_DIM
        _default_model = SyntaxGCN(in_dim=in_dim).to(device)
        _default_model.eval()
    return _default_model

@torch.no_grad()
def score(text: str, device="cpu") -> float:
    """Return sigmoid-probability from an untrained default model.
       (Replace with your trained checkpoint in real use.)"""
    g = doc_to_graph(text)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    model = _load_default(device)
    logit = model(g.to(device))[0].item()
    return float(torch.sigmoid(torch.tensor(logit)))

# --------------------------------------------------------------------------- #
# Quick manual test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = "The quick brown fox jumps over the lazy dog."
    print("prob =", score(demo))
