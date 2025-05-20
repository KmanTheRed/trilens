# File: lenses/lens3_syntax_gcn.py
import torch
import torch.nn as nn
import spacy
from torch_geometric.nn import GCNConv, global_mean_pool

# Lightweight English parser
_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

class SyntaxGCN(nn.Module):
    def __init__(self, in_dim, hid=32):
        super().__init__()
        self.g1 = GCNConv(in_dim, hid)
        self.lin = nn.Linear(hid, 1)

    def forward(self, data):
        # Node embeddings
        x = torch.relu(self.g1(data.x, data.edge_index))
        # Global mean-pool over each graph in batch
        h = global_mean_pool(x, data.batch)
        # Sigmoid classifier
        return torch.sigmoid(self.lin(h)).view(-1)


def doc_to_graph(input_text):
    """
    Convert raw text or spaCy Doc into a PyG Data graph with
    one feature per node and undirected edges to its syntactic head.
    """
    from torch_geometric.data import Data

    # Parse text if needed
    doc = _nlp(input_text) if isinstance(input_text, str) else input_text

    edges_src, edges_dst = [], []
    for tok in doc:
        i, head = tok.i, tok.head.i
        if head != i:
            edges_src.append(i)
            edges_dst.append(head)

    # 1-dim feature per token
    x = torch.ones((len(doc), 1), dtype=torch.float)
    edge_index = torch.tensor([
        edges_src + edges_dst,
        edges_dst + edges_src
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
