#!/usr/bin/env python3
import sys, os
import torch

# 1) Make sure "lenses/" is importable
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lenses.lens3_syntax_gcn import doc_to_graph, SyntaxGCN

def main():
    # toy sentence
    text = "This is a quick test of the syntax lens."

    # convert to a PyG graph
    g = doc_to_graph(text)
    print(f"🗺  Graph — nodes={g.x.size(0)}, features_per_node={g.x.size(1)}, edges={g.edge_index.size(1)}")

    # load checkpoint
    ckpt = torch.load("data/best_syntax_gcn.pt", map_location="cpu")

    # infer hidden dimension from the checkpoint's final linear layer weight
    # checkpoint stores 'lin.weight' of shape [1, hidden_dim]
    if "lin.weight" not in ckpt:
        raise KeyError("Couldn't find 'lin.weight' in the checkpoint.")
    hidden_dim = ckpt["lin.weight"].size(1)
    print(f"🎯 Inferred hidden_dim = {hidden_dim} from checkpoint")

    # build model with that hidden size
    model = SyntaxGCN(in_dim=g.x.size(1), hid=hidden_dim)
    model.eval()

    #  filter out any weights whose shapes don't match exactly
    model_state = model.state_dict()
    to_load = {}
    skipped = []
    for key, value in ckpt.items():
        if key in model_state and model_state[key].shape == value.shape:
            to_load[key] = value
        else:
            skipped.append(key)

    # report
    print(f"✅ Loading {len(to_load)} params; skipping {len(skipped)} checkpoint keys:")
    for k in skipped:
        print("   ✗", k)

    # update and load
    model_state.update(to_load)
    model.load_state_dict(model_state)

    # run
    with torch.no_grad():
        # create fake batch index so we can pool
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        out = model(g)

    print("🔍 GCN raw output:", out)
    if out.numel() == 1:
        print("🔢 as Python float:", out.item())

if __name__ == "__main__":
    main()
