
#!/usr/bin/env python3
import torch
from lenses.lens3_syntax_gcn import doc_to_graph, SyntaxGCN

# 1) pick a toy sentence
text = "This is a quick test of the syntax lens."

# 2) turn it into a torch_geometric graph
g = doc_to_graph(text)
print(f"📊  Graph for “{text}” — nodes={g.x.size(0)}, features_per_node={g.x.size(1)}, edges={g.edge_index.size(1)}")

# 3) load and run your trained model
model = SyntaxGCN(in_dim=g.x.size(1))
ckpt = torch.load("data/best_syntax_gcn.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

with torch.no_grad():
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    out = model(g)

print("🔢  GCN raw output tensor:", out)
print("🔢  as Python float:", out.item() if out.numel()==1 else out)

