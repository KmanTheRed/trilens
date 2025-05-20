# ---------------------------------------------
# File: scripts/train_syntax_gcn.py
#!/usr/bin/env python3
import os
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from lenses.lens3_syntax_gcn import doc_to_graph, SyntaxGCN

class JSONLDataset(IterableDataset):
    """Streams graph Data objects from JSONL for train/val splits."""
    def __init__(self, jsonl_path, max_graphs=None, split_ratio=0.8, train=True):
        self.path = jsonl_path
        self.max_graphs = max_graphs
        self.split_ratio = split_ratio
        self.train = train
        if max_graphs:
            self.cutoff = int(max_graphs * split_ratio)
        else:
            self.cutoff = None

    def __iter__(self):
        count = 0
        with open(self.path, encoding='utf8') as f:
            for line in f:
                if self.max_graphs and count >= self.max_graphs:
                    break
                obj = json.loads(line)
                text = obj.get('text')
                if not isinstance(text, str):
                    continue
                try:
                    g = doc_to_graph(text)
                except Exception:
                    continue
                label = float(obj.get('label', 0))
                g.y = torch.tensor([label], dtype=torch.float)
                count += 1
                # yield to train or val
                if self.train:
                    if self.cutoff is None or count <= self.cutoff:
                        yield g
                else:
                    if self.cutoff is None or count > self.cutoff:
                        yield g


def train_gcn(data_root, batch_size, epochs, lr, hid_dim, device, max_graphs):
    jsonl = os.path.join(data_root, 'raw', 'train.jsonl')
    print(f"Streaming from {jsonl} (max={max_graphs})")

    train_ds = JSONLDataset(jsonl, max_graphs, 0.8, train=True)
    val_ds   = JSONLDataset(jsonl, max_graphs, 0.8, train=False)

    train_loader = GeoDataLoader(train_ds, batch_size=batch_size)
    val_loader   = GeoDataLoader(val_ds,   batch_size=batch_size)

    # Infer feature dimension
    sample = next(iter(train_loader))
    in_dim = sample.x.size(1)

    model = SyntaxGCN(in_dim=in_dim, hid=hid_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    os.makedirs(data_root, exist_ok=True)

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        tot_train, n_train = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            tot_train += loss.item() * batch.num_graphs
            n_train   += batch.num_graphs
        train_loss = tot_train / n_train

        # Validation
        model.eval()
        tot_val, n_val, correct = 0.0, 0, 0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.view(-1)
            tot_val += criterion(pred, target).item() * batch.num_graphs
            n_val   += batch.num_graphs
            correct += ((pred > 0.5).float() == target).sum().item()
        val_loss = tot_val / n_val
        val_acc  = correct / n_val

        print(f"Epoch {epoch:02d} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(data_root, 'best_syntax_gcn.pt')
            torch.save(model.state_dict(), ckpt)
            print(f"  [✔] Saved best model to {ckpt}")

    print("🎉 Training complete 🎉")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str, default='data')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--hid_dim',    type=int, default=32)
    p.add_argument('--device',     type=str, default='cuda')
    p.add_argument('--max_graphs', type=int, default=5000,
                   help='max number of graphs to stream')
    args = p.parse_args()

    train_gcn(
        data_root  = args.data_root,
        batch_size = args.batch_size,
        epochs     = args.epochs,
        lr         = args.lr,
        hid_dim    = args.hid_dim,
        device     = args.device,
        max_graphs = args.max_graphs
    )
