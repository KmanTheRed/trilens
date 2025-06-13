#!/usr/bin/env python3
"""
Stream-friendly Syntax-GCN trainer with tqdm + gradient accumulation
and robust JSONL parsing.
"""

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score

from lenses.lens3_syntax_gcn import SyntaxGCN, doc_to_graph

# --------------------------------------------------------------------------- #
# CONFIGURABLE FIELD NAMES                                                    #
# --------------------------------------------------------------------------- #
TEXT_KEYS  = ("text", "content", "body", "human_text", "machine_text")
LABEL_KEYS = ("label", "is_ai")

# --------------------------------------------------------------------------- #
# UTILITIES                                                                   #
# --------------------------------------------------------------------------- #
def extract(row):
    """
    Return (text, label) or (None, None) if the row is unusable.
    • Accepts explicit or implicit labels
    • Skips non-string or empty text
    """
    txt = next((row[k] for k in TEXT_KEYS if k in row), None)
    lbl = next((row[k] for k in LABEL_KEYS if k in row), None)

    # implicit labels
    if lbl is None:
        if "machine_text" in row:
            lbl, txt = 1, row["machine_text"]
        elif "human_text" in row:
            lbl, txt = 0, row["human_text"]

    # sanity checks
    if not isinstance(txt, str) or not txt.strip():
        return None, None
    if lbl not in (0, 1):
        return None, None

    return txt, lbl

# --------------------------------------------------------------------------- #
# STREAMING DATASET                                                           #
# --------------------------------------------------------------------------- #
class JsonlStream(IterableDataset):
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf8") as fh:
            for line in fh:
                text, label = extract(json.loads(line))
                if text is None:
                    continue
                g = doc_to_graph(text)
                g.y = torch.tensor([label], dtype=torch.float)
                yield g

def collate(batch):
    return Batch.from_data_list(batch)

# --------------------------------------------------------------------------- #
# EVALUATION                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        prob  = torch.sigmoid(model(batch)).view(-1).cpu()
        y_true += batch.y.view(-1).cpu().tolist()
        y_pred += (prob >= 0.5).int().tolist()
    return (
        f1_score(y_true, y_pred, zero_division=0),
        accuracy_score(y_true, y_pred),
    )

# --------------------------------------------------------------------------- #
# TRAIN LOOP                                                                  #
# --------------------------------------------------------------------------- #
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        JsonlStream(args.train_jsonl),
        batch_size=args.batch,
        collate_fn=collate,
        num_workers=args.workers,
    )

    # dev set (assumed small) is materialised once
    dev_graphs = list(JsonlStream(args.dev_jsonl))
    dev_loader = DataLoader(
        dev_graphs, batch_size=args.batch, collate_fn=collate, shuffle=False
    )

    in_dim = doc_to_graph("dummy").x.size(1)
    net    = SyntaxGCN(in_dim=in_dim, hid=args.hid,
                       num_layers=args.layers, dropout=args.drop).to(device)

    loss_fn = BCEWithLogitsLoss()
    opt     = Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    best_f1 = 0.0
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        bar = tqdm(train_loader, desc=f"E{epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(bar, 1):
            batch = batch.to(device)
            loss  = loss_fn(net(batch).view(-1), batch.y.view(-1)) / args.accum
            loss.backward()
            if step % args.accum == 0:
                opt.step(); opt.zero_grad()
            total += loss.item() * args.accum
            bar.set_postfix(loss=total/step)

        f1, acc = evaluate(net, dev_loader, device)
        print(f"Epoch {epoch:02d}  dev-F1={f1:.4f}  dev-Acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(net.state_dict(), args.ckpt)
            print(f"  ↳ new best; saved → {args.ckpt}")

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--dev_jsonl",   required=True)
    p.add_argument("--ckpt",        required=True)

    p.add_argument("--hid",    type=int,   default=64)
    p.add_argument("--layers", type=int,   default=2)
    p.add_argument("--drop",   type=float, default=0.3)

    p.add_argument("--batch",  type=int,   default=4)
    p.add_argument("--accum",  type=int,   default=16)
    p.add_argument("--workers",type=int,   default=0)

    p.add_argument("--epochs", type=int,   default=12)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--wd",     type=float, default=1e-5)
    p.add_argument("--device", default="cuda")

    run(p.parse_args())
