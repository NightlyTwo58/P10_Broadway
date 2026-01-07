import torch
from temporal_dataset import TemporalBroadwayDataset
from model import BroadwayGNN
from config import *

dataset = TemporalBroadwayDataset("../data/processed/graph_snapshots.pt")
train_graphs, val_graphs, test_graphs = dataset.split()

model = BroadwayGNN(train_graphs[0].metadata()).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

def train_epoch(graphs):
    model.train()
    total_loss = 0

    for g in graphs:
        g = g.to(DEVICE)
        pred = model(g)
        loss = torch.nn.functional.huber_loss(pred, g["show"].y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(graphs)

for epoch in range(50):
    loss = train_epoch(train_graphs)
    print(f"Epoch {epoch}: train loss = {loss:.4f}")
