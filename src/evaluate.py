import torch
from sklearn.metrics import mean_squared_error

def evaluate(model, graphs):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for g in graphs:
            pred = model(g)
            preds.append(pred.cpu())
            ys.append(g["show"].y.cpu())

    return mean_squared_error(
        torch.cat(ys),
        torch.cat(preds)
    )
