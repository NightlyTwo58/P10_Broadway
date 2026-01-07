import torch

class TemporalBroadwayDataset:
    def __init__(self, path):
        self.graphs = torch.load(path)

    def split(self):
        train, val, test = [], [], []

        for g in self.graphs:
            year = g["show"].opening_year.item()
            if year <= 2012:
                train.append(g)
            elif year <= 2016:
                val.append(g)
            else:
                test.append(g)

        return train, val, test
