import pandas as pd
import torch
from torch_geometric.data import HeteroData

def build_graph_snapshots():
    shows = pd.read_csv("../data/raw/shows.csv")
    people = pd.read_csv("../data/raw/people.csv")
    credits = pd.read_csv("../data/raw/credits.csv")
    theaters = pd.read_csv("../data/raw/theaters.csv")

    shows = shows.sort_values("opening_date")

    snapshots = []
    active_shows = []

    for _, show in shows.iterrows():
        data = HeteroData()

        # --- add show nodes ---
        active_shows.append(show)
        data["show"].x = build_show_features(active_shows)

        # --- add person nodes ---
        data["person"].x = build_person_features(
            people, credits, show["opening_date"]
        )

        # --- add theater nodes ---
        data["theater"].x = build_theater_features(theaters)

        # --- edges ---
        add_person_show_edges(data, credits, show["opening_date"])
        add_show_theater_edges(data, theaters)

        # --- label ---
        data["show"].y = build_labels(active_shows)

        snapshots.append(data)

    torch.save(snapshots, "data/processed/graph_snapshots.pt")
