import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def build_graph_snapshots():
    shows = pd.read_csv("../data/raw/shows.csv", parse_dates=["opening_date"])
    people = pd.read_csv("../data/raw/people.csv")
    credits = pd.read_csv("../data/raw/credits.csv", parse_dates=["opening_date"])
    theaters = pd.read_csv("../data/raw/theaters.csv")

    shows = shows.sort_values("opening_date").reset_index(drop=True)
    snapshots = []

    for i, show in shows.iterrows():
        cutoff_date = show["opening_date"]

        # Active shows up to this opening
        active_shows = shows[shows["opening_date"] <= cutoff_date].reset_index(drop=True)

        data = HeteroData()

        # ---- node features ----
        data["show"].x = build_show_features(active_shows)
        data["person"].x = build_person_features(people, credits, cutoff_date)
        data["theater"].x = build_theater_features(theaters, active_shows, cutoff_date)

        # ---- edges ----
        add_person_show_edges(data, credits, active_shows, cutoff_date)
        add_show_theater_edges(data, active_shows)

        # ---- labels & mask ----
        y = build_labels(active_shows)

        mask = torch.zeros(len(active_shows), dtype=torch.bool)
        mask[-1] = True  # supervise only the newest show

        data["show"].y = y
        data["show"].train_mask = mask

        snapshots.append(data)

    torch.save(snapshots, "../data/processed/graph_snapshots.pt")


# ---------------- Node Features ---------------- #

def build_show_features(shows_df: pd.DataFrame) -> torch.Tensor:
    genre_dummies = pd.get_dummies(shows_df["genre"])
    is_revival = shows_df["is_revival"].astype(int).to_numpy()[:, None]
    previews = np.log1p(shows_df["num_previews"].fillna(0).to_numpy())[:, None]

    month = shows_df["opening_date"].dt.month.to_numpy()
    month_sin = np.sin(2 * np.pi * month / 12)[:, None]
    month_cos = np.cos(2 * np.pi * month / 12)[:, None]

    capacity = shows_df["theater_capacity"].to_numpy()[:, None]

    X = np.hstack([genre_dummies.to_numpy(), is_revival, previews, month_sin, month_cos, capacity])
    return torch.tensor(X, dtype=torch.float)


def build_person_features(people_df: pd.DataFrame, credits_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> torch.Tensor:
    prior_credits = credits_df[credits_df["opening_date"] < cutoff_date]

    total_credits = prior_credits.groupby("person_id").size().reindex(people_df["person_id"]).fillna(0).to_numpy()[:, None]

    role_counts = []
    for role in ["actor", "producer"]:
        role_counts.append(
            prior_credits[prior_credits["role"] == role]
            .groupby("person_id").size()
            .reindex(people_df["person_id"]).fillna(0).to_numpy()[:, None]
        )

    last_credit = prior_credits.groupby("person_id")["opening_date"].max().reindex(people_df["person_id"])
    recency_days = ((cutoff_date - last_credit).dt.days.fillna(3650)).to_numpy()[:, None]
    recency = np.log1p(recency_days)

    X = np.hstack([total_credits, *role_counts, recency])
    return torch.tensor(X, dtype=torch.float)


def build_theater_features(theaters_df: pd.DataFrame, shows_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> torch.Tensor:
    prior_shows = shows_df[shows_df["opening_date"] < cutoff_date]

    historical_count = prior_shows.groupby("theater_id").size().reindex(theaters_df["theater_id"]).fillna(0).to_numpy()[:, None]
    capacity = theaters_df["capacity"].to_numpy()[:, None]

    X = np.hstack([capacity, np.log1p(historical_count)])
    return torch.tensor(X, dtype=torch.float)


# ---------------- Edges ---------------- #

def add_person_show_edges(data: HeteroData, credits_df: pd.DataFrame, active_shows: pd.DataFrame, cutoff_date: pd.Timestamp):
    role_map = {"actor": "acted_in", "producer": "produced"}
    show_id_to_idx = {sid: i for i, sid in enumerate(active_shows["show_id"])}

    valid = credits_df[(credits_df["opening_date"] < cutoff_date) & (credits_df["show_id"].isin(show_id_to_idx))]

    for role, edge_name in role_map.items():
        subset = valid[valid["role"] == role]
        if subset.empty:
            continue

        src = torch.tensor(subset["person_idx"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(subset["show_id"].map(show_id_to_idx).to_numpy(), dtype=torch.long)

        data["person", edge_name, "show"].edge_index = torch.stack([src, dst])


def add_show_theater_edges(data: HeteroData, active_shows: pd.DataFrame):
    src = torch.arange(len(active_shows), dtype=torch.long)
    dst = torch.tensor(active_shows["theater_idx"].to_numpy(), dtype=torch.long)
    data["show", "at", "theater"].edge_index = torch.stack([src, dst])


# ---------------- Labels ---------------- #

def build_labels(shows_df: pd.DataFrame) -> torch.Tensor:
    # Assumes a precomputed binary column 'success'
    return torch.tensor(shows_df["success"].astype(int).to_numpy(), dtype=torch.long)
