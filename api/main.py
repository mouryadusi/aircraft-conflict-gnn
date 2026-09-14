import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.nn import GCNConv


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/conflict_gcn_large.pth"

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class ConflictGCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.gcn1 = GCNConv(
            in_channels=6,
            out_channels=64,
        )

        self.gcn2 = GCNConv(
            in_channels=64,
            out_channels=64,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(64 * 2 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):

        x = F.relu(
            self.gcn1(
                data.x,
                data.edge_index,
            )
        )

        x = self.gcn2(
            x,
            data.edge_index,
        )

        src = x[data.edge_index[0]]
        dst = x[data.edge_index[1]]

        edge_features = torch.cat(
            [
                src,
                dst,
                data.edge_attr,
            ],
            dim=1,
        )

        logits = self.edge_mlp(edge_features)

        return logits.squeeze(-1)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = ConflictGCN().to(device)

state = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(state)
model.eval()


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Aircraft Conflict Prediction API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class PredictionRequest(BaseModel):

    x: list[list[float]]

    edge_index: list[list[int]]

    edge_attr: list[list[float]]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Aircraft Conflict Prediction API",
        "device": str(device),
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    x = torch.tensor(
        request.x,
        dtype=torch.float32,
        device=device,
    )

    edge_index = torch.tensor(
        request.edge_index,
        dtype=torch.long,
        device=device,
    )

    edge_attr = torch.tensor(
        request.edge_attr,
        dtype=torch.float32,
        device=device,
    )

    # Basic validation
    if x.ndim != 2 or x.shape[1] != 6:
        return {
            "error": "x must have shape [num_nodes, 6]"
        }

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return {
            "error": "edge_index must have shape [2, num_edges]"
        }

    if edge_attr.ndim != 2 or edge_attr.shape[1] != 6:
        return {
            "error": "edge_attr must have shape [num_edges, 6]"
        }

    if edge_index.shape[1] != edge_attr.shape[0]:
        return {
            "error": "Number of edges in edge_index and edge_attr must match"
        }

    # Create PyG-style data object
    from torch_geometric.data import Data

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Predict
    with torch.no_grad():

        logits = model(data)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).long()

    return {
        "predictions": predictions.cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
        "num_edges": int(edge_attr.shape[0]),
        "threshold": 0.5,
    }

