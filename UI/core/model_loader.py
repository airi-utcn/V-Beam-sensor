import os
import torch
import torch.nn as nn
import joblib

class ForwardNet(nn.Module):
    def __init__(self, in_dim=6, out_dim=3, hidden_sizes=(128, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Models")
)

MODEL_REGISTRY = {
    "MLP": {
        "model_file":  "mlp_forward_model.pt",
        "scaler_file": "mlp_scalers.pkl",
    },
    "MLP_FILTER": {
        "model_file":  "mlp_filtered_forward_model.pt",
        "scaler_file": "mlp_filtered_scalers.pkl",
    },
    "PINN": {
        "model_file":  "pinn_forward_model.pt",
        "scaler_file": "pinn_scalers.pkl",
    },
    "PINN_FILTER": {
        "model_file":  "pinn_filter_forward_model.pt",
        "scaler_file": "pinn_filter_scalers.pkl",
    },
}

# API
def load_model(model_type: str):
    cfg = MODEL_REGISTRY[model_type]
    model_path  = os.path.join(_MODELS_DIR, cfg["model_file"])
    scaler_path = os.path.join(_MODELS_DIR, cfg["scaler_file"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = checkpoint["best_params"]

    scalers  = joblib.load(scaler_path)
    scaler_x = scalers["scaler_x"]
    scaler_y = scalers["scaler_y"]

    model = ForwardNet(
        in_dim=6, out_dim=3,
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, scaler_x, scaler_y, device
