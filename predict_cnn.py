"""Run prediction with trained CNN weights.

This script is a public, path-independent version of the original
`CNN/CNN_predict.py` workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class ImprovedCNNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (input_size // 4), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def preprocess_future_table(path: Path) -> pd.DataFrame:
    future_data = pd.read_csv(path)
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    future_imputed = pd.DataFrame(imputer.fit_transform(future_data), columns=future_data.columns)
    future_normalized = pd.DataFrame(scaler.fit_transform(future_imputed), columns=future_data.columns)
    return future_normalized.iloc[:, 12:]


def predict(args: argparse.Namespace) -> None:
    X_future = preprocess_future_table(Path(args.future))
    weights_dir = Path(args.weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = pd.DataFrame()

    for target_idx in range(12):
        target_name = f"F{target_idx + 1:02}"
        model_path = weights_dir / f"{target_name}_model.pth"
        if not model_path.exists():
            continue

        model = ImprovedCNNClassifier(input_size=X_future.shape[1], num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        X_tensor = torch.tensor(X_future.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, target_predictions = torch.max(outputs, 1)
        predictions[target_name] = target_predictions.cpu().numpy()

    predictions.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future", required=True, help="Future/prediction input CSV, e.g. 2017.csv.")
    parser.add_argument("--weights", required=True, help="Directory containing F*_model.pth files.")
    parser.add_argument("--output", required=True, help="Output CSV for predictions.")
    parser.add_argument("--num-classes", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
