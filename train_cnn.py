"""Train the CNN models used in the manuscript.

This script is a public, path-independent version of the original `CNN/V1.py`
workflow. It trains one CNN classifier for each target column (`F01` to `F12`)
in a processed basin-level table.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


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


def discretize_target(column: pd.Series) -> pd.Series:
    labels = column.copy()
    zero_mask = labels == 0
    non_zero = labels[~zero_mask]
    if len(non_zero) == 0:
        return labels.astype(int)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(non_zero.values.reshape(-1, 1))
    sorted_clusters = [cls for cls, _ in Counter(clusters).most_common()]
    mapping = {cls: idx + 1 for idx, cls in enumerate(sorted_clusters)}
    labels.loc[~zero_mask] = np.vectorize(mapping.get)(clusters)
    return labels.astype(int)


def preprocess_table(data: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    scaler = MinMaxScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)
    for column in target_columns:
        data_normalized[column] = discretize_target(data_normalized[column])
    return data_normalized


def train_model(model, criterion, optimizer, train_loader, device) -> float:
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(test_loader), all_preds, all_labels


def plot_loss_curve(train_losses, test_losses, output_dir: Path, target_name: str) -> None:
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve for {target_name}")
    plt.legend()
    plt.savefig(output_dir / f"loss_curve_{target_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(confusion, output_dir: Path, target_name: str) -> None:
    plt.figure()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {target_name}")
    plt.savefig(output_dir / f"confusion_matrix_{target_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar_chart(report: dict, output_dir: Path, target_name: str) -> None:
    categories = list(report.keys())[:-3]
    values = [report[c]["f1-score"] for c in categories]
    values.append(report["weighted avg"]["f1-score"])
    labels = categories + ["Weighted Avg"]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.fill(angles, values, color="blue", alpha=0.25)
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"Radar Chart for {target_name}")
    plt.savefig(output_dir / f"radar_chart_{target_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_weights(model, output_dir: Path, target_name: str) -> None:
    torch.save(model.state_dict(), output_dir / f"{target_name}_model.pth")


def train_targets(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    target_columns = args.targets or [f"F{i:02}" for i in range(1, 13)]
    data_normalized = preprocess_table(data, target_columns)
    X = data_normalized.iloc[:, 12:]
    Y = data_normalized[target_columns]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []

    for target_name in Y.columns:
        y = Y[target_name]
        if y.nunique() <= 1:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        X_train, y_train = SMOTE(random_state=args.random_state).fit_resample(X_train, y_train)

        train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

        model = ImprovedCNNClassifier(input_size=X_train.shape[1], num_classes=y.nunique()).to(device)
        if args.init_weights:
            init_path = Path(args.init_weights) / f"{target_name}_model.pth"
            if init_path.exists():
                model.load_state_dict(torch.load(init_path, map_location=device))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        train_losses, test_losses = [], []

        for epoch in range(args.epochs):
            train_loss = train_model(model, criterion, optimizer, train_loader, device)
            test_loss, all_preds, all_labels = evaluate_model(model, criterion, test_loader, device)
            scheduler.step(test_loss)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

        save_model_weights(model, output_dir, target_name)
        report = classification_report(all_labels, all_preds, zero_division=1, output_dict=True)
        confusion = confusion_matrix(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)

        pd.DataFrame(report).transpose().to_csv(output_dir / f"classification_report_{target_name}.csv")
        pd.DataFrame(confusion).to_csv(output_dir / f"confusion_matrix_{target_name}.csv", index=False)
        plot_loss_curve(train_losses, test_losses, output_dir, target_name)
        plot_confusion_matrix(confusion, output_dir, target_name)
        plot_radar_chart(report, output_dir, target_name)
        summary_rows.append({"leadtime": args.leadtime, "target": target_name, "accuracy": accuracy})

    pd.DataFrame(summary_rows).to_csv(output_dir / f"performance_metrics_leadtime_{args.leadtime}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Processed basin CSV table, e.g. basin1.csv.")
    parser.add_argument("--leadtime", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--output", required=True, help="Directory for weights, figures, and metrics.")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional target subset, e.g. F01 F02.")
    parser.add_argument("--init-weights", default=None, help="Optional directory containing existing F*_model.pth files.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    train_targets(parse_args())
