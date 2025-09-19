#!/usr/bin/env python
"""
main.py
-------

Entry point for training and evaluating the CIFAR-10 image classifier.  This script
loads the dataset, builds the model, and runs a configurable training loop.

Example usage:

    python main.py --epochs 10 --batch-size 64 --learning-rate 0.001 --save-model models/cifar10_cnn.pt

"""
import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CIFAR10CNN


def train_one_epoch(model: torch.nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module,
                    device: torch.device) -> tuple[float, float]:
    """Perform one epoch of training.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             criterion: torch.nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    loss = running_loss / total
    acc = correct / total
    return loss, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on the CIFAR-10 dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data to use for validation")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained model (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Load CIFAR-10
    trainval_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Split into training and validation
    val_size = int(len(trainval_dataset) * args.val_split)
    train_size = len(trainval_dataset) - val_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} — "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Save model if requested
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
