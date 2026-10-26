import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore
from torch.utils.data import DataLoader, ConcatDataset

from model import BandWiseHyperbolicTransformer, BandContrastiveLoss
from data_loader import split_subject_data, load_DE_SEED, eegDataset
from domain_discriminator import UnifiedDomainDiscriminator


def set_requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


def load_data(data_dir):
    files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and not f.startswith(".")
    )

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    data_all, label_all = [], []

    for filename in files:
        data, label = load_DE_SEED(os.path.join(data_dir, filename))
        data_all.append(zscore(data, axis=0))
        label_all.append(label.reshape(-1, 1))

    return np.stack(data_all), np.stack(label_all), files


def train_epoch(
    model, discriminator, loader,
    optimizer_model, optimizer_dis,
    criterion, domain_criterion, contrastive_loss_fn,
    device, epoch, num_epochs
):
    model.train()
    discriminator.train()

    total_loss = np.zeros(5, dtype=np.float64)
    correct = total = 0
    lambda_contrast = 0.5 * epoch / num_epochs

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        d = d.to(device).long().view(-1)

        source_mask = y != -1
        target_mask = ~source_mask

        # 1. Train domain discriminator
        set_requires_grad(model, False)
        set_requires_grad(discriminator, True)
        optimizer_dis.zero_grad()

        with torch.no_grad():
            _, fused, _, _ = model(x)
            features = fused.flatten(1)

        loss_domain = domain_criterion(discriminator(features), d)
        loss_domain.backward()
        optimizer_dis.step()

        # 2. Train emotion recognition model
        set_requires_grad(model, True)
        set_requires_grad(discriminator, False)
        optimizer_model.zero_grad()

        logits, fused, x1, x2 = model(x)
        features = fused.flatten(1)
        zero = logits.new_zeros(())

        loss_cls = zero
        loss_contrast = zero
        loss_pseudo = zero

        # Source-domain classification and contrastive learning
        if source_mask.any():
            loss_cls = criterion(logits[source_mask], y[source_mask])
            loss_contrast = contrastive_loss_fn(
                x1[source_mask], x2[source_mask], y[source_mask]
            )

        # Target-domain pseudo-labeling
        if target_mask.any():
            target_logits = logits[target_mask]
            probabilities = target_logits.detach().softmax(dim=1)
            confidence, pseudo_labels = probabilities.max(dim=1)
            confident_mask = confidence > 0.7

            if confident_mask.any():
                loss_pseudo = criterion(
                    target_logits[confident_mask],
                    pseudo_labels[confident_mask]
                )

        # Adversarial domain loss
        domain_logits = discriminator(features)
        loss_adv = domain_criterion(domain_logits, torch.zeros_like(d))

        loss = (
            loss_cls
            + 0.4 * loss_pseudo
            + lambda_contrast * loss_contrast
            + 0.6 * loss_adv
        )

        loss.backward()
        optimizer_model.step()

        if source_mask.any():
            predictions = logits[source_mask].argmax(dim=1)
            correct += (predictions == y[source_mask]).sum().item()
            total += source_mask.sum().item()

        total_loss += [
            loss.item(),
            loss_cls.item(),
            loss_adv.item(),
            loss_contrast.item(),
            loss_pseudo.item()
        ]

    total_loss /= max(len(loader), 1)
    train_acc = correct / max(total, 1)

    return total_loss, train_acc


def train_subject(data_all, label_all, subject_idx, args, device):
    data_s, label_s, data_t, label_t = split_subject_data(
        data_all, label_all, subject_idx
    )

    domain_s = np.zeros(len(label_s), dtype=np.int64)
    domain_t = np.ones(len(label_t), dtype=np.int64)

    source_dataset = eegDataset(
        data_s, label_s, domain_labels=domain_s
    )
    target_dataset = eegDataset(
        data_t, np.full_like(label_t, -1), domain_labels=domain_t
    )
    test_dataset = eegDataset(
        data_t, label_t, domain_labels=domain_t
    )

    train_loader = DataLoader(
        ConcatDataset([source_dataset, target_dataset]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = BandWiseHyperbolicTransformer().to(device)

    # Infer the fused feature dimension
    model.eval()
    with torch.no_grad():
        x, _, _ = next(iter(train_loader))
        _, fused, _, _ = model(x[:1].to(device))
        feature_dim = fused.flatten(1).size(1)
    model.train()

    discriminator = UnifiedDomainDiscriminator(
        feature_dim, alpha=1.0
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.CrossEntropyLoss()
    contrastive_loss_fn = BandContrastiveLoss(temperature=0.1)

    optimizer_model = optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=1e-2
    )
    optimizer_dis = optim.Adam(
        discriminator.parameters(), lr=1e-4, weight_decay=1e-3
    )

    scheduler_model = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, T_max=args.epochs
    )
    scheduler_dis = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_dis, T_max=args.epochs
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_path = os.path.join(
        args.ckpt_dir, f"fold{subject_idx}.pth"
    )
    best_acc = 0.0

    for epoch in range(args.epochs):
        stats, train_acc = train_epoch(
            model, discriminator, train_loader,
            optimizer_model, optimizer_dis,
            criterion, domain_criterion, contrastive_loss_fn,
            device, epoch, args.epochs
        )

        scheduler_model.step()
        scheduler_dis.step()

        # Test directly inside the training loop
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device)
                y = y.to(device).long().view(-1)

                logits, _, _, _ = model(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()

        test_acc = correct / max(total, 1)

        print(
            f"Subject {subject_idx} | "
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {stats[0]:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} "

        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved: {save_path}")

        model.train()

    print(f"Subject {subject_idx} best accuracy: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Cross-subject EEG emotion recognition"
    )

    # Dataset and training settings
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/eegdata"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.ckpt_dir}")

    data_all, label_all, files = load_data(args.data_dir)
    subject_acc = []

    for idx, filename in enumerate(files):
        print(
            f"\n[{idx + 1}/{len(files)}] "
            f"Target subject: {filename}"
        )

        acc = train_subject(
            data_all, label_all, idx, args, device
        )
        subject_acc.append(acc)

    subject_acc = np.asarray(subject_acc)

    print("\nSubject-wise accuracy:", np.round(subject_acc, 4))
    print(f"Mean Accuracy: {subject_acc.mean():.4f}")
    print(f"Std Accuracy: {subject_acc.std():.4f}")


if __name__ == "__main__":
    main()