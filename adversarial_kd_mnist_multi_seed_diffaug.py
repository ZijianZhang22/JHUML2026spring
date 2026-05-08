"""MNIST adversarial robustness framework.

This version keeps the original project pipeline, but changes the distillation
part to match the notebook-style defensive distillation:

1. Train the teacher with high-temperature cross entropy: CE(logits / T, y).
2. Precompute teacher soft labels with softmax(teacher_logits / T).
3. Train the student from hard labels + the precomputed soft labels.
4. --kd-alpha follows the notebook convention: alpha weights hard-label CE,
   and (1 - alpha) weights the soft-label distillation loss.

Baseline, adversarial training, FGSM/PGD evaluation, CSV saving, checkpointing,
and plotting are kept compatible with the original script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.6]


@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    run_name: str | None = None
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    num_runs: int = 3
    seeds: str | None = None
    temperature: float = 20.0
    kd_alpha: float = 0.5  # notebook alpha: alpha * hard CE + (1 - alpha) * soft KD
    adv_weight: float = 0.5
    pgd_steps: int = 20
    pgd_step_size: float = 0.01
    adv_train_epsilon: float = 0.1
    quick_test: bool = False
    enable_diffusion_aug: bool = False
    diffusion_epochs: int = 5
    synthetic_per_class: int = 500
    num_diffusion_steps: int = 200
    diffusion_lr: float = 2e-4
    synthetic_cache_path: str | None = None


class SmallCNN(nn.Module):
    """Compact CNN suitable for MNIST and fast experiments."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MNIST adversarial KD framework")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated train/eval runs when --seeds is not given")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 42,43,44. Overrides --num-runs")
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-step-size", type=float, default=0.01)
    parser.add_argument("--adv-train-epsilon", type=float, default=0.1)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument(
        "--enable-diffusion-aug",
        action="store_true",
        help="Enable notebook-style diffusion augmentation and train baseline_diffaug, adv_training_diffaug, and adv_kd_diffaug",
    )
    parser.add_argument("--diffusion-epochs", type=int, default=5)
    parser.add_argument("--synthetic-per-class", type=int, default=500)
    parser.add_argument("--num-diffusion-steps", type=int, default=200)
    parser.add_argument("--diffusion-lr", type=float, default=2e-4)
    parser.add_argument(
        "--synthetic-cache-path",
        default=None,
        help="Optional path to a saved synthetic_mnist.pt with {'images': tensor, 'labels': tensor}. If given, diffusion training/generation is skipped.",
    )
    return Config(**vars(parser.parse_args()))


def make_run_dir(config: Config) -> Path:
    """Create a fresh output directory for every experiment run."""

    root = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.run_name:
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in config.run_name)
        run_dir = root / f"{safe_name}_{timestamp}"
    else:
        run_dir = root / f"run_{timestamp}"

    suffix = 1
    candidate = run_dir
    while candidate.exists():
        candidate = Path(f"{run_dir}_{suffix}")
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)

    if config.quick_test:
        train_set = Subset(train_set, range(2048))
        test_set = Subset(test_set, range(1024))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader




# ============================================================
# Notebook-style diffusion augmentation utilities
# ============================================================


def require_diffusers():
    """Import diffusers only when diffusion augmentation is requested."""
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise ImportError(
            "Diffusion augmentation requires extra packages. Install them with:\n"
            "  pip install diffusers accelerate\n"
            "Then rerun with --enable-diffusion-aug."
        ) from exc
    return DDPMScheduler, UNet2DModel


def build_conditional_diffusion_model(config: Config, device: torch.device):
    """Create the same class-conditional DDPM components used in the notebook."""
    DDPMScheduler, UNet2DModel = require_diffusers()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_steps,
        beta_schedule="linear",
    )

    diffusion_model = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        num_class_embeds=10,
    ).to(device)

    return diffusion_model, noise_scheduler


def move_scheduler_to_device(noise_scheduler, device: torch.device) -> None:
    """Move scheduler tensors to the model device for Colab/GPU compatibility."""
    if hasattr(noise_scheduler, "timesteps"):
        noise_scheduler.timesteps = noise_scheduler.timesteps.to(device)
    for attr in ["betas", "alphas", "alphas_cumprod"]:
        if hasattr(noise_scheduler, attr):
            value = getattr(noise_scheduler, attr)
            if torch.is_tensor(value):
                setattr(noise_scheduler, attr, value.to(device))


def train_diffusion_model(
    diffusion_model: nn.Module,
    noise_scheduler,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> list[float]:
    """Train conditional DDPM to predict added noise, matching the notebook."""
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in tqdm(train_loader, desc=f"Diffusion epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            # MNIST loader gives images in [0, 1]; DDPM trains on [-1, 1].
            clean_images = x * 2.0 - 1.0
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = diffusion_model(noisy_images, timesteps, class_labels=y).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_count += batch_size

        avg_loss = total_loss / total_count
        losses.append(avg_loss)
        print(f"diffusion: epoch {epoch}/{epochs}, loss={avg_loss:.4f}")

    return losses


@torch.no_grad()
def generate_synthetic_mnist(
    diffusion_model: nn.Module,
    noise_scheduler,
    device: torch.device,
    num_per_class: int = 500,
    batch_size: int = 128,
    num_inference_steps: int = 200,
) -> tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    """Generate class-balanced synthetic MNIST samples from the trained DDPM."""
    if num_per_class <= 0:
        raise ValueError("--synthetic-per-class must be positive when diffusion augmentation is enabled")

    diffusion_model.eval()
    all_labels = torch.arange(10).repeat_interleave(num_per_class)
    all_images: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    noise_scheduler.set_timesteps(num_inference_steps)
    move_scheduler_to_device(noise_scheduler, device)

    for start in tqdm(range(0, len(all_labels), batch_size), desc="Generating synthetic MNIST", leave=False):
        labels = all_labels[start:start + batch_size].to(device)
        current_batch_size = labels.shape[0]

        images = torch.randn(current_batch_size, 1, 28, 28, device=device)

        for t in noise_scheduler.timesteps:
            noise_pred = diffusion_model(images, t, class_labels=labels).sample
            images = noise_scheduler.step(noise_pred, t, images).prev_sample

        # Convert generated images from [-1, 1] back to [0, 1].
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)

        all_images.append(images.cpu())
        all_y.append(labels.cpu())

    synthetic_images = torch.cat(all_images, dim=0).float()
    synthetic_labels = torch.cat(all_y, dim=0).long().view(-1)
    synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)
    return synthetic_dataset, synthetic_images, synthetic_labels


def dataset_to_tensors(dataset: Dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MNIST/Subset/TensorDataset to tensors, matching the notebook augmentation step."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for x, y in loader:
        images_list.append(x.cpu().float())
        labels_list.append(torch.as_tensor(y).cpu().long().view(-1))

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0).long().view(-1)
    return images, labels


def save_synthetic_samples_grid(synthetic_dataset: Dataset, output_dir: Path, n: int = 20) -> None:
    """Save a small grid of generated examples for sanity checking."""
    if len(synthetic_dataset) == 0:
        return

    n = min(n, len(synthetic_dataset))
    cols = min(10, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 1.2, rows * 1.4))
    for i in range(n):
        x, y = synthetic_dataset[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x.squeeze(), cmap="gray")
        plt.title(str(int(y)))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_samples.png", dpi=200)
    plt.close()


def load_cached_synthetic_dataset(path: str | Path) -> tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    """Load notebook-style synthetic_mnist.pt: {'images': tensor, 'labels': tensor}."""
    saved = torch.load(path, map_location="cpu")
    if not isinstance(saved, dict) or "images" not in saved or "labels" not in saved:
        raise ValueError("Synthetic cache must be a dict with keys 'images' and 'labels'")

    images = saved["images"].cpu().float()
    labels = saved["labels"].cpu().long().view(-1)
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(f"Synthetic images must have shape [N, 1, 28, 28], got {tuple(images.shape)}")
    if len(images) != len(labels):
        raise ValueError("Synthetic images and labels have different lengths")

    return TensorDataset(images, labels), images, labels


def build_diffusion_augmented_loader(
    real_train_loader: DataLoader,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> tuple[DataLoader, dict[str, list[float]]]:
    """Train/load diffusion samples and concatenate real MNIST + synthetic MNIST."""
    diffusion_losses: dict[str, list[float]] = {}

    if config.synthetic_cache_path:
        print(f"Loading cached synthetic MNIST from: {config.synthetic_cache_path}")
        synthetic_dataset, synthetic_images, synthetic_labels = load_cached_synthetic_dataset(config.synthetic_cache_path)
    else:
        print("\nTraining notebook-style class-conditional diffusion model")
        diffusion_model, noise_scheduler = build_conditional_diffusion_model(config, device)
        diffusion_optimizer = optim.AdamW(diffusion_model.parameters(), lr=config.diffusion_lr)
        diffusion_losses["diffusion"] = train_diffusion_model(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            train_loader=real_train_loader,
            optimizer=diffusion_optimizer,
            device=device,
            epochs=config.diffusion_epochs,
        )
        torch.save(diffusion_model.state_dict(), output_dir / "mnist_conditional_diffusion_model.pt")

        synthetic_dataset, synthetic_images, synthetic_labels = generate_synthetic_mnist(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            device=device,
            num_per_class=config.synthetic_per_class,
            batch_size=config.batch_size,
            num_inference_steps=config.num_diffusion_steps,
        )
        torch.save({"images": synthetic_images, "labels": synthetic_labels}, output_dir / "synthetic_mnist.pt")

    save_synthetic_samples_grid(synthetic_dataset, output_dir, n=20)

    real_images, real_labels = dataset_to_tensors(real_train_loader.dataset)
    synthetic_images_fixed = synthetic_images.cpu().float()
    synthetic_labels_fixed = synthetic_labels.cpu().long().view(-1)

    perm = torch.randperm(len(synthetic_images_fixed))
    synthetic_images_fixed = synthetic_images_fixed[perm]
    synthetic_labels_fixed = synthetic_labels_fixed[perm]

    augmented_images = torch.cat([real_images, synthetic_images_fixed], dim=0)
    augmented_labels = torch.cat([real_labels, synthetic_labels_fixed], dim=0).long().view(-1)
    augmented_train_dataset = TensorDataset(augmented_images, augmented_labels)

    augmented_train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Real train size: {len(real_images)}")
    print(f"Synthetic train size: {len(synthetic_images_fixed)}")
    print(f"Augmented train size: {len(augmented_train_dataset)}")

    return augmented_train_loader, diffusion_losses


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if epsilon == 0:
        return x.detach()

    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * grad.sign()
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    if epsilon == 0:
        return x.detach()

    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + step_size * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


def evaluate_under_attack(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, float], torch.Tensor],
    epsilons: list[float],
) -> dict[float, float]:
    model.eval()
    results = {}

    for epsilon in epsilons:
        correct = 0
        total = 0
        for x, y in tqdm(loader, desc=f"eval eps={epsilon}", leave=False):
            x = x.to(device)
            y = y.to(device)
            x_adv = attack_fn(model, x, y, epsilon)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        results[epsilon] = correct / total

    return results


def kd_loss_from_teacher_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KD loss used when teacher logits are computed online."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


# Keep this alias for compatibility with other project code that may import kd_loss.
def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return kd_loss_from_teacher_logits(student_logits, teacher_logits, temperature)


def distillation_loss_from_soft_labels(
    student_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Notebook-style loss: alpha * CE + (1 - alpha) * KL."""
    hard_loss = F.cross_entropy(student_logits, hard_targets)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        soft_targets,
        reduction="batchmean",
    ) * (temperature**2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    name: str,
) -> list[float]:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            count += y.size(0)
        losses.append(running / count)
        print(f"{name}: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_teacher_distillation_style(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Train teacher like the notebook: CE(logits / temperature, y)."""
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"teacher epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits / config.temperature, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"teacher: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


class DistillDataset(Dataset):
    """Wrap a dataset with precomputed teacher soft labels."""

    def __init__(self, base_dataset: Dataset, soft_labels: torch.Tensor) -> None:
        self.base_dataset = base_dataset
        self.soft_labels = soft_labels
        assert len(self.base_dataset) == len(self.soft_labels)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        soft_y = self.soft_labels[idx]
        return x, y, soft_y


@torch.no_grad()
def generate_soft_labels(
    teacher: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    """Generate notebook-style teacher soft labels with softmax(logits / T)."""
    teacher.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    soft_labels = []
    for x, _ in tqdm(loader, desc="generate soft labels", leave=False):
        x = x.to(device)
        logits = teacher(x)
        probs = F.softmax(logits / temperature, dim=1)
        soft_labels.append(probs.cpu())

    return torch.cat(soft_labels, dim=0)


def train_standard_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Notebook-style defensive distillation student.

    The important change is that teacher labels are precomputed in the fixed
    dataset order with shuffle=False, then attached to the dataset. This matches
    the notebook's DistillDataset + generate_soft_labels design.
    """
    soft_labels = generate_soft_labels(
        teacher=teacher,
        dataset=loader.dataset,
        batch_size=config.batch_size,
        device=device,
        temperature=config.temperature,
    )

    distill_dataset = DistillDataset(loader.dataset, soft_labels)
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = optim.Adam(student.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        student.train()
        running = 0.0
        count = 0
        for x, hard_y, soft_y in tqdm(distill_loader, desc=f"standard_kd epoch {epoch}", leave=False):
            x = x.to(device)
            hard_y = hard_y.to(device)
            soft_y = soft_y.to(device)

            optimizer.zero_grad()
            student_logits = student(x)
            loss = distillation_loss_from_soft_labels(
                student_logits=student_logits,
                hard_targets=hard_y,
                soft_targets=soft_y,
                temperature=config.temperature,
                alpha=config.kd_alpha,
            )
            loss.backward()
            optimizer.step()

            running += loss.item() * hard_y.size(0)
            count += hard_y.size(0)

        losses.append(running / count)
        print(f"standard_kd: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_adversarial(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Train on clean and adversarial images using only hard labels."""

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"adv_training epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            model.eval()
            x_adv = fgsm_attack(model, x, y, config.adv_train_epsilon)
            model.train()

            clean_logits = model(x)
            adv_logits = model(x_adv)

            clean_loss = F.cross_entropy(clean_logits, y)
            adv_loss = F.cross_entropy(adv_logits, y)
            loss = (1 - config.adv_weight) * clean_loss + config.adv_weight * adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"adv_training: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_adversarial_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Adversarial KD on clean and adversarial images.

    Kept for compatibility, but the alpha convention is made consistent with
    the notebook: kd_alpha weights hard CE and (1 - kd_alpha) weights KD.
    """

    optimizer = optim.Adam(student.parameters(), lr=config.lr)
    teacher.eval()
    losses = []

    for epoch in range(1, config.epochs + 1):
        student.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"adv_kd epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            student.eval()
            x_adv = fgsm_attack(student, x, y, config.adv_train_epsilon)
            student.train()

            with torch.no_grad():
                teacher_clean_logits = teacher(x)
                teacher_adv_logits = teacher(x_adv)

            clean_logits = student(x)
            adv_logits = student(x_adv)

            clean_hard = F.cross_entropy(clean_logits, y)
            clean_soft = kd_loss_from_teacher_logits(clean_logits, teacher_clean_logits, config.temperature)
            adv_hard = F.cross_entropy(adv_logits, y)
            adv_soft = kd_loss_from_teacher_logits(adv_logits, teacher_adv_logits, config.temperature)

            clean_loss = config.kd_alpha * clean_hard + (1.0 - config.kd_alpha) * clean_soft
            adv_loss = config.kd_alpha * adv_hard + (1.0 - config.kd_alpha) * adv_soft
            loss = (1 - config.adv_weight) * clean_loss + config.adv_weight * adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"adv_kd: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def plot_curves(
    results: dict[str, dict[str, dict[float, float]]],
    output_dir: Path,
) -> None:
    for attack_name in ["fgsm", "pgd"]:
        plt.figure(figsize=(7, 5))
        for model_name, model_results in results.items():
            attack_results = model_results[attack_name]
            xs = list(attack_results.keys())
            ys = [attack_results[e] for e in xs]
            plt.plot(xs, ys, marker="o", label=model_name)
        plt.xlabel("epsilon")
        plt.ylabel("accuracy")
        plt.title(f"MNIST accuracy under {attack_name.upper()} attack")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{attack_name}_accuracy_vs_epsilon.png", dpi=200)
        plt.close()


def save_training_losses_csv(
    losses_by_model: dict[str, list[float]],
    output_dir: Path,
) -> None:
    path = output_dir / "training_losses.csv"
    max_epochs = max(len(losses) for losses in losses_by_model.values())
    model_names = list(losses_by_model.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *model_names])
        for epoch_idx in range(max_epochs):
            row = [epoch_idx + 1]
            for model_name in model_names:
                losses = losses_by_model[model_name]
                row.append(losses[epoch_idx] if epoch_idx < len(losses) else "")
            writer.writerow(row)


def plot_training_losses(
    losses_by_model: dict[str, list[float]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    for model_name, losses in losses_by_model.items():
        xs = list(range(1, len(losses) + 1))
        plt.plot(xs, losses, marker="o", label=model_name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Training loss by model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curves.png", dpi=200)
    plt.close()


def save_results_csv(
    results: dict[str, dict[str, dict[float, float]]],
    output_dir: Path,
) -> None:
    path = output_dir / "attack_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "attack", "epsilon", "accuracy"])
        for model_name, model_results in results.items():
            for attack_name, attack_results in model_results.items():
                for epsilon, acc in attack_results.items():
                    writer.writerow([model_name, attack_name, epsilon, acc])


def visualize_adversarial_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    output_dir: Path,
) -> None:
    model.eval()
    x, y = next(iter(loader))
    x = x[:8].to(device)
    y = y[:8].to(device)
    x_adv = fgsm_attack(model, x, y, config.adv_train_epsilon)

    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        adv_pred = model(x_adv).argmax(dim=1)

    plt.figure(figsize=(10, 4))
    for i in range(x.size(0)):
        plt.subplot(2, x.size(0), i + 1)
        plt.imshow(x[i].detach().cpu().squeeze(), cmap="gray")
        plt.title(f"y={y[i].item()}\np={clean_pred[i].item()}")
        plt.axis("off")

        plt.subplot(2, x.size(0), x.size(0) + i + 1)
        plt.imshow(x_adv[i].detach().cpu().squeeze(), cmap="gray")
        plt.title(f"adv p={adv_pred[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "fgsm_adversarial_examples.png", dpi=200)
    plt.close()


def save_checkpoint(model: nn.Module, output_dir: Path, name: str) -> None:
    torch.save(model.state_dict(), output_dir / f"{name}.pt")



def parse_seed_list(config: Config) -> list[int]:
    """Return the exact seeds used for repeated experiments."""
    if config.seeds:
        seeds = [int(s.strip()) for s in config.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integer seeds were found")
        return seeds
    return [config.seed + i for i in range(config.num_runs)]


def write_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_attack_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["attack"]), float(row["epsilon"]))
        grouped.setdefault(key, []).append(float(row["accuracy"]))

    summary = []
    for (model, attack, epsilon), values in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        arr = np.array(values, dtype=float)
        summary.append(
            {
                "model": model,
                "attack": attack,
                "epsilon": epsilon,
                "mean_accuracy": float(arr.mean()),
                "std_accuracy": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "num_runs": int(len(arr)),
                "accuracies": ";".join(f"{v:.6f}" for v in values),
            }
        )
    return summary


def summarize_clean_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(float(row["clean_accuracy"]))

    summary = []
    for model, values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        summary.append(
            {
                "model": model,
                "mean_clean_accuracy": float(arr.mean()),
                "std_clean_accuracy": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "num_runs": int(len(arr)),
                "clean_accuracies": ";".join(f"{v:.6f}" for v in values),
            }
        )
    return summary


def plot_mean_curves(summary_rows: list[dict], output_dir: Path) -> None:
    """Plot mean accuracy over repeated runs, with std error bars."""
    for attack_name in ["fgsm", "pgd"]:
        plt.figure(figsize=(7, 5))
        models = sorted({row["model"] for row in summary_rows if row["attack"] == attack_name})
        for model_name in models:
            model_rows = [
                row for row in summary_rows
                if row["attack"] == attack_name and row["model"] == model_name
            ]
            model_rows = sorted(model_rows, key=lambda r: float(r["epsilon"]))
            xs = [float(row["epsilon"]) for row in model_rows]
            ys = [float(row["mean_accuracy"]) for row in model_rows]
            yerr = [float(row["std_accuracy"]) for row in model_rows]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=model_name)

        plt.xlabel("epsilon")
        plt.ylabel("mean accuracy over runs")
        plt.title(f"MNIST mean accuracy under {attack_name.upper()} attack")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{attack_name}_mean_accuracy_vs_epsilon.png", dpi=200)
        plt.close()


def plot_clean_bar(summary_rows: list[dict], output_dir: Path) -> None:
    models = [row["model"] for row in summary_rows]
    means = [float(row["mean_clean_accuracy"]) for row in summary_rows]
    stds = [float(row["std_clean_accuracy"]) for row in summary_rows]

    plt.figure(figsize=(7, 5))
    plt.bar(models, means, yerr=stds, capsize=4)
    plt.ylabel("mean clean accuracy over runs")
    plt.ylim(0, 1.05)
    plt.title("MNIST clean accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "clean_mean_accuracy.png", dpi=200)
    plt.close()


def run_one_seed(config: Config, seed_output_dir: Path, run_index: int) -> tuple[list[dict], list[dict]]:
    """Train/evaluate all model categories once for one seed."""
    set_seed(config.seed)
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    with (seed_output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    device = get_device()
    print(f"\n========== Run {run_index + 1}, seed={config.seed}, device={device} ==========")
    print(f"Saving this seed run to: {seed_output_dir.resolve()}")
    train_loader, test_loader = get_loaders(config)

    teacher = SmallCNN(dropout=0.25).to(device)
    baseline = SmallCNN(dropout=0.25).to(device)
    standard_kd = SmallCNN(dropout=0.25).to(device)
    adv_training = SmallCNN(dropout=0.25).to(device)
    adv_kd = SmallCNN(dropout=0.25).to(device)

    losses_by_model: dict[str, list[float]] = {}
    models: dict[str, nn.Module] = {}

    print("\nTraining teacher")
    losses_by_model["teacher"] = train_teacher_distillation_style(teacher, train_loader, device, config)
    save_checkpoint(teacher, seed_output_dir, "teacher")

    print("\nTraining baseline")
    losses_by_model["baseline"] = train_supervised(baseline, train_loader, device, config, name="baseline")
    save_checkpoint(baseline, seed_output_dir, "baseline")
    models["baseline"] = baseline

    print("\nTraining standard KD student")
    losses_by_model["standard_kd"] = train_standard_kd(standard_kd, teacher, train_loader, device, config)
    save_checkpoint(standard_kd, seed_output_dir, "standard_kd")
    models["standard_kd"] = standard_kd

    print("\nTraining adversarial training baseline")
    losses_by_model["adv_training"] = train_adversarial(adv_training, train_loader, device, config)
    save_checkpoint(adv_training, seed_output_dir, "adv_training")
    models["adv_training"] = adv_training

    print("\nTraining adversarial KD student")
    losses_by_model["adv_kd"] = train_adversarial_kd(adv_kd, teacher, train_loader, device, config)
    save_checkpoint(adv_kd, seed_output_dir, "adv_kd")
    models["adv_kd"] = adv_kd

    if config.enable_diffusion_aug:
        print("\n========== Building diffusion-augmented train loader ==========")
        augmented_train_loader, diffusion_losses = build_diffusion_augmented_loader(
            real_train_loader=train_loader,
            config=config,
            device=device,
            output_dir=seed_output_dir,
        )
        losses_by_model.update(diffusion_losses)

        teacher_diffaug = SmallCNN(dropout=0.25).to(device)
        baseline_diffaug = SmallCNN(dropout=0.25).to(device)
        adv_training_diffaug = SmallCNN(dropout=0.25).to(device)
        adv_kd_diffaug = SmallCNN(dropout=0.25).to(device)

        # Match the notebook: teacher_diffaug is trained supervised on augmented data.
        print("\nTraining teacher_diffaug on augmented data")
        losses_by_model["teacher_diffaug"] = train_supervised(
            teacher_diffaug,
            augmented_train_loader,
            device,
            config,
            name="teacher_diffaug",
        )
        save_checkpoint(teacher_diffaug, seed_output_dir, "teacher_diffaug")

        print("\nTraining baseline_diffaug on augmented data")
        losses_by_model["baseline_diffaug"] = train_supervised(
            baseline_diffaug,
            augmented_train_loader,
            device,
            config,
            name="baseline_diffaug",
        )
        save_checkpoint(baseline_diffaug, seed_output_dir, "baseline_diffaug")
        models["baseline_diffaug"] = baseline_diffaug

        print("\nTraining adv_training_diffaug on augmented data")
        losses_by_model["adv_training_diffaug"] = train_adversarial(
            adv_training_diffaug,
            augmented_train_loader,
            device,
            config,
        )
        save_checkpoint(adv_training_diffaug, seed_output_dir, "adv_training_diffaug")
        models["adv_training_diffaug"] = adv_training_diffaug

        print("\nTraining adv_kd_diffaug on augmented data")
        losses_by_model["adv_kd_diffaug"] = train_adversarial_kd(
            adv_kd_diffaug,
            teacher_diffaug,
            augmented_train_loader,
            device,
            config,
        )
        save_checkpoint(adv_kd_diffaug, seed_output_dir, "adv_kd_diffaug")
        models["adv_kd_diffaug"] = adv_kd_diffaug

    save_training_losses_csv(losses_by_model, seed_output_dir)
    plot_training_losses(losses_by_model, seed_output_dir)

    pgd_fn = lambda m, x, y, eps: pgd_attack(
        m,
        x,
        y,
        epsilon=eps,
        steps=config.pgd_steps,
        step_size=config.pgd_step_size,
    )

    per_seed_nested_results: dict[str, dict[str, dict[float, float]]] = {}
    raw_attack_rows: list[dict] = []
    clean_rows: list[dict] = []

    for model_name, model in models.items():
        clean_acc = accuracy(model, test_loader, device)
        print(f"{model_name}: clean accuracy={clean_acc:.4f}")
        clean_rows.append(
            {
                "run_index": run_index + 1,
                "seed": config.seed,
                "model": model_name,
                "clean_accuracy": clean_acc,
            }
        )

        fgsm_results = evaluate_under_attack(model, test_loader, device, fgsm_attack, EPSILONS)
        pgd_results = evaluate_under_attack(model, test_loader, device, pgd_fn, EPSILONS)
        per_seed_nested_results[model_name] = {"fgsm": fgsm_results, "pgd": pgd_results}

        for attack_name, attack_results in [("fgsm", fgsm_results), ("pgd", pgd_results)]:
            for epsilon, acc in attack_results.items():
                raw_attack_rows.append(
                    {
                        "run_index": run_index + 1,
                        "seed": config.seed,
                        "model": model_name,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        "accuracy": acc,
                    }
                )

    save_results_csv(per_seed_nested_results, seed_output_dir)
    plot_curves(per_seed_nested_results, seed_output_dir)
    visualize_adversarial_examples(baseline, test_loader, device, config, seed_output_dir)

    write_dict_rows(
        seed_output_dir / "clean_results.csv",
        clean_rows,
        ["run_index", "seed", "model", "clean_accuracy"],
    )

    return raw_attack_rows, clean_rows


def main() -> None:
    config = parse_args()
    seeds = parse_seed_list(config)
    output_dir = make_run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**asdict(config), "resolved_seeds": seeds}, f, indent=2)

    all_attack_rows: list[dict] = []
    all_clean_rows: list[dict] = []

    print(f"Running repeated experiment with seeds: {seeds}")
    print(f"Root output directory: {output_dir.resolve()}")

    for run_index, seed in enumerate(seeds):
        run_config = replace(config, seed=seed)
        seed_output_dir = output_dir / f"seed_{seed}"
        attack_rows, clean_rows = run_one_seed(run_config, seed_output_dir, run_index)
        all_attack_rows.extend(attack_rows)
        all_clean_rows.extend(clean_rows)

    write_dict_rows(
        output_dir / "all_runs_attack_results.csv",
        all_attack_rows,
        ["run_index", "seed", "model", "attack", "epsilon", "accuracy"],
    )
    write_dict_rows(
        output_dir / "all_runs_clean_results.csv",
        all_clean_rows,
        ["run_index", "seed", "model", "clean_accuracy"],
    )

    attack_summary = summarize_attack_rows(all_attack_rows)
    clean_summary = summarize_clean_rows(all_clean_rows)

    write_dict_rows(
        output_dir / "attack_results_summary.csv",
        attack_summary,
        ["model", "attack", "epsilon", "mean_accuracy", "std_accuracy", "num_runs", "accuracies"],
    )
    write_dict_rows(
        output_dir / "clean_results_summary.csv",
        clean_summary,
        ["model", "mean_clean_accuracy", "std_clean_accuracy", "num_runs", "clean_accuracies"],
    )

    plot_mean_curves(attack_summary, output_dir)
    plot_clean_bar(clean_summary, output_dir)

    print(f"\nDone. Raw and mean results saved to: {output_dir.resolve()}")
    print("Main files:")
    print(f"  - {output_dir / 'all_runs_attack_results.csv'}")
    print(f"  - {output_dir / 'attack_results_summary.csv'}")
    print(f"  - {output_dir / 'fgsm_mean_accuracy_vs_epsilon.png'}")
    print(f"  - {output_dir / 'pgd_mean_accuracy_vs_epsilon.png'}")
    print(f"  - {output_dir / 'clean_mean_accuracy.png'}")


if __name__ == "__main__":
    main()
