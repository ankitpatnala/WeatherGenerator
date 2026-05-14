import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ── Model ─────────────────────────────────────────────────────────────────────

class AttentionRotation(nn.Module):
    def __init__(self, dim=2048, num_heads=8, target_cosine=0.7):
        super().__init__()
        self.alpha = target_cosine
        self.beta  = math.sqrt(1 - target_cosine ** 2)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (batch, n_spatial, 2048)
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        perp, _ = self.attn(x_norm, x_norm, x_norm)
        perp = perp - (perp * x_norm).sum(-1, keepdim=True) * x_norm
        perp = perp / (perp.norm(dim=-1, keepdim=True) + 1e-8)
        return self.alpha * x_norm + self.beta * perp


# ── Dataset ───────────────────────────────────────────────────────────────────

class EncoderLatentDataset(Dataset):
    """
    Consecutive pairs (z_t, z_{t+1}) from sorted encoder latent files.
    A fixed random set of n_spatial locations is sampled once at construction.
    """
    def __init__(self, files, n_spatial=1000, total_spatial=12288):
        self.files       = files
        self.spatial_idx = torch.randperm(total_spatial)[:n_spatial]

    def __len__(self):
        return len(self.files) - 1

    def __getitem__(self, idx):
        z      = self._load(self.files[idx])
        z_next = self._load(self.files[idx + 1])
        return z[self.spatial_idx], z_next[self.spatial_idx]   # (n_spatial, 2048) each

    @staticmethod
    def _load(path):
        t = torch.load(path, map_location="cpu")["latent"].float()
        if t.dim() == 2:
            t = t.unsqueeze(0)
        elif t.dim() > 3:
            t = t.view(-1, t.shape[-2], t.shape[-1])
        return t.squeeze(0)   # (12288, 2048)


# ── Loss ──────────────────────────────────────────────────────────────────────

def combined_loss(pred, target, z_input, target_cosine=0.7, lambda_mse=1.0, lambda_rot=1.0):
    """
    Three terms:
      1. cosine  – align pred direction with target z_{t+1}
      2. mse     – L2 between normalised pred and normalised target (sharper than cosine alone)
      3. rotation – enforce cosine(pred, z_input) ≈ target_cosine (rotation angle constraint)
    """
    pred_n   = pred    / (pred.norm(dim=-1,    keepdim=True) + 1e-8)
    target_n = target  / (target.norm(dim=-1,  keepdim=True) + 1e-8)
    z_norm   = z_input / (z_input.norm(dim=-1, keepdim=True) + 1e-8)

    cos_loss = 1.0 - (pred_n * target_n).sum(dim=-1).mean()
    mse_loss = (pred_n - target_n).pow(2).mean()
    rot_loss = ((pred_n * z_norm).sum(dim=-1).mean() - target_cosine).pow(2)

    return cos_loss + lambda_mse * mse_loss + lambda_rot * rot_loss


# ── Eval helper ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total = 0.0
    for z, z_next in loader:
        z, z_next = z.to(device), z_next.to(device)
        total += combined_loss(model(z), z_next, z,
                               target_cosine=cfg["target_cosine"],
                               lambda_mse=cfg["lambda_mse"],
                               lambda_rot=cfg["lambda_rot"]).item()
    return total / len(loader)


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    """Returns (local_rank, is_distributed). Works with torchrun or plain python."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, True
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return local_rank, False


def cleanup_ddp(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


# ── Main training routine ─────────────────────────────────────────────────────

def run(cfg):
    local_rank, is_distributed = setup_ddp()
    world_size = dist.get_world_size() if is_distributed else 1
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main    = local_rank == 0

    # ── Files & splits ────────────────────────────────────────────────────────
    all_files = sorted(
        glob.glob(cfg["latent_glob"]),
        key=lambda p: int(p.split("/")[-1].split(".")[0].split("_")[-1]),
    )
    n       = len(all_files)
    n_test  = max(1, int(0.1 * n))
    n_val   = max(1, int(0.1 * n))
    n_train = n - n_val - n_test

    train_files = all_files[:n_train]
    val_files   = all_files[n_train : n_train + n_val]
    test_files  = all_files[n_train + n_val :]

    if is_main:
        print(f"Files — train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = EncoderLatentDataset(train_files, cfg["n_spatial"], cfg["total_spatial"])
    val_ds   = EncoderLatentDataset(val_files,   cfg["n_spatial"], cfg["total_spatial"])
    test_ds  = EncoderLatentDataset(test_files,  cfg["n_spatial"], cfg["total_spatial"])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) \
                    if is_distributed else None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AttentionRotation(
        dim=cfg["dim"], num_heads=cfg["num_heads"], target_cosine=cfg["target_cosine"]
    ).to(device)
    model = DDP(model, device_ids=[local_rank]) if is_distributed else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    global_step   = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_n    = 0

        for z, z_next in train_loader:
            z, z_next = z.to(device), z_next.to(device)
            loss = combined_loss(model(z), z_next, z,
                                 target_cosine=cfg["target_cosine"],
                                 lambda_mse=cfg["lambda_mse"],
                                 lambda_rot=cfg["lambda_rot"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            global_step  += 1
            running_loss += loss.item()
            running_n    += 1

            if is_main and global_step % cfg["log_every"] == 0:
                print(f"step {global_step:05d}  epoch {epoch:03d}  "
                      f"loss {running_loss / running_n:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0
                running_n    = 0

        scheduler.step()

        if is_main:
            inner = model.module if is_distributed else model
            val_loss = evaluate(inner, val_loader, device, cfg)
            print(f"── epoch {epoch:03d} end  val {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(inner.state_dict(), cfg["checkpoint"])
                print(f"  → saved checkpoint (val {val_loss:.4f})")

    # ── Test ──────────────────────────────────────────────────────────────────
    if is_main:
        inner = model.module if is_distributed else model
        inner.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
        test_loss = evaluate(inner, test_loader, device, cfg)
        print(f"\nTest loss (best ckpt): {test_loss:.4f}")

    cleanup_ddp(is_distributed)


# ── Config ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = dict(
        latent_glob    = "/p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator/latents_2/*.pt",
        total_spatial  = 12288,
        n_spatial      = 12288//64,       # spatial tokens fed to attention per sample
        dim            = 2048,
        num_heads      = 8,
        target_cosine  = 0.7,
        batch_size     = 4,
        epochs         = 50,
        lr             = 1e-4,
        weight_decay   = 1e-2,
        grad_clip      = 1.0,
        lambda_mse     = 1.0,   # weight for L2 between normalised pred and target
        lambda_rot     = 1.0,   # weight for rotation angle constraint
        log_every      = 10,
        num_workers    = 4,
        checkpoint     = "attention_rotation_best.pt",
    )
    run(cfg)
