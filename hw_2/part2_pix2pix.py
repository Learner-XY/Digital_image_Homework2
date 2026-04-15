from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class PairRecord:
    stem: str
    input_path: Path
    target_path: Path


class CMPFacadeDataset(Dataset):
    def __init__(self, records: Sequence[PairRecord], image_size: int = 256, load_size: int = 286, augment: bool = False):
        self.records = list(records)
        self.image_size = image_size
        self.load_size = load_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        input_image = cv2.imread(str(record.input_path), cv2.IMREAD_COLOR)
        target_image = cv2.imread(str(record.target_path), cv2.IMREAD_COLOR)
        if input_image is None or target_image is None:
            raise FileNotFoundError(f"Failed to read pair: {record.stem}")

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        if self.augment:
            input_image = cv2.resize(input_image, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)
            target_image = cv2.resize(target_image, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA)

            max_offset = self.load_size - self.image_size
            offset_x = random.randint(0, max_offset)
            offset_y = random.randint(0, max_offset)
            input_image = input_image[offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size]
            target_image = target_image[offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size]

            if random.random() < 0.5:
                input_image = np.ascontiguousarray(input_image[:, ::-1])
                target_image = np.ascontiguousarray(target_image[:, ::-1])
        else:
            input_image = cv2.resize(input_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            target_image = cv2.resize(target_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        input_tensor = torch.from_numpy(input_image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "stem": record.stem,
        }


def discover_pairs(data_dir: Path) -> List[PairRecord]:
    records: List[PairRecord] = []
    for image_path in sorted(data_dir.glob("cmp_b*.png")):
        stem = image_path.stem
        target_path = data_dir / f"{stem}.jpg"
        if target_path.exists():
            records.append(PairRecord(stem=stem, input_path=image_path, target_path=target_path))
    if not records:
        raise FileNotFoundError(f"No paired cmp_b*.png/jpg samples found in {data_dir}")
    return records


def split_records(records: Sequence[PairRecord], train_ratio: float, seed: int) -> tuple[list[PairRecord], list[PairRecord]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_ratio))
    train_records = shuffled[:train_count]
    val_records = shuffled[train_count:]
    if not val_records:
        val_records = shuffled[-max(1, len(shuffled) // 10) :]
        train_records = shuffled[: len(shuffled) - len(val_records)]
    return train_records, val_records


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.down1 = DownBlock(in_channels, base_channels, normalize=False)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)
        self.down5 = DownBlock(base_channels * 8, base_channels * 8)

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, dropout=True)
        self.up2 = UpBlock(base_channels * 16, base_channels * 4, dropout=True)
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)
        self.up4 = UpBlock(base_channels * 4, base_channels)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bottleneck = self.down5(d4)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final(torch.cat([u4, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([source, target], dim=1))


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    return ((image.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


def save_tensor_image(image: torch.Tensor, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = denormalize(image.cpu()).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), bgr)


def save_triptych(source: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, image_path: Path) -> None:
    source_np = denormalize(source.cpu()).permute(1, 2, 0).numpy()
    prediction_np = denormalize(prediction.cpu()).permute(1, 2, 0).numpy()
    target_np = denormalize(target.cpu()).permute(1, 2, 0).numpy()
    merged = np.concatenate([source_np, prediction_np, target_np], axis=1)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))


def evaluate_model(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    l1_loss: nn.Module,
) -> dict:
    generator.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            source = batch["input"].to(device)
            target = batch["target"].to(device)
            prediction = generator(source)
            total_l1 += l1_loss(prediction, target).item()

            mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
            psnr = 10.0 * torch.log10(4.0 / mse.clamp_min(1e-8))
            total_psnr += psnr.mean().item()
            count += 1
    generator.train()
    return {
        "val_l1": total_l1 / max(count, 1),
        "val_psnr": total_psnr / max(count, 1),
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    predictions_dir = output_dir / "predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    records = discover_pairs(data_dir)
    train_records, val_records = split_records(records, train_ratio=args.train_ratio, seed=args.seed)
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples:
        val_records = val_records[: args.max_val_samples]

    train_dataset = CMPFacadeDataset(
        train_records,
        image_size=args.image_size,
        load_size=args.load_size,
        augment=not args.disable_augment,
    )
    val_dataset = CMPFacadeDataset(val_records, image_size=args.image_size, load_size=args.image_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    generator = UNetGenerator(base_channels=args.base_channels).to(device)
    discriminator = PatchDiscriminator(base_channels=args.base_channels).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lr_lambda=lambda epoch: 1.0
        if epoch < max(args.epochs // 2, 1)
        else max(0.0, 1.0 - (epoch - max(args.epochs // 2, 1)) / max(args.epochs - max(args.epochs // 2, 1), 1)),
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optimizer_d,
        lr_lambda=lambda epoch: 1.0
        if epoch < max(args.epochs // 2, 1)
        else max(0.0, 1.0 - (epoch - max(args.epochs // 2, 1)) / max(args.epochs - max(args.epochs // 2, 1), 1)),
    )

    history: List[dict] = []
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g = 0.0
        epoch_d = 0.0
        step_count = 0

        for batch in train_loader:
            source = batch["input"].to(device)
            target = batch["target"].to(device)

            fake = generator(source)

            optimizer_d.zero_grad()
            pred_real = discriminator(source, target)
            pred_fake = discriminator(source, fake.detach())
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            loss_d = 0.5 * (adv_loss(pred_real, real_labels) + adv_loss(pred_fake, fake_labels))
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            pred_fake_for_g = discriminator(source, fake)
            loss_g_adv = adv_loss(pred_fake_for_g, real_labels)
            loss_g_l1 = l1_loss(fake, target)
            loss_g = loss_g_adv + args.lambda_l1 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            step_count += 1

        metrics = evaluate_model(generator, val_loader, device, l1_loss)
        epoch_summary = {
            "epoch": epoch,
            "train_g_loss": epoch_g / max(step_count, 1),
            "train_d_loss": epoch_d / max(step_count, 1),
            **metrics,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        sample_batch = next(iter(val_loader))
        source = sample_batch["input"].to(device)
        target = sample_batch["target"].to(device)
        with torch.no_grad():
            prediction = generator(source)
        save_triptych(source[0], prediction[0], target[0], samples_dir / f"epoch_{epoch:03d}.png")

        if metrics["val_l1"] < best_val_l1:
            best_val_l1 = metrics["val_l1"]
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": vars(args),
                    "history": history,
                },
                checkpoints_dir / "best.pt",
            )

        scheduler_g.step()
        scheduler_d.step()

    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "args": vars(args),
            "history": history,
        },
        checkpoints_dir / "last.pt",
    )

    with torch.no_grad():
        generator.eval()
        for batch in val_loader:
            source = batch["input"].to(device)
            stems = batch["stem"]
            predictions = generator(source)
            for stem, prediction in zip(stems, predictions):
                save_tensor_image(prediction, predictions_dir / f"{stem}.png")

    with (output_dir / "train_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    generator = UNetGenerator(base_channels=checkpoint["args"].get("base_channels", 32)).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load {args.input}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
    input_tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = generator(input_tensor)[0]
    save_tensor_image(prediction, Path(args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pix2Pix training on CMP facades.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=str, default="base")
    train_parser.add_argument("--output-dir", type=str, default="outputs/part2")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--load-size", type=int, default=286)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--lambda-l1", type=float, default=100.0)
    train_parser.add_argument("--train-ratio", type=float, default=0.9)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-val-samples", type=int, default=None)
    train_parser.add_argument("--disable-augment", action="store_true")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--input", type=str, required=True)
    predict_parser.add_argument("--output", type=str, required=True)
    predict_parser.add_argument("--image-size", type=int, default=256)
    predict_parser.add_argument("--device", type=str, default="cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
