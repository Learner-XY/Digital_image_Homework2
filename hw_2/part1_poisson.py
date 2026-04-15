from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


Point = Tuple[int, int]


DEFAULT_POINTS_A1_TO_A2: List[Point] = [
    (69, 101),
    (103, 78),
    (167, 77),
    (201, 99),
    (194, 149),
    (179, 214),
    (132, 262),
    (93, 226),
    (72, 162),
]

DEFAULT_POINTS_A2_TO_A1: List[Point] = [
    (57, 95),
    (92, 74),
    (160, 77),
    (194, 104),
    (190, 154),
    (171, 217),
    (124, 264),
    (84, 229),
    (61, 167),
]


def parse_points(raw_points: str | None, fallback: Sequence[Point]) -> List[Point]:
    if not raw_points:
        return list(fallback)
    points: List[Point] = []
    for chunk in raw_points.split(";"):
        x_str, y_str = chunk.split(",")
        points.append((int(x_str), int(y_str)))
    if len(points) < 3:
        raise ValueError("Polygon needs at least three points.")
    return points


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), bgr)


def create_mask_from_points(points: Sequence[Point], img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    polygon = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def overlay_polygon(image: np.ndarray, points: Sequence[Point], color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    polygon = np.asarray(points, dtype=np.int32)
    cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2)
    return canvas


def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin: int = 12) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty.")
    left = max(int(xs.min()) - margin, 0)
    right = min(int(xs.max()) + margin + 1, image.shape[1])
    top = max(int(ys.min()) - margin, 0)
    bottom = min(int(ys.max()) + margin + 1, image.shape[0])
    return image[top:bottom, left:right], mask[top:bottom, left:right], (left, top, right, bottom)


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy((mask.astype(np.float32) / 255.0)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def cal_laplacian_loss(
    foreground_img: torch.Tensor,
    foreground_mask: torch.Tensor,
    blended_img: torch.Tensor,
    background_mask: torch.Tensor,
) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=foreground_img.dtype,
        device=foreground_img.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(foreground_img.shape[1], 1, 1, 1)

    fg_laplacian = F.conv2d(foreground_img, kernel, padding=1, groups=foreground_img.shape[1])
    blended_laplacian = F.conv2d(blended_img, kernel, padding=1, groups=blended_img.shape[1])

    valid_mask = foreground_mask * background_mask
    denominator = valid_mask.sum().clamp_min(1.0) * foreground_img.shape[1]
    laplacian_term = ((fg_laplacian - blended_laplacian) ** 2) * valid_mask
    color_term = ((foreground_img - blended_img) ** 2) * valid_mask
    return laplacian_term.sum() / denominator + 0.05 * color_term.sum() / denominator


def poisson_blend(
    source_image: np.ndarray,
    target_image: np.ndarray,
    polygon_points: Sequence[Point],
    iterations: int = 1200,
    learning_rate: float = 1e-2,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_image.shape != target_image.shape:
        target_image = cv2.resize(
            target_image,
            (source_image.shape[1], source_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    full_mask = create_mask_from_points(polygon_points, source_image.shape[0], source_image.shape[1])
    source_crop, mask_crop, (left, top, right, bottom) = crop_to_mask(source_image, full_mask)
    target_crop = target_image[top:bottom, left:right]

    device_obj = torch.device(device)
    fg_tensor = image_to_tensor(source_crop, device_obj)
    bg_tensor = image_to_tensor(target_crop, device_obj)
    mask_tensor = mask_to_tensor(mask_crop, device_obj)

    blended = bg_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([blended], lr=learning_rate)

    for step in range(iterations):
        blended_for_loss = blended * mask_tensor + bg_tensor.detach() * (1.0 - mask_tensor)
        loss = cal_laplacian_loss(fg_tensor, mask_tensor, blended_for_loss, mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            blended.clamp_(0.0, 1.0)

        if step == iterations // 2:
            optimizer.param_groups[0]["lr"] *= 0.2

    result_crop = (
        blended.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )
    result_crop = (result_crop * 255.0).astype(np.uint8)

    naive = target_image.copy()
    naive[top:bottom, left:right][mask_crop > 0] = source_crop[mask_crop > 0]

    result = target_image.copy()
    result[top:bottom, left:right][mask_crop > 0] = result_crop[mask_crop > 0]
    return result, full_mask, naive


def run_case(
    source_path: Path,
    target_path: Path,
    points: Sequence[Point],
    output_dir: Path,
    prefix: str,
    iterations: int,
    learning_rate: float,
    device: str,
) -> None:
    source = load_image(source_path)
    target = load_image(target_path)
    blended, mask, naive = poisson_blend(
        source_image=source,
        target_image=target,
        polygon_points=points,
        iterations=iterations,
        learning_rate=learning_rate,
        device=device,
    )

    save_image(output_dir / f"{prefix}_source.png", source)
    save_image(output_dir / f"{prefix}_target.png", target)
    save_image(output_dir / f"{prefix}_source_polygon.png", overlay_polygon(source, points, (255, 0, 0)))
    save_image(output_dir / f"{prefix}_target_polygon.png", overlay_polygon(target, points, (0, 255, 0)))
    save_image(output_dir / f"{prefix}_mask.png", np.repeat(mask[:, :, None], 3, axis=2))
    save_image(output_dir / f"{prefix}_naive_clone.png", naive)
    save_image(output_dir / f"{prefix}_poisson.png", blended)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poisson image editing with PyTorch.")
    parser.add_argument("--source", type=Path, default=Path("base/a1.png"))
    parser.add_argument("--target", type=Path, default=Path("base/a2.png"))
    parser.add_argument("--points", type=str, default=None, help="Format: x1,y1;x2,y2;...")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part1"))
    parser.add_argument("--prefix", type=str, default="a1_to_a2")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--run-both-examples",
        action="store_true",
        help="Run a1->a2 and a2->a1 with built-in polygon points.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.run_both_examples:
        run_case(
            source_path=Path("base/a1.png"),
            target_path=Path("base/a2.png"),
            points=DEFAULT_POINTS_A1_TO_A2,
            output_dir=args.output_dir,
            prefix="a1_to_a2",
            iterations=args.iterations,
            learning_rate=args.lr,
            device=args.device,
        )
        run_case(
            source_path=Path("base/a2.png"),
            target_path=Path("base/a1.png"),
            points=DEFAULT_POINTS_A2_TO_A1,
            output_dir=args.output_dir,
            prefix="a2_to_a1",
            iterations=args.iterations,
            learning_rate=args.lr,
            device=args.device,
        )
        return

    points = parse_points(args.points, DEFAULT_POINTS_A1_TO_A2)
    run_case(
        source_path=args.source,
        target_path=args.target,
        points=points,
        output_dir=args.output_dir,
        prefix=args.prefix,
        iterations=args.iterations,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
