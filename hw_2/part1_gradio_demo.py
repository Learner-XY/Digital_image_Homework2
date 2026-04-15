from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np

from part1_poisson import load_image, poisson_blend


Point = Tuple[int, int]


def ensure_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def draw_points(image: np.ndarray | None, points: Sequence[Point], closed: bool) -> np.ndarray | None:
    image = ensure_rgb(image)
    if image is None:
        return None

    canvas = image.copy()
    for idx, (x, y) in enumerate(points):
        cv2.circle(canvas, (int(x), int(y)), 4, (255, 0, 0), -1)
        cv2.putText(canvas, str(idx + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if len(points) >= 2:
        poly = np.asarray(points, dtype=np.int32)
        cv2.polylines(canvas, [poly], isClosed=closed, color=(255, 0, 0), thickness=2)
    return canvas


def points_to_text(points: Sequence[Point], closed: bool) -> str:
    if not points:
        return "No points yet. Click on the foreground image to add polygon vertices."
    chunks = [f"{i + 1}. ({x}, {y})" for i, (x, y) in enumerate(points)]
    status = "Closed polygon" if closed else "Polygon is open"
    return status + "\n" + "\n".join(chunks)


def get_mask_and_crop(source_image: np.ndarray, points: Sequence[Point]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Polygon is empty.")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return mask, (left, top, right, bottom)


def compose_preview(
    foreground_image: np.ndarray | None,
    background_image: np.ndarray | None,
    points: Sequence[Point],
    closed: bool,
    offset_x: float,
    offset_y: float,
) -> np.ndarray | None:
    background_image = ensure_rgb(background_image)
    foreground_image = ensure_rgb(foreground_image)
    if background_image is None:
        return None

    preview = background_image.copy()
    if foreground_image is None or len(points) < 3:
        return preview

    try:
        mask, (left, top, right, bottom) = get_mask_and_crop(foreground_image, points)
    except ValueError:
        return preview

    crop = foreground_image[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]

    paste_x = int(np.clip(round(offset_x), 0, max(preview.shape[1] - crop.shape[1], 0)))
    paste_y = int(np.clip(round(offset_y), 0, max(preview.shape[0] - crop.shape[0], 0)))
    end_x = paste_x + crop.shape[1]
    end_y = paste_y + crop.shape[0]

    roi = preview[paste_y:end_y, paste_x:end_x].copy()
    alpha = (crop_mask.astype(np.float32) / 255.0)[:, :, None] * 0.55
    preview[paste_y:end_y, paste_x:end_x] = (crop * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    shifted_points = [(x - left + paste_x, y - top + paste_y) for x, y in points]
    if len(shifted_points) >= 2:
        cv2.polylines(
            preview,
            [np.asarray(shifted_points, dtype=np.int32)],
            isClosed=closed,
            color=(0, 255, 0),
            thickness=2,
        )
    return preview


def add_point(
    foreground_image: np.ndarray | None,
    points: List[Point],
    closed: bool,
    evt: gr.SelectData,
):
    foreground_image = ensure_rgb(foreground_image)
    if foreground_image is None:
        return None, points, closed, points_to_text(points, closed), gr.update(), gr.update()
    if closed:
        return (
            draw_points(foreground_image, points, closed),
            points,
            closed,
            "Polygon is already closed. Use Clear or Undo to modify it.",
            gr.update(),
            gr.update(),
        )

    index = getattr(evt, "index", None)
    if index is None or not isinstance(index, (tuple, list)) or len(index) < 2:
        return (
            draw_points(foreground_image, points, closed),
            points,
            closed,
            points_to_text(points, closed),
            gr.update(),
            gr.update(),
        )

    x, y = int(index[0]), int(index[1])
    points = list(points) + [(x, y)]
    max_x = max(foreground_image.shape[1] - 1, 0)
    max_y = max(foreground_image.shape[0] - 1, 0)
    return (
        draw_points(foreground_image, points, closed),
        points,
        closed,
        points_to_text(points, closed),
        gr.update(maximum=max_x),
        gr.update(maximum=max_y),
    )


def close_polygon(foreground_image: np.ndarray | None, points: List[Point]):
    foreground_image = ensure_rgb(foreground_image)
    closed = len(points) >= 3
    message = points_to_text(points, closed)
    if len(points) < 3:
        message = "Need at least 3 points before closing the polygon."
    return draw_points(foreground_image, points, closed), closed, message


def undo_point(foreground_image: np.ndarray | None, points: List[Point], closed: bool):
    foreground_image = ensure_rgb(foreground_image)
    if points:
        points = list(points[:-1])
    closed = False
    return draw_points(foreground_image, points, closed), points, closed, points_to_text(points, closed)


def clear_points(foreground_image: np.ndarray | None):
    foreground_image = ensure_rgb(foreground_image)
    points: List[Point] = []
    closed = False
    return foreground_image, points, closed, points_to_text(points, closed)


def on_foreground_upload(foreground_image: np.ndarray | None):
    foreground_image = ensure_rgb(foreground_image)
    if foreground_image is None:
        return None, [], False, points_to_text([], False), gr.update(maximum=1, value=0), gr.update(maximum=1, value=0)
    return (
        foreground_image,
        [],
        False,
        points_to_text([], False),
        gr.update(maximum=max(foreground_image.shape[1] - 1, 1), value=0),
        gr.update(maximum=max(foreground_image.shape[0] - 1, 1), value=0),
    )


def on_background_upload(background_image: np.ndarray | None):
    background_image = ensure_rgb(background_image)
    if background_image is None:
        return None
    return background_image


def update_background_preview(foreground_image, background_image, points, closed, offset_x, offset_y):
    return compose_preview(foreground_image, background_image, points, closed, offset_x, offset_y)


def run_blending(foreground_image, background_image, points, closed, offset_x, offset_y, iterations):
    foreground_image = ensure_rgb(foreground_image)
    background_image = ensure_rgb(background_image)

    if foreground_image is None or background_image is None:
        return None, None, "Please upload both foreground and background images first."
    if len(points) < 3 or not closed:
        return None, None, "Please finish the polygon before blending."

    mask, (left, top, right, bottom) = get_mask_and_crop(foreground_image, points)
    crop = foreground_image[top:bottom, left:right]
    crop_points = [(x - left, y - top) for x, y in points]
    crop_mask = mask[top:bottom, left:right]

    paste_x = int(np.clip(round(offset_x), 0, max(background_image.shape[1] - crop.shape[1], 0)))
    paste_y = int(np.clip(round(offset_y), 0, max(background_image.shape[0] - crop.shape[0], 0)))

    source_canvas = np.zeros_like(background_image)
    source_canvas[paste_y : paste_y + crop.shape[0], paste_x : paste_x + crop.shape[1]] = crop
    shifted_points = [(x + paste_x, y + paste_y) for x, y in crop_points]

    blended, _, naive = poisson_blend(
        source_image=source_canvas,
        target_image=background_image,
        polygon_points=shifted_points,
        iterations=int(iterations),
        learning_rate=1e-2,
        device="cpu",
    )

    return naive, blended, "Blending complete."


def load_examples():
    fg = load_image(Path("base/a1.png"))
    bg = load_image(Path("base/a2.png"))
    preview = fg.copy()
    return (
        fg,
        bg,
        preview,
        bg,
        [],
        False,
        points_to_text([], False),
        gr.update(maximum=max(fg.shape[1] - 1, 1), value=0),
        gr.update(maximum=max(fg.shape[0] - 1, 1), value=0),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Poisson Image Editing Demo") as demo:
        gr.Markdown(
            """
            # Poisson Image Editing Demo
            1. Upload a foreground and a background image.
            2. Click on the foreground image to add polygon vertices.
            3. Use `Close Polygon` after selecting the region.
            4. Move the selected region on the background with the sliders.
            5. Click `Blend` to compare naive copy-paste and Poisson blending.
            """
        )

        points_state = gr.State([])
        closed_state = gr.State(False)

        with gr.Row():
            foreground_input = gr.Image(label="Foreground Image", type="numpy", image_mode="RGB", interactive=True)
            background_input = gr.Image(label="Background Image", type="numpy", image_mode="RGB", interactive=True)

        with gr.Row():
            foreground_preview = gr.Image(label="Foreground With Polygon", type="numpy", image_mode="RGB", interactive=True)
            background_preview = gr.Image(label="Background Preview", type="numpy", image_mode="RGB", interactive=False)

        with gr.Row():
            offset_x = gr.Slider(label="Move X", minimum=0, maximum=1, step=1, value=0)
            offset_y = gr.Slider(label="Move Y", minimum=0, maximum=1, step=1, value=0)
            iterations = gr.Slider(label="Optimization Iterations", minimum=100, maximum=3000, step=100, value=1200)

        with gr.Row():
            load_button = gr.Button("Use Example Images")
            close_button = gr.Button("Close Polygon")
            undo_button = gr.Button("Undo Last Point")
            clear_button = gr.Button("Clear Points")
            blend_button = gr.Button("Blend")

        status_box = gr.Textbox(label="Status / Polygon Points", lines=10, interactive=False)

        with gr.Row():
            naive_output = gr.Image(label="Naive Copy-Paste", type="numpy", image_mode="RGB", interactive=False)
            blend_output = gr.Image(label="Poisson Blending Result", type="numpy", image_mode="RGB", interactive=False)

        foreground_input.upload(
            on_foreground_upload,
            inputs=[foreground_input],
            outputs=[foreground_preview, points_state, closed_state, status_box, offset_x, offset_y],
            api_name=False,
            show_api=False,
        )
        background_input.upload(
            on_background_upload,
            inputs=[background_input],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )

        load_button.click(
            load_examples,
            outputs=[
                foreground_input,
                background_input,
                foreground_preview,
                background_preview,
                points_state,
                closed_state,
                status_box,
                offset_x,
                offset_y,
            ],
            api_name=False,
            show_api=False,
        )

        foreground_preview.select(
            add_point,
            inputs=[foreground_input, points_state, closed_state],
            outputs=[foreground_preview, points_state, closed_state, status_box, offset_x, offset_y],
            api_name=False,
            show_api=False,
        )

        close_button.click(
            close_polygon,
            inputs=[foreground_input, points_state],
            outputs=[foreground_preview, closed_state, status_box],
            api_name=False,
            show_api=False,
        ).then(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )

        undo_button.click(
            undo_point,
            inputs=[foreground_input, points_state, closed_state],
            outputs=[foreground_preview, points_state, closed_state, status_box],
            api_name=False,
            show_api=False,
        ).then(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )

        clear_button.click(
            clear_points,
            inputs=[foreground_input],
            outputs=[foreground_preview, points_state, closed_state, status_box],
            api_name=False,
            show_api=False,
        ).then(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )

        offset_x.change(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )
        offset_y.change(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )
        background_input.change(
            update_background_preview,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],
            outputs=[background_preview],
            api_name=False,
            show_api=False,
        )

        blend_button.click(
            run_blending,
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y, iterations],
            outputs=[naive_output, blend_output, status_box],
            api_name=False,
            show_api=False,
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=False)
