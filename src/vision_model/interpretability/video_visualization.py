"""Dependency-light temporal plots and target-conditioned spatial overlays."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

from .temporal_attribution import TemporalAttribution
from .spatial_final_block import SpatialFinalBlockResult
from .map_geometry import spatial_evidence_overlay


def _color(value: float, maximum: float) -> tuple[int, int, int]:
    ratio = min(1., max(0., value / maximum)) if maximum > 0 else 0.
    return (int(255 * ratio), int(235 * (1 - ratio)), int(245 * (1 - ratio)))


def _spatial_color(intensity: np.ndarray) -> np.ndarray:
    """Map a shared [0,1] attribution scale from blue through pale to red."""
    low = np.array((35, 80, 220), dtype=np.float32)
    middle = np.array((235, 235, 235), dtype=np.float32)
    high = np.array((230, 45, 35), dtype=np.float32)
    amount = np.clip(intensity, 0, 1)[..., None]
    color = np.where(amount <= .5,
                     low + 2 * amount * (middle - low),
                     middle + 2 * (amount - .5) * (high - middle))
    return color.astype(np.uint8)


def save_temporal_figures(result: TemporalAttribution, output: Path, *,
                          video_path: Path | None = None, storyboard_count: int = 8,
                          max_plot_tokens: int = 256,
                          frames_by_index: dict[int, np.ndarray] | None = None) -> list[Path]:
    """Render a target-specific heatmap row, layer rows, timeline, and storyboard.

    Video decoding reads selected frames only. Plotting may downsample columns;
    saved numeric attribution remains at original temporal resolution.
    """
    output.mkdir(parents=True, exist_ok=True)
    valid = result.valid_length
    if valid < 1 or storyboard_count < 1:
        raise ValueError("Invalid visualization length")
    columns = np.unique(np.linspace(0, valid - 1, min(valid, max_plot_tokens)).round().astype(int))
    timestamps = result.timestamps.numpy()
    rows = [result.relevance.numpy(), *result.layer_relevance.numpy()]
    labels = ["target scalar; all layers", *[f"layer {i}" for i in result.layer_indices]]
    cell = max(4, min(18, 1400 // len(columns)))
    left, top, row_height = 190, 75, 65
    width, height = left + len(columns) * cell + 20, top + len(rows) * row_height + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "Temporal key relevance for selected scalar target", fill="black")
    draw.text((12, 31), f"{result.target['kind']}={result.target_value:.5g}; source time in seconds", fill="black")
    maximum = max(float(np.max(row[:valid])) for row in rows)
    for row_index, (name, scores) in enumerate(zip(labels, rows)):
        y = top + row_index * row_height
        draw.text((10, y + 12), name, fill="black")
        for col, token in enumerate(columns):
            x = left + col * cell
            draw.rectangle((x, y, x + cell - 1, y + 37), fill=_color(float(scores[token]), maximum))
        draw.text((left, y + 40), f"{timestamps[columns[0]]:.2f}s", fill="black")
        draw.text((width - 85, y + 40), f"{timestamps[columns[-1]]:.2f}s", fill="black")
    draw.text((10, height - 30), "One row per scalar target; no pixel or spill mask", fill="black")
    heatmap = output / "temporal_heatmap.png"
    image.save(heatmap)

    timeline = Image.new("RGB", (max(640, width), 240), "white")
    draw = ImageDraw.Draw(timeline)
    draw.text((12, 10), f"{result.video_id} | {result.target['kind']} | temporal relevance", fill="black")
    scores = result.relevance.numpy()[:valid]
    event = result.event_pooling_weights.numpy()[:valid]
    plot_width = timeline.width - 95
    for name, series, y0, color in (("Transformer gradient", scores, 45, (190, 40, 50)),
                                     ("Event pooling", event, 135, (40, 70, 180))):
        draw.text((12, y0), name, fill=color)
        high = float(np.max(series))
        for i, value in enumerate(series):
            x = 65 + round(plot_width * i / max(1, valid - 1))
            draw.line((x, y0 + 68, x, y0 + 68 - round(48 * float(value) / max(high, 1e-12))),
                      fill=color, width=max(1, plot_width // max(valid, 1)))
        draw.text((65, y0 + 72), f"{timestamps[0]:.2f}s", fill="black")
        draw.text((timeline.width - 75, y0 + 72), f"{timestamps[valid-1]:.2f}s", fill="black")
    timeline_path = output / "timeline.png"
    timeline.save(timeline_path)
    paths = [heatmap, timeline_path]
    if result.relevance_maps is not None:
        # These rows are attention-query times for ONE selected scalar target.
        # They are not separate per-row target attributions.
        matrices = result.relevance_maps.mean(1).numpy()
        panels = [matrices.mean(0), *matrices]
        panel_labels = ["selected scalar; layers mean", *[f"layer {i}" for i in result.layer_indices]]
        plotted = columns
        cell_size = max(3, min(9, 850 // len(plotted)))
        panel_span = len(plotted) * cell_size + 65
        figure = Image.new("RGB", (100 + panel_span * len(panels),
                                   85 + len(plotted) * cell_size + 55), "white")
        axis = ImageDraw.Draw(figure)
        axis.text((12, 10), "Query time (vertical) versus source time (horizontal)", fill="black")
        axis.text((12, 28), "All panels explain the SAME scalar target; white upper triangle is causally masked", fill="black")
        scale = max(float(np.max(panel[:valid, :valid])) for panel in panels)
        for pane, (label, matrix) in enumerate(zip(panel_labels, panels)):
            x0 = 80 + pane * panel_span
            axis.text((x0, 58), label, fill="black")
            for y_slot, query in enumerate(plotted):
                for x_slot, key in enumerate(plotted):
                    fill = "white" if key > query else _color(float(matrix[query, key]), scale)
                    x = x0 + x_slot * cell_size
                    y = 82 + y_slot * cell_size
                    axis.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), fill=fill)
            axis.text((x0, figure.height - 38),
                      f"source: {timestamps[plotted[0]]:.2f} to {timestamps[plotted[-1]]:.2f}s", fill="black")
        axis.text((5, 83), f"query {timestamps[plotted[0]]:.2f}s", fill="black")
        axis.text((5, figure.height - 62), f"{timestamps[plotted[-1]]:.2f}s", fill="black")
        map_path = output / "causal_query_key_maps.png"
        figure.save(map_path)
        paths.append(map_path)
    if video_path is not None or frames_by_index is not None:
        if result.frame_indices is None:
            raise ValueError("Storyboard requires original frame indices")
        rank = np.argsort(-scores, kind="stable")
        chosen = list(rank[:min(storyboard_count, valid)])
        chosen.sort()
        indices = [int(result.frame_indices[i]) for i in chosen]
        if frames_by_index is None:
            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise ImportError("TorchCodec is required for frame storyboard decoding") from exc
            decoder = VideoDecoder(str(video_path), dimension_order="NCHW", device="cpu", seek_mode="exact")
            if any(index < 0 or index >= len(decoder) for index in indices):
                raise ValueError("Frame index outside video")
            decoded = decoder.get_frames_at(indices=indices).data
            frames_by_index = {index: decoded[position].permute(1, 2, 0).numpy()
                               for position, index in enumerate(indices)}
        if any(index not in frames_by_index for index in indices):
            raise ValueError("Storyboard frame missing from decoded input")
        tile_w, tile_h = 256, 195
        board = Image.new("RGB", (tile_w * 4, tile_h * ((len(chosen) + 3) // 4)), "white")
        rank_by_index = {int(token): position + 1 for position, token in enumerate(rank)}
        for position, token in enumerate(chosen):
            frame = Image.fromarray(frames_by_index[indices[position]]).convert("RGB")
            frame.thumbnail((tile_w - 8, 150))
            x, y = (position % 4) * tile_w, (position // 4) * tile_h
            board.paste(frame, (x + (tile_w - frame.width) // 2, y))
            caption = (f"frame {indices[position]} | {timestamps[token]:.2f}s\n"
                       f"score {scores[token]:.4g} | rank {rank_by_index[token]}")
            ImageDraw.Draw(board).text((x + 5, y + 151), caption, fill="black")
        storyboard = output / "storyboard.png"
        board.save(storyboard)
        paths.append(storyboard)
    return paths


def save_temporal_overlay_video(result: TemporalAttribution, output: Path, *,
                                frames_by_index: dict[int, np.ndarray],
                                sampled_frame_indices: list[int], source_fps: float,
                                playback_fps: float) -> Path:
    """Put a moving temporal heatmap strip below unchanged sampled video frames.

    Two source frames belong to each tubelet. The strip colors *times*, not
    pixels; the frame image is never tinted or presented as a spill mask.
    The MP4 contains the sampled clip, without source audio.
    """
    if (result.frame_indices is None or len(sampled_frame_indices) != 2 * result.valid_length
            or source_fps <= 0 or playback_fps <= 0):
        raise ValueError("Video rendering needs two sampled frames per valid tubelet and positive FPS")
    if any(index not in frames_by_index for index in sampled_frame_indices):
        raise ValueError("A sampled source frame is missing from decoded input")
    if sampled_frame_indices[1::2] != result.frame_indices[:result.valid_length].tolist():
        raise ValueError("Sampled frame indices disagree with attribution metadata")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "temporal_attention_video.mp4"
    temporary = output / "temporal_attention_video.tmp.mp4"
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    width, image_height, panel_height = 640, 360, 160
    height = image_height + panel_height
    scores = result.relevance[:result.valid_length].numpy()
    maximum = float(np.max(scores))
    times = result.timestamps.numpy()
    first, last = float(times[0]), float(times[result.valid_length - 1])
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(playback_fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for position, frame_index in enumerate(sampled_frame_indices):
            token = position // 2
            canvas = Image.new("RGB", (width, height), (18, 23, 32))
            frame = Image.fromarray(frames_by_index[frame_index]).convert("RGB")
            frame.thumbnail((width, image_height), Image.Resampling.BILINEAR)
            canvas.paste(frame, ((width - frame.width) // 2,
                                 (image_height - frame.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((14, image_height + 10),
                      f"Temporal attention relevance | {result.target['kind']}={result.target_value:.4g}",
                      fill="white")
            draw.text((14, image_height + 30),
                      f"Source frame {frame_index} at {frame_index / source_fps:.2f}s"
                      f" | token {token} score {scores[token]:.4g}", fill="white")
            left, right, top, bottom = 18, width - 18, image_height + 61, image_height + 109
            for index, score in enumerate(scores):
                x0 = left + round((right - left) * index / len(scores))
                x1 = left + round((right - left) * (index + 1) / len(scores))
                draw.rectangle((x0, top, max(x0, x1 - 1), bottom),
                               fill=_color(float(score), maximum))
            playhead = left + round((right - left) * (token + .5) / len(scores))
            draw.line((playhead, top - 6, playhead, bottom + 6), fill=(255, 255, 255), width=3)
            draw.text((left, bottom + 9), f"{first:.2f}s", fill="white")
            draw.text((right - 54, bottom + 9), f"{last:.2f}s", fill="white")
            draw.text((left + 92, bottom + 9), "Heatmap marks relevant times, not spill pixels", fill="white")
            assert process.stdin is not None
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not encode temporal video: {error[-1000:]}")
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return destination


def save_spatial_final_block_video(result: SpatialFinalBlockResult, output: Path, *,
                                   frames_by_index: dict[int, np.ndarray],
                                   sampled_frame_indices: list[int], source_fps: float,
                                   playback_fps: float, checkpoint_image_size: int = 256,
                                   shared_scale: float | None = None) -> Path:
    """Overlay target-conditioned V-JEPA patch sensitivity on sampled frames.

    Opacity encodes magnitude and zero leaves original RGB pixels unchanged.
    Native key-cell boundaries are retained. Supply shared_scale for videos
    being compared; otherwise a single clip uses its own fixed maximum.
    """
    maps = result.positive_maps.numpy()
    length, grid_height, grid_width = maps.shape
    if grid_height != grid_width or len(sampled_frame_indices) != 2 * length:
        raise ValueError("Spatial maps and sampled two-frame tubelets disagree")
    if source_fps <= 0 or playback_fps <= 0:
        raise ValueError("FPS must be positive")
    if any(index not in frames_by_index for index in sampled_frame_indices):
        raise ValueError("A sampled source frame is missing")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "spatial_final_block_overlay.mp4"
    temporary = output / "spatial_final_block_overlay.tmp.mp4"
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    panel_width, panel_height, caption_height = 640, 400, 100
    width, height = panel_width * 2, panel_height + caption_height
    maximum = float(maps.max()) if shared_scale is None else float(shared_scale)
    if not np.isfinite(maximum) or maximum < 0 or (shared_scale is not None and maximum==0):
        raise ValueError('Shared scale must be positive and finite')
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(playback_fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for position, source_index in enumerate(sampled_frame_indices):
            token = position // 2
            original = frames_by_index[source_index]
            # Resize the source first. Resizing a composited overlay afterward
            # would interpolate red across a zero-evidence cell boundary.
            source_frame = Image.fromarray(original).convert('RGB')
            panel_scale = min(panel_width / source_frame.width, panel_height / source_frame.height)
            source_frame = source_frame.resize((max(1, round(source_frame.width * panel_scale)),
                max(1, round(source_frame.height * panel_scale))), Image.Resampling.BILINEAR)
            display = np.asarray(source_frame).copy()
            if maximum > 0:
                overlay = spatial_evidence_overlay(torch.from_numpy(display),
                    torch.from_numpy(maps[token]),scale=maximum,signed=False,alpha=.4,
                    scope='conditional_attention_key_gradient',checkpoint_size=checkpoint_image_size).numpy()
            else:
                overlay = display.copy()  # no positive gradients to color
            canvas = Image.new("RGB", (width, height), (18, 23, 32))
            for column, source in enumerate((display, overlay)):
                frame = Image.fromarray(source).convert("RGB")
                canvas.paste(frame, (column * panel_width + (panel_width - frame.width) // 2,
                                     (panel_height - frame.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, panel_height + 8), "Original", fill="white")
            draw.text((panel_width + 12, panel_height + 8),
                      "Causal trajectory target via frozen V-JEPA patches", fill="white")
            draw.text((12, panel_height + 32),
                      f"Frame {source_index} | {source_index / source_fps:.2f}s | tubelet {token}",
                      fill="white")
            draw.text((12, panel_height + 57),
                      "Zero: transparent | Red opacity: conditional key sensitivity; not a pixel/spill mask",
                      fill="white")
            if maximum > 0:
                bar_left, bar_right, bar_top = panel_width + 95, width - 95, panel_height + 76
                gradient = np.linspace(0, 1, bar_right - bar_left + 1, dtype=np.float32)
                amount=.4*gradient[:,None]
                colors=(np.array([18,23,32])*(1-amount)+np.array([255,0,0])*amount).astype(np.uint8)
                for x, color in enumerate(colors, start=bar_left):
                    draw.line((x, bar_top, x, bar_top + 12), fill=tuple(int(c) for c in color))
                draw.text((panel_width + 12, bar_top), "Low", fill="white")
                draw.text((bar_right + 10, bar_top), "High", fill="white")
            else:
                draw.text((panel_width + 12, panel_height + 76),
                          "No positive attribution for this target", fill="white")
            assert process.stdin is not None
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not encode spatial video: {error[-1000:]}")
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return destination
