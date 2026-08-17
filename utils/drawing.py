"""
Drawing utilities for annotating video frames.
Provides functions for bounding boxes, info overlays, check status displays,
brightness meters, darkness warnings, and enhancement indicators.
"""

import cv2
import numpy as np


def draw_face_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name: str = "Unknown",
    confidence: float = 0.0,
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """Draw a face bounding box with name label."""
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Corner accents
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

    label = f"{name} ({confidence:.0%})" if confidence > 0 else name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 8, th + 8)
    bg_x1, bg_y1 = x1, label_y - th - 8
    bg_x2, bg_y2 = x1 + tw + 12, label_y + 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, label, (x1 + 6, label_y - 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def draw_info_overlay(
    frame: np.ndarray,
    text_lines: list[str],
    position: str = "top-left",
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.6,
) -> np.ndarray:
    """Draw a semi-transparent info panel with multiple lines of text."""
    if not text_lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    padding = 10

    max_width = 0
    for line in text_lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, tw)

    panel_w = max_width + padding * 2
    panel_h = line_height * len(text_lines) + padding * 2
    h, w = frame.shape[:2]

    if position == "top-left":
        px, py = 10, 10
    elif position == "top-right":
        px, py = w - panel_w - 10, 10
    elif position == "bottom-left":
        px, py = 10, h - panel_h - 10
    elif position == "bottom-right":
        px, py = w - panel_w - 10, h - panel_h - 10
    else:
        px, py = 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (100, 100, 100), 1, cv2.LINE_AA)

    for i, line in enumerate(text_lines):
        text_y = py + padding + (i + 1) * line_height - 4
        cv2.putText(frame, line, (px + padding, text_y), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)
    return frame


def draw_check_status(
    frame: np.ndarray,
    check_num: int,
    total: int,
    countdown: float,
) -> np.ndarray:
    """Draw check progress bar and countdown at the bottom of the frame."""
    h, w = frame.shape[:2]

    bar_h = 30
    bar_y = h - bar_h
    progress = check_num / total if total > 0 else 0
    bar_fill_w = int(w * progress)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if bar_fill_w > 0:
        cv2.rectangle(frame, (0, bar_y), (bar_fill_w, h), (0, 180, 80), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = f"Check {check_num}/{total}"
    cv2.putText(frame, status_text, (10, bar_y + 20), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    if countdown > 0:
        if countdown >= 60:
            cd_text = f"Next check in {int(countdown // 60)}m {int(countdown % 60)}s"
        else:
            cd_text = f"Next check in {countdown:.0f}s"
        (tw, _), _ = cv2.getTextSize(cd_text, font, 0.5, 1)
        cv2.putText(frame, cd_text, (w - tw - 10, bar_y + 20), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    dot_start_x = w // 2 - (total * 18) // 2
    for i in range(total):
        cx = dot_start_x + i * 18
        cy = bar_y + 15
        if i < check_num:
            cv2.circle(frame, (cx, cy), 5, (0, 220, 100), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 5, (100, 100, 100), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# NEW: Brightness / Darkness Drawing Utilities
# ──────────────────────────────────────────────

def draw_brightness_meter(
    frame: np.ndarray,
    brightness_value: float,
) -> np.ndarray:
    """
    Draw a vertical brightness meter in the top-right corner.

    Green (80–200)  = Good
    Yellow (40–80)  = Low Light (enhancing)
    Red   (<40)     = Too Dark
    Cyan  (>200)    = Too Bright

    Args:
        frame: BGR image.
        brightness_value: Mean brightness 0–255.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Meter dimensions
    meter_w = 22
    meter_h = 120
    margin = 12
    mx = w - meter_w - margin
    my = margin

    # Background with border
    overlay = frame.copy()
    cv2.rectangle(overlay, (mx - 4, my - 4), (mx + meter_w + 4, my + meter_h + 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (mx - 4, my - 4), (mx + meter_w + 4, my + meter_h + 30),
                  (80, 80, 80), 1, cv2.LINE_AA)

    # Fill level (clamped 0–255 mapped to meter height)
    fill_ratio = max(0.0, min(1.0, brightness_value / 255.0))
    fill_h = int(meter_h * fill_ratio)
    fill_y = my + meter_h - fill_h

    # Color based on brightness
    if brightness_value < 40:
        bar_color = (0, 0, 220)     # Red
        label = "DARK"
        label_color = (0, 0, 255)
    elif brightness_value < 80:
        bar_color = (0, 200, 255)   # Yellow
        label = "LOW"
        label_color = (0, 200, 255)
    elif brightness_value > 240:
        bar_color = (255, 200, 0)   # Cyan
        label = "HIGH"
        label_color = (255, 200, 0)
    else:
        bar_color = (0, 200, 0)     # Green
        label = "OK"
        label_color = (0, 220, 0)

    # Draw meter background (dark gray)
    cv2.rectangle(frame, (mx, my), (mx + meter_w, my + meter_h), (40, 40, 40), -1)

    # Draw fill
    if fill_h > 0:
        cv2.rectangle(frame, (mx, fill_y), (mx + meter_w, my + meter_h), bar_color, -1)

    # Threshold lines
    low_line_y = my + meter_h - int(meter_h * (40 / 255))
    high_line_y = my + meter_h - int(meter_h * (240 / 255))
    cv2.line(frame, (mx, low_line_y), (mx + meter_w, low_line_y), (0, 0, 200), 1)
    cv2.line(frame, (mx, high_line_y), (mx + meter_w, high_line_y), (200, 200, 0), 1)

    # Numeric value + label text below the meter
    font = cv2.FONT_HERSHEY_SIMPLEX
    val_text = f"{int(brightness_value)}"
    (tw, _), _ = cv2.getTextSize(val_text, font, 0.4, 1)
    tx = mx + (meter_w - tw) // 2
    cv2.putText(frame, val_text, (tx, my + meter_h + 14), font, 0.4,
                label_color, 1, cv2.LINE_AA)
    (tw2, _), _ = cv2.getTextSize(label, font, 0.35, 1)
    tx2 = mx + (meter_w - tw2) // 2
    cv2.putText(frame, label, (tx2, my + meter_h + 26), font, 0.35,
                label_color, 1, cv2.LINE_AA)

    return frame


def draw_darkness_warning(frame: np.ndarray) -> np.ndarray:
    """
    Draw a semi-transparent dark overlay with a large centered warning.

    Args:
        frame: BGR image.

    Returns:
        Annotated frame with darkness warning.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = (10, 10, 30)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Warning border
    cv2.rectangle(frame, (20, h // 2 - 50), (w - 20, h // 2 + 50), (0, 0, 200), 3, cv2.LINE_AA)

    # Background for text
    text_overlay = frame.copy()
    cv2.rectangle(text_overlay, (20, h // 2 - 50), (w - 20, h // 2 + 50), (0, 0, 80), -1)
    cv2.addWeighted(text_overlay, 0.5, frame, 0.5, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main warning text
    main_text = "LOW LIGHT - CHECK PAUSED"
    (tw, th), _ = cv2.getTextSize(main_text, font, 0.9, 2)
    tx = (w - tw) // 2
    ty = h // 2 - 5
    cv2.putText(frame, main_text, (tx, ty), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Sub text
    sub_text = "Waiting for lights to be restored..."
    (tw2, _), _ = cv2.getTextSize(sub_text, font, 0.5, 1)
    tx2 = (w - tw2) // 2
    cv2.putText(frame, sub_text, (tx2, h // 2 + 30), font, 0.5,
                (180, 180, 200), 1, cv2.LINE_AA)

    return frame


def draw_enhancement_indicator(frame: np.ndarray) -> np.ndarray:
    """
    Draw a small 'Enhanced' badge in the top-left when low-light
    enhancement is active.

    Args:
        frame: BGR image.

    Returns:
        Annotated frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "ENHANCED"
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    badge_w = tw + 16
    badge_h = th + 12
    bx, by = 10, 10

    # Badge background (amber)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (0, 160, 255), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (0, 180, 255), 1, cv2.LINE_AA)

    # Sun icon (small circle)
    cx = bx + 10
    cy = by + badge_h // 2
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # Text
    cv2.putText(frame, text, (bx + 20, by + th + 5), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def draw_retry_countdown(
    frame: np.ndarray,
    check_number: int,
    seconds_remaining: float,
    attempt: int,
    max_attempts: int,
) -> np.ndarray:
    """
    Draw a retry countdown overlay on the frame.

    Args:
        frame: BGR image.
        check_number: Which check is being retried.
        seconds_remaining: Seconds until retry.
        attempt: Current retry attempt number.
        max_attempts: Maximum retry attempts.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background bar
    bar_h = 40
    bar_y = h // 2 + 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (40, bar_y), (w - 40, bar_y + bar_h), (0, 0, 60), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (40, bar_y), (w - 40, bar_y + bar_h), (0, 100, 200), 1, cv2.LINE_AA)

    text = f"Retrying Check {check_number} in {seconds_remaining:.0f}s  (attempt {attempt}/{max_attempts})"
    (tw, _), _ = cv2.getTextSize(text, font, 0.5, 1)
    tx = (w - tw) // 2
    cv2.putText(frame, text, (tx, bar_y + 26), font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Color scheme constants
# ──────────────────────────────────────────────

COLOR_CONFIRMED = (0, 200, 0)       # Green  — confirmed recognized
COLOR_UNKNOWN = (0, 0, 220)         # Red    — unknown person
COLOR_UNCERTAIN = (0, 165, 255)     # Orange — twin/uncertain
COLOR_PROCESSING = (255, 100, 0)    # Blue   — detecting/processing
COLOR_SPOOF = (0, 0, 255)           # Red    — spoofing detected (checks)
COLOR_SPOOF_PREVIEW = (0, 255, 255)  # Yellow — spoofing detected (live preview)


def draw_spoof_warning(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    spoof_type: str = "FAKE",
) -> np.ndarray:
    """
    Draw a RED dashed bounding box with "FAKE" label for spoofing detections.

    Args:
        frame: BGR image.
        bbox: (x1, y1, x2, y2).
        spoof_type: Type of spoofing detected.

    Returns:
        Annotated frame.
    """
    x1, y1, x2, y2 = bbox
    color = COLOR_SPOOF

    # Dashed rectangle
    dash_length = 10
    gap_length = 6
    # Top edge
    for x in range(x1, x2, dash_length + gap_length):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 2, cv2.LINE_AA)
    # Bottom edge
    for x in range(x1, x2, dash_length + gap_length):
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 2, cv2.LINE_AA)
    # Left edge
    for y in range(y1, y2, dash_length + gap_length):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2, cv2.LINE_AA)
    # Right edge
    for y in range(y1, y2, dash_length + gap_length):
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2, cv2.LINE_AA)

    # Semi-transparent red overlay on face
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Label: "FAKE — screen_photo"
    label = f"FAKE - {spoof_type}" if spoof_type and spoof_type != "FAKE" else "FAKE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 8, th + 8)
    bg_x1, bg_y1 = x1, label_y - th - 10
    bg_x2, bg_y2 = x1 + tw + 16, label_y + 6

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, label, (x1 + 8, label_y - 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Red circle badge with X
    badge_x = x2 - 16
    badge_y = y1 + 16
    cv2.circle(frame, (badge_x, badge_y), 14, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (badge_x, badge_y), 14, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "X", (badge_x - 6, badge_y + 5), font, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_uncertain_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name_a: str,
    name_b: str,
    score_a: float,
    score_b: float,
) -> np.ndarray:
    """
    Draw an ORANGE bounding box for twin/uncertain detections.
    Label shows both candidate names and scores.

    Args:
        frame: BGR image.
        bbox: (x1, y1, x2, y2).
        name_a: First candidate name.
        name_b: Second candidate name.
        score_a: Similarity score for candidate A.
        score_b: Similarity score for candidate B.

    Returns:
        Annotated frame.
    """
    x1, y1, x2, y2 = bbox
    color = COLOR_UNCERTAIN

    # Orange box with corners
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

    # Label: "Name_A(91%) / Name_B(89%) ?"
    label = f"{name_a}({score_a:.0%}) / {name_b}({score_b:.0%}) ?"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 8, th + 8)
    bg_x1, bg_y1 = x1, label_y - th - 8
    bg_x2, bg_y2 = x1 + tw + 12, label_y + 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, label, (x1 + 6, label_y - 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Question mark badge
    qx = x2 - 18
    qy = y1 + 18
    cv2.circle(frame, (qx, qy), 12, color, -1, cv2.LINE_AA)
    cv2.putText(frame, "?", (qx - 5, qy + 5), font, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_spoof_detected_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name: str = "",
    spoof_type: str = "",
) -> np.ndarray:
    """
    Draw a YELLOW bounding box with "SPOOF DETECTED" label for the live
    preview. Used when a face is recognized (e.g. "MG") but the liveness
    check flags it as a spoof (phone photo / screen photo / print).

    Args:
        frame: BGR image.
        bbox: (x1, y1, x2, y2).
        name: Recognized identity name (shown as a sub-label).
        spoof_type: Type of spoof ("screen_photo", "printed_photo", ...).

    Returns:
        Annotated frame.
    """
    x1, y1, x2, y2 = bbox
    color = COLOR_SPOOF_PREVIEW  # Yellow

    # Solid yellow box (thicker for visibility)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    # Semi-transparent yellow fill on the face
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main label: "SPOOF DETECTED"
    main_label = "SPOOF DETECTED"
    font_scale_main = 0.62
    thickness_main = 2
    (tw, th), _ = cv2.getTextSize(main_label, font, font_scale_main, thickness_main)

    label_y = max(y1 - 8, th + 10)
    bg_x1, bg_y1 = x1, label_y - th - 10
    bg_x2, bg_y2 = x1 + tw + 16, label_y + 6

    bg_overlay = frame.copy()
    cv2.rectangle(bg_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(bg_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, main_label, (x1 + 8, label_y - 2), font, font_scale_main,
                (0, 0, 0), thickness_main, cv2.LINE_AA)

    # Sub-label: recognized name + spoof type (e.g. "MG — screen_photo")
    sub_parts = []
    if name:
        sub_parts.append(name)
    if spoof_type:
        sub_parts.append(spoof_type.replace("_", " "))
    if sub_parts:
        sub_label = " — ".join(sub_parts)
        font_scale_sub = 0.45
        thickness_sub = 1
        (stw, sth), _ = cv2.getTextSize(sub_label, font, font_scale_sub, thickness_sub)

        sub_y = label_y + sth + 6
        sub_bg_overlay = frame.copy()
        cv2.rectangle(sub_bg_overlay,
                       (x1, sub_y - sth - 4),
                       (x1 + stw + 12, sub_y + 4),
                       color, -1)
        cv2.addWeighted(sub_bg_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, sub_label, (x1 + 6, sub_y - 1), font, font_scale_sub,
                    (0, 0, 0), thickness_sub, cv2.LINE_AA)

    # Warning triangle badge in the top-right corner of the box
    bx = x2 - 18
    by = y1 + 18
    pts = np.array([[bx, by - 12], [bx - 11, by + 8], [bx + 11, by + 8]], np.int32)
    cv2.fillPoly(frame, [pts], color)
    cv2.putText(frame, "!", (bx - 3, by + 5), font, 0.5,
                (0, 0, 0), 2, cv2.LINE_AA)

    return frame

