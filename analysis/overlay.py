"""
Post-hoc 2D overlays for recorded Isaac Gym frames — arrows, impact rings,
subtitles. The physics (forces) are real and applied in the env; these overlays
only *annotate* a frame after it is rendered, so nothing touches the simulator
state. This is how we make an external disturbance read as external (a labelled
arrow hitting the robot) rather than the robot appearing to thrash on its own.

Pure numpy + PIL; no Isaac Gym dependency here. The caller supplies the robot's
2D screen position (see project_world_to_screen in record_video.py).
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_arrow(draw: ImageDraw.ImageDraw, tip, angle_deg, length=90,
               color=(255, 60, 60), width=6):
    """Arrow whose TIP is at `tip`, pointing toward it from `angle_deg`
    (0 = coming from the right, 90 = from below, 180 = from left)."""
    a = math.radians(angle_deg)
    tx, ty = tip
    tail = (tx + length * math.cos(a), ty + length * math.sin(a))
    draw.line([tail, (tx, ty)], fill=color, width=width)
    # arrowhead
    head = math.radians(28)
    hl = 22
    for s in (+1, -1):
        hx = tx + hl * math.cos(a + s * head)
        hy = ty + hl * math.sin(a + s * head)
        draw.line([(tx, ty), (hx, hy)], fill=color, width=width)


def draw_impact_ring(draw, center, r, color=(255, 210, 40), width=5):
    x, y = center
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)
    for k in range(8):  # spark lines
        a = math.radians(k * 45)
        draw.line([(x + r * math.cos(a), y + r * math.sin(a)),
                   (x + (r + 14) * math.cos(a), y + (r + 14) * math.sin(a))],
                  fill=color, width=3)


def annotate(frame: np.ndarray, *, subtitle=None, arrow=None, ring=None,
             banner_color=(0, 0, 0)) -> np.ndarray:
    """Return a copy of `frame` (H,W,3 uint8) with overlays.
    - subtitle: str shown in a bottom banner
    - arrow: dict(tip=(x,y), angle_deg=.., color=.., length=..)
    - ring:  dict(center=(x,y), r=.., color=..)
    """
    img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    if arrow:
        draw_arrow(d, arrow["tip"], arrow.get("angle_deg", 0),
                   length=arrow.get("length", 90),
                   color=arrow.get("color", (255, 60, 60)))
    if ring:
        draw_impact_ring(d, ring["center"], ring.get("r", 34),
                         color=ring.get("color", (255, 210, 40)))
    if subtitle:
        f = _font(max(20, W // 30))
        tb = d.textbbox((0, 0), subtitle, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 12
        bx0, by0 = (W - tw) // 2 - pad, H - th - 3 * pad
        d.rectangle([bx0, by0, bx0 + tw + 2 * pad, by0 + th + 2 * pad],
                    fill=(*banner_color, 170))
        d.text((bx0 + pad, by0 + pad), subtitle, font=f, fill=(255, 255, 255))
    return np.asarray(img)
