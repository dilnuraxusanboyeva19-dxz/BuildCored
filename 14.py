"""
PWMSimulator  —  Windows Edition
==================================
Interactive PWM (Pulse Width Modulation) simulator.
A slider controls duty cycle 0-100%.
The square wave visualises in real time.
The computed average voltage displays.
A virtual LED brightens and dims proportionally.

Install dependencies:
    pip install matplotlib numpy
    (tkinter is bundled with Python on Windows)

Usage:
    python pwmsimulator.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

# ── Config ────────────────────────────────────────────────────────────────────

V_HIGH      = 5.0        # HIGH voltage level (5V like Arduino)
V_LOW       = 0.0        # LOW voltage level
FREQ_HZ     = 1.0        # PWM frequency shown (1 Hz = easy to visualise)
CYCLES      = 3          # how many full cycles to show
T_TOTAL     = CYCLES / FREQ_HZ
POINTS      = 2000       # resolution of the waveform
FPS         = 30         # animation frame rate

INIT_DUTY   = 50.0       # starting duty cycle %

# Layout colours
BG          = "#0a0a10"
PANEL_BG    = "#0d0d1a"
CARD_BG     = "#12121f"
ACCENT      = "#00e5ff"
GREEN       = "#69ff47"
YELLOW      = "#ffcc00"
RED         = "#ff4081"
DIM         = "#44445a"
TEXT        = "#ccccdd"
GRID        = "#1a1a2e"
SLIDER_BG   = "#1a1a2e"

# LED colours at various intensities
LED_OFF_COLOR    = "#1a0a0a"
LED_DIM_COLOR    = "#660000"
LED_MID_COLOR    = "#cc3300"
LED_BRIGHT_COLOR = "#ff6600"
LED_MAX_COLOR    = "#ffff00"

# ── PWM waveform generation ───────────────────────────────────────────────────

def make_pwm_wave(duty_pct: float, t_total: float = T_TOTAL,
                  n_points: int = POINTS, freq: float = FREQ_HZ) -> tuple:
    """
    Generate a PWM square wave for the given duty cycle.

    duty_pct: 0–100
    Returns (t, v) arrays where v is 0 or V_HIGH.

    The wave has sharp edges: within each period T = 1/freq,
    the output is HIGH for duty_cycle * T seconds, then LOW.
    """
    t       = np.linspace(0, t_total, n_points, endpoint=False)
    period  = 1.0 / freq
    phase   = (t % period) / period          # normalised phase [0, 1)
    duty    = duty_pct / 100.0
    v       = np.where(phase < duty, V_HIGH, V_LOW).astype(np.float32)
    return t, v


def average_voltage(duty_pct: float) -> float:
    """V_avg = duty_cycle × V_high (for 0V low)."""
    return (duty_pct / 100.0) * V_HIGH


def led_color(duty_pct: float) -> str:
    """Map duty cycle to an LED colour from off → dim → bright."""
    d = duty_pct / 100.0
    if d < 0.01:
        return LED_OFF_COLOR
    # Interpolate R, G, B channels
    r0, g0, b0 = 0x1a, 0x0a, 0x0a   # off
    if d < 0.25:
        frac = d / 0.25
        r1, g1, b1 = 0x66, 0x00, 0x00
    elif d < 0.5:
        frac = (d - 0.25) / 0.25
        r0, g0, b0 = 0x66, 0x00, 0x00
        r1, g1, b1 = 0xcc, 0x33, 0x00
    elif d < 0.75:
        frac = (d - 0.5) / 0.25
        r0, g0, b0 = 0xcc, 0x33, 0x00
        r1, g1, b1 = 0xff, 0x66, 0x00
    else:
        frac = (d - 0.75) / 0.25
        r0, g0, b0 = 0xff, 0x66, 0x00
        r1, g1, b1 = 0xff, 0xff, 0x00

    r = int(r0 + frac * (r1 - r0))
    g = int(g0 + frac * (g1 - g0))
    b = int(b0 + frac * (b1 - b0))
    return f"#{r:02x}{g:02x}{b:02x}"


def led_glow_alpha(duty_pct: float) -> float:
    """Glow halo transparency scales with duty cycle."""
    return max(0.0, (duty_pct / 100.0) ** 0.6)

# ── Figure construction ───────────────────────────────────────────────────────

def build_figure():
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.canvas.manager.set_window_title(
        "PWMSimulator  —  Day 17 | BUILDCORED ORCAS"
    )

    # Main layout: top area for plots, bottom strip for slider + controls
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[5.5, 1],
        top=0.92, bottom=0.04,
        left=0.05, right=0.97,
        hspace=0.12,
    )

    # Top: waveform (left) + info panel (right)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[0],
        width_ratios=[3, 1],
        wspace=0.08,
    )

    ax_wave = fig.add_subplot(gs_top[0])
    ax_info = fig.add_subplot(gs_top[1])

    # Waveform axes styling
    ax_wave.set_facecolor(PANEL_BG)
    ax_wave.set_xlim(0, T_TOTAL)
    ax_wave.set_ylim(-0.5, V_HIGH + 0.8)
    ax_wave.set_xlabel("Time (s)", color=DIM, fontsize=9)
    ax_wave.set_ylabel("Voltage (V)", color=DIM, fontsize=9)
    ax_wave.set_title("PWM Square Wave", color=ACCENT,
                      fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_wave.tick_params(colors=DIM, labelsize=8)
    ax_wave.spines[:].set_color(DIM)
    ax_wave.grid(True, color=GRID, linewidth=0.6, linestyle="--")

    # V_HIGH and V_LOW reference lines
    ax_wave.axhline(V_HIGH, color=DIM, linewidth=0.8, linestyle=":",
                    alpha=0.6)
    ax_wave.axhline(V_LOW,  color=DIM, linewidth=0.8, linestyle=":",
                    alpha=0.6)
    ax_wave.text(T_TOTAL + 0.02, V_HIGH, f"{V_HIGH}V",
                 color=DIM, fontsize=7, va="center")
    ax_wave.text(T_TOTAL + 0.02, V_LOW, f"{V_LOW}V",
                 color=DIM, fontsize=7, va="center")

    # Period annotation arrow (will be updated)
    ax_wave.text(0.02, V_HIGH + 0.45, "T = 1/f", color=DIM,
                 fontsize=7, va="center")

    # Average voltage fill (shaded horizontal band)
    avg_fill = ax_wave.axhspan(0, 0, alpha=0.15, color=YELLOW, zorder=1)

    # Average voltage line
    avg_line = ax_wave.axhline(0, color=YELLOW, linewidth=1.5,
                                linestyle="--", alpha=0.8, zorder=2)
    avg_label = ax_wave.text(0.02, 0, "", color=YELLOW, fontsize=8,
                              va="bottom", zorder=3)

    # Main PWM waveform line
    t_init, v_init = make_pwm_wave(INIT_DUTY)
    wave_line, = ax_wave.plot(t_init, v_init, color=ACCENT,
                               linewidth=2.0, alpha=0.95, zorder=5)

    # Fill under wave
    wave_fill = ax_wave.fill_between(t_init, v_init, alpha=0.12,
                                      color=ACCENT, zorder=4)

    # Duty cycle shading (first period HIGH region)
    duty_shade = ax_wave.axvspan(0, 0, alpha=0.08, color=GREEN, zorder=3)

    # ── Info panel (right side) ───────────────────────────────────────────────
    ax_info.set_facecolor(PANEL_BG)
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.set_title("Live Readout", color=ACCENT,
                       fontsize=11, fontweight="bold", pad=8, loc="left")

    # ── Metric cards ──────────────────────────────────────────────────────────
    # Each card: background rect + label text + value text

    cards = {}
    card_defs = [
        ("duty",    0.75, "Duty Cycle",     GREEN,  "%"),
        ("avg_v",   0.52, "Avg Voltage",    YELLOW, "V"),
        ("period",  0.30, "Period",         ACCENT, "s"),
        ("freq",    0.10, "Frequency",      ACCENT, "Hz"),
    ]
    for key, y_frac, label, color, unit in card_defs:
        # Card background
        card = FancyBboxPatch((0.05, y_frac - 0.09), 0.9, 0.175,
                               boxstyle="round,pad=0.02",
                               linewidth=1, edgecolor=color,
                               facecolor=CARD_BG, alpha=0.9,
                               transform=ax_info.transAxes, zorder=2)
        ax_info.add_patch(card)

        # Label
        ax_info.text(0.12, y_frac + 0.06, label,
                     transform=ax_info.transAxes, color=DIM,
                     fontsize=7.5, va="center", zorder=3)

        # Value (large)
        val_txt = ax_info.text(0.5, y_frac - 0.01, "—",
                                transform=ax_info.transAxes,
                                color=color, fontsize=20,
                                fontweight="bold", fontfamily="monospace",
                                ha="center", va="center", zorder=3)
        # Unit
        ax_info.text(0.88, y_frac - 0.01, unit,
                     transform=ax_info.transAxes, color=color,
                     fontsize=9, va="center", zorder=3)

        cards[key] = val_txt

    # ── Virtual LED ───────────────────────────────────────────────────────────
    # Placed above the right panel as a separate axes
    ax_led = fig.add_axes([0.79, 0.76, 0.16, 0.14], facecolor=BG)
    ax_led.set_xlim(-1.2, 1.2)
    ax_led.set_ylim(-1.2, 1.2)
    ax_led.set_aspect("equal")
    ax_led.axis("off")

    # Glow halo (multiple concentric circles)
    glow_circles = []
    for r, a in [(1.1, 0.06), (0.95, 0.10), (0.80, 0.15)]:
        gc = Circle((0, 0), r, color=LED_OFF_COLOR, alpha=a, zorder=1)
        ax_led.add_patch(gc)
        glow_circles.append(gc)

    # LED body
    led_body = Circle((0, 0), 0.6, color=LED_OFF_COLOR,
                       zorder=2, linewidth=2, edgecolor="#333344")
    ax_led.add_patch(led_body)

    # Specular highlight (white dot in upper-left of LED)
    led_highlight = Circle((-0.2, 0.2), 0.12,
                            color="white", alpha=0.0, zorder=3)
    ax_led.add_patch(led_highlight)

    ax_led.text(0, -1.05, "Virtual LED", color=DIM,
                ha="center", va="top", fontsize=7)

    # ── Slider axis (bottom strip) ────────────────────────────────────────────
    ax_slider = fig.add_axes([0.12, 0.045, 0.72, 0.030],
                              facecolor=SLIDER_BG)

    slider = Slider(
        ax=ax_slider,
        label="Duty Cycle  ",
        valmin=0.0,
        valmax=100.0,
        valinit=INIT_DUTY,
        valstep=0.5,
        color=ACCENT,
    )
    slider.label.set_color(TEXT)
    slider.label.set_fontsize(9)
    slider.valtext.set_color(ACCENT)
    slider.valtext.set_fontweight("bold")
    slider.valtext.set_fontfamily("monospace")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "PWMSimulator", ha="center", va="top",
             fontsize=17, fontweight="bold", color=ACCENT,
             fontfamily="monospace")
    fig.text(0.5, 0.945,
             "Pulse Width Modulation  ·  Arduino analogWrite() / Pico pwm.duty_u16()  ·  Day 17 — BUILDCORED ORCAS",
             ha="center", va="top", fontsize=7.5, color=DIM)

    handles = dict(
        fig=fig,
        ax_wave=ax_wave,
        ax_info=ax_info,
        ax_led=ax_led,
        wave_line=wave_line,
        wave_fill=[wave_fill],     # mutable ref
        avg_line=avg_line,
        avg_fill=[avg_fill],       # mutable ref
        avg_label=avg_label,
        duty_shade=[duty_shade],   # mutable ref
        cards=cards,
        led_body=led_body,
        led_highlight=led_highlight,
        glow_circles=glow_circles,
        slider=slider,
    )
    return handles

# ── Update function ───────────────────────────────────────────────────────────

def update(duty_pct: float, handles: dict):
    """
    Redraw everything for the new duty cycle value.
    Called whenever the slider changes.
    """
    t, v = make_pwm_wave(duty_pct)
    v_avg = average_voltage(duty_pct)

    # ── Waveform ──────────────────────────────────────────────────────────────
    handles["wave_line"].set_data(t, v)

    # Replace wave fill
    handles["wave_fill"][0].remove()
    handles["wave_fill"][0] = handles["ax_wave"].fill_between(
        t, v, alpha=0.12, color=ACCENT, zorder=4
    )

    # Replace average fill
    handles["avg_fill"][0].remove()
    handles["avg_fill"][0] = handles["ax_wave"].axhspan(
        0, v_avg, alpha=0.12, color=YELLOW, zorder=1
    )

    # Average line + label
    handles["avg_line"].set_ydata([v_avg, v_avg])
    handles["avg_label"].set_position((0.02, v_avg + 0.05))
    handles["avg_label"].set_text(f"V_avg = {v_avg:.2f} V")

    # Duty cycle shading (first period high region)
    handles["duty_shade"][0].remove()
    high_end = (duty_pct / 100.0) / FREQ_HZ
    handles["duty_shade"][0] = handles["ax_wave"].axvspan(
        0, high_end, alpha=0.10, color=GREEN, zorder=3
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    handles["cards"]["duty"].set_text(f"{duty_pct:.1f}")
    handles["cards"]["avg_v"].set_text(f"{v_avg:.2f}")
    handles["cards"]["period"].set_text(f"{1/FREQ_HZ:.3f}")
    handles["cards"]["freq"].set_text(f"{FREQ_HZ:.1f}")

    # ── Virtual LED ───────────────────────────────────────────────────────────
    color   = led_color(duty_pct)
    alpha_g = led_glow_alpha(duty_pct)

    handles["led_body"].set_facecolor(color)
    handles["led_highlight"].set_alpha(alpha_g * 0.5)

    # Glow halos pulse with duty cycle
    glow_alphas = [0.06, 0.10, 0.15]
    for i, gc in enumerate(handles["glow_circles"]):
        gc.set_facecolor(color)
        gc.set_alpha(glow_alphas[i] * alpha_g)

    handles["fig"].canvas.draw_idle()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "─" * 54)
    print("  PWMSimulator  ·  Pulse Width Modulation")
    print("  Day 17 — BUILDCORED ORCAS")
    print("─" * 54)
    print(f"  V_HIGH     : {V_HIGH} V")
    print(f"  V_LOW      : {V_LOW} V")
    print(f"  Frequency  : {FREQ_HZ} Hz  (for visualisation)")
    print(f"  Cycles     : {CYCLES} shown")
    print("─" * 54)
    print("  Drag the slider to change duty cycle.")
    print("  Close the window to exit.")
    print("─" * 54 + "\n")

    handles = build_figure()

    # Initial draw
    update(INIT_DUTY, handles)

    # Slider callback
    def on_slider(val):
        update(val, handles)

    handles["slider"].on_changed(on_slider)

    plt.show()


if __name__ == "__main__":
    main()
