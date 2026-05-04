"""Small HTML helpers used by app.py.

Pulled out of app.py so the file stays focused on layout/reactivity.
"""
from constants import LABEL_MAP, COLORS, BADGE_CLS


def metric_card(value, label) -> str:
    """A small grey card with a big number on top and a label below."""
    return (
        f'<div style="background:#f3f4f6;border-radius:8px;padding:.9rem;text-align:center;">'
        f'<div style="font-size:24px;font-weight:600;">{value}</div>'
        f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">{label}</div>'
        f'</div>'
    )


def decision_badge(decision: str, *, large: bool = False) -> str:
    """Pill-shaped colored badge for a decision label."""
    style = "font-size:15px;padding:8px 18px;display:inline-block;margin-bottom:12px;" if large else ""
    return (
        f'<span class="badge {BADGE_CLS[decision]}" style="{style}">'
        f'{LABEL_MAP[decision]}'
        f'</span>'
    )


def prob_bars(probs: dict[str, float]) -> str:
    """Stacked horizontal probability bars, sorted descending."""
    out = ""
    for d in sorted(probs, key=probs.get, reverse=True):
        pct = probs[d] * 100
        out += (
            f'<div style="margin-bottom:10px;">'
            f'  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">'
            f'    <span style="color:#6b7280;">{LABEL_MAP[d]}</span>'
            f'    <span style="font-weight:600;">{pct:.0f}%</span>'
            f'  </div>'
            f'  <div style="height:8px;background:#f3f4f6;border-radius:4px;overflow:hidden;">'
            f'    <div style="height:100%;width:{pct:.0f}%;background:{COLORS[d]};'
            f'border-radius:4px;transition:width .3s;"></div>'
            f'  </div>'
            f'</div>'
        )
    return out


def alert_box(level: str, message: str) -> str:
    """level in {'green','amber'}."""
    cls = f"alert-{level}"
    return f'<div class="{cls}">{message}</div>'


def field_position_label(yardline_100: int) -> str:
    """Format a yardline_100 value as readable field position."""
    if yardline_100 < 50:
        return f"Opp {yardline_100}"
    if yardline_100 == 50:
        return "Midfield"
    return f"Own {100 - yardline_100}"


CSS = """
body { background:#f8f9fa; font-family:'Segoe UI',system-ui,sans-serif; }
.card { background:white; border-radius:10px; border:1px solid #e5e7eb;
        padding:1.25rem; margin-bottom:1rem; }
.card-title { font-size:11px; font-weight:600; text-transform:uppercase;
              letter-spacing:.07em; color:#6b7280; margin-bottom:.75rem; }
.badge { display:inline-block; padding:5px 14px; border-radius:6px;
         font-size:13px; font-weight:600; }
.badge-go   { background:#EAF3DE; color:#3B6D11; border:1.5px solid #639922; }
.badge-punt { background:#E6F1FB; color:#0C447C; border:1.5px solid #378ADD; }
.badge-fg   { background:#FAEEDA; color:#854F0B; border:1.5px solid #BA7517; }
.alert-green { background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
               padding:.75rem 1rem; color:#15803d; font-size:14px; margin-top:.75rem; }
.alert-amber { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
               padding:.75rem 1rem; color:#92400e; font-size:14px; margin-top:.75rem; }
.loading-box { background:white; border-radius:10px; border:1px solid #e5e7eb;
               padding:3rem; text-align:center; color:#6b7280; font-size:15px; }
.two-col  { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.four-col { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem; }
.era-compare { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.era-compare .card { border-left: 4px solid #d1d5db; }
.era-compare .card.dynamic { border-left-color: #639922; }
.era-compare .card.traditional { border-left-color: #6b7280; }
"""
