# Design system: the Deepfake Check site

The look of the public site. The mockups are in [`designs/website/`](designs/website/README.md); the site's structure and behaviour are in the [website design](superpowers/specs/2026-11-17-website-design.md).

Chosen by Vaibhav on 2026-11-17: **the original direction B** (`b-signal.html`), with the useful parts of `b-signal-v2.html`: colour with meaning, the loudness and mouth trace, the three-checks layout, and the accessibility basics.

## Feel

Dark, precise, instrument-like. The site measures things and says how sure it is. Numbers are set in mono, like readings. Nothing is decorative unless it carries a number or a signal.

## Colour

Tokens on `:root`. The base is original B; the meaning colours come from v2.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0b0f14` | Page |
| `--panel` | `#121820` | Panels, bento cells |
| `--panel2` | `#161e28` | Top of the panel gradient |
| `--line` | `#232d3a` | Borders and rules |
| `--text` | `#e7ecf2` | Body text |
| `--muted` | `#8a96a6` | Secondary text, labels |
| `--accent` | `#5eead4` | Brand, primary actions, links, focus rings, and the lip-sync check |
| `--face` | `#a9c4ff` | Anything about the face: the face check, the mouth trace |
| `--sound` | `#f2b544` | Anything about sound: the voice check, the loudness trace |
| `--alert` | `#f87171` | Only for fakes and shortcuts: "signs of manipulation", raw file-cue ceilings |

Rules:
- Red means "fake" or "shortcut" and nothing else. Errors use plain text with an icon, not red.
- "No signs found" is never green. It uses `--text` on a `--panel` with an `--accent` outline, because it is not a clean bill of health.
- Body text keeps at least 4.5:1 contrast on its background; mono labels at least 3:1.

Original B's background glow (`radial-gradient` at the top right) and the gradient headline span stay.

## Type

- **Geist** (400, 500, 600, 700) for text and headings; **Geist Mono** (400, 500) for numbers, labels and scores. Both from Google Fonts with `display=swap`, as in original B.
- Headline: `clamp(42px, 6vw, 76px)`, weight 700, letter-spacing `-0.035em`, the second line in the teal-to-blue gradient (`--accent` to `#93c5fd`).
- Body 16px / 1.55. Lede 19px in `--muted`. Labels 12px mono, uppercase, letter-spacing `0.12em`.
- Big numbers ("stat"): 54px, weight 600, mono figures.

## Layout

- Max width 1180px, side padding 28px (16px under 600px).
- The bento grid: 6 columns, 14px gaps, cells with 14px radius, a 1px `--line` border and the panel gradient. Cells span 2, 3 or 6 columns.
- Under 860px: one column, the nav collapses to a menu button, bento cells stack, and no page scrolls sideways.

## Components

- **Nav:** brand dot and name, five text links, one teal "Check a video" button.
- **Pill:** the mono line above the headline (`P-001 / preregistered / every number has a receipt`).
- **Buttons:** primary (teal fill, dark text) and ghost (outline). One primary per screen.
- **Drop zone:** dashed border, the file rules underneath, keyboard-operable (a real file input behind it).
- **Check row:** the check's name in its colour (face blue, voice amber, lip-sync teal), its score in mono, and a bar showing where the score sits against real clips.
- **Outcome card:** the outcome in caps, which check raised it, the three check rows, the trace, and "How sure?" in plain words.
- **Trace:** speech loudness (amber) and mouth opening (blue) over the 2 s sync window, as in v2, but drawn from the uploaded clip. A labelled example (real against dubbed) explains it on How it works.
- **Dumbbell chart (from v2):** file-cue ceiling per corpus, raw (red) to cleaned (blue), with chance marked.

## Motion and access

- Traces draw in once (1.4 s ease-out); nothing loops. All motion off under `prefers-reduced-motion`.
- Visible focus ring (2px `--accent`) on every control. Every chart has a text equivalent.
- Status updates during a check go to an `aria-live` region.
