# Static foveated sampling player

Scroll-driven playback of pre-rendered fixation sequences. No Python at view time — works on GitHub Pages.

**Live:** `https://nblauch.github.io/fovi/`

## Create a run

```bash
python scripts/interactive_sampling_demo.py path/to/image.jpg outputs/seoul/ --device cpu
```

Writes PNGs plus `manifest.json` and `input.png`.

For older folders without a manifest:

```bash
python scripts/backfill_demo_manifest.py outputs/seoul
cp -r outputs/seoul web/foveated-player/runs/seoul
```

Optional `fixations.json` in the run folder (`[[row, col], …]`) enables cumulative cross overlays instead of the composite reference image.

## Preview locally

Player only:

```bash
cd web/foveated-player
python -m http.server 8080
```

Full GitHub Pages layout (player at `/`, docs at `/docs/`):

```bash
bash scripts/assemble_pages_site.sh
python -m http.server 8080 --directory site
```

Open `http://localhost:8080/` (defaults to `runs/seoul`) or `?run=../../outputs/seoul`.

## Experience

Scroll downward to unfold the sequence:

1. **Fixation** — source image
2. **Views** — global cartesian and flat manifold, side by side
3. Repeat for each fixation

Content below the fold starts blurred and soft; scrolling gradually sharpens and brings it into place.

## Publish

The docs deploy workflow assembles this folder at the site root and Sphinx docs at `/docs` on GitHub Pages. Copy runs into `web/foveated-player/runs/` and list them in `runs.json`.
