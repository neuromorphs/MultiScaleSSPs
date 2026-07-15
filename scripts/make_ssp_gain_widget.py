#!/usr/bin/env python
"""Generate a standalone HTML widget for exploring SSP scale gains.

Layout: a 2x2 grid of square plots on the left -- the gain-weighted
similarity map phi(x) . phi(x') and the encoding vectors (rows of A) on top,
and below them the two encoded scene maps of make_scene_maps.py (map 1:
fine-line rooms, map 2: large vegetation), decoded from their bundled SSPs
with the user-chosen scale gains. The narrow right column has a preset
dropdown ('uniform', 'map 1', 'map 2' -- the learned gains for each scene,
fit by scene_gain_fit.py machinery at generation time) above one card per
scale: its hex-grid pattern and a 0..1 gain slider, fine at the top.

All computation is client-side JavaScript. Each scene bundle is embedded as
its half-spectrum Fourier coefficients b_k = sum_i e^{i a_k x_i}; at load the
page builds one basis map per scale for the kernel and for both scenes in a
single trig pass, so every slider move is a 5-term weighted sum per pixel and
all four panels re-render live (~10 ms).

    python scripts/make_ssp_gain_widget.py
    xdg-open results/widgets/ssp_gain_widget.html
"""

import argparse
import json
from pathlib import Path as FSPath

import numpy as np
import matplotlib.pyplot as plt

from vsagym.spaces import HexagonalSSPSpace
from make_scene_maps import build_scenes
from scene_gain_fit import (bundle_coeffs, scale_basis_maps, fit_gains,
                            wall_segments, segment_bundle_coeffs)

# same scheme as ssp_encoding_animations.py
COLORS = {
    "background": "white",
    "agent": "black",
    "arrow": "#12406b",
    "frame_edge": "#c8cdd3",
    "text": "#444444",
}

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SSP scale-gain explorer</title>
<style>
  body { background: __BG__; color: __TEXT__; margin: 24px;
         font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  p.sub { font-size: 13px; color: #777; margin: 0 0 18px; }
  .wrap { display: flex; gap: 22px; align-items: flex-start; }
  .grid2 { display: grid; grid-template-columns: auto auto; gap: 14px; }
  .panel { border: 1px solid __EDGE__; padding: 8px 8px 4px; border-radius: 6px; }
  .panel h2 { font-size: 11.5px; font-weight: 600; margin: 0 0 6px;
              letter-spacing: 0.4px; color: #666; text-transform: uppercase; }
  canvas.sq { width: 320px; height: 320px; display: block; }
  .right { display: flex; flex-direction: column; gap: 12px; width: 250px; }
  .presetrow { display: flex; flex-direction: column; gap: 4px; }
  .presetrow label { font-size: 12px; font-weight: 600; }
  select { font-size: 13px; padding: 4px 6px; border: 1px solid __EDGE__;
           border-radius: 4px; background: __BG__; color: __TEXT__; }
  .card { border: 1px solid __EDGE__; border-radius: 6px; padding: 8px;
          display: flex; gap: 10px; align-items: center; }
  .card canvas { width: 96px; height: 96px; flex: none; }
  .ctl { display: flex; flex-direction: column; gap: 4px; flex: 1; }
  .ctl .name { font-size: 12px; font-weight: 600; }
  .ctl .val { font-size: 12px; color: #777; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; accent-color: __ARROW__; }
</style>
</head>
<body>
<h1>SSP scale-gain explorer</h1>
<p class="sub">Slide the per-scale gains g&#8347; (or pick a preset) &mdash; the
similarity kernel, the encoding vectors, and both encoded scene maps update
with the modulation.</p>
<div class="wrap">
  <div class="grid2">
    <div class="panel"><h2>similarity &phi;(x)&middot;&phi;(x&prime;)</h2>
      <canvas id="sim" class="sq" width="__GRIDN__" height="__GRIDN__"></canvas></div>
    <div class="panel"><h2>encoding vectors (rows of A)</h2>
      <canvas id="arr" class="sq" width="360" height="360"></canvas></div>
    <div class="panel"><h2>map 1: rooms &mdash; encoded</h2>
      <canvas id="map0" class="sq" width="__GRIDN__" height="__GRIDN__"></canvas></div>
    <div class="panel"><h2>map 2: vegetation &mdash; encoded</h2>
      <canvas id="map1" class="sq" width="__GRIDN__" height="__GRIDN__"></canvas></div>
  </div>
  <div class="right">
    <div class="presetrow"><label for="preset">gain preset</label>
      <select id="preset">
        <option value="uniform">uniform</option>
        <option value="map1">map 1 (rooms)</option>
        <option value="map2">map 2 (vegetation)</option>
        <option value="custom">custom</option>
      </select></div>
    <div id="cards"></div>
  </div>
</div>
<script>
const D = __DATA__;
const nS = D.scales.length, gridN = D.gridN, nPix = gridN * gridN;
const gains = new Array(nS).fill(1.0);

// ---- basis maps: kernel + both scenes, one trig pass at load --------------
const kernelB = Array.from({length: nS}, () => new Float32Array(nPix));
const sceneB = D.scenes.map(() => Array.from({length: nS},
                                             () => new Float32Array(nPix)));
const counts = new Array(nS).fill(0);
D.rowScale.forEach(s => counts[s]++);
{
  const half = D.world / 2;
  for (let k = 0; k < D.ax.length; k++) {
    const s = D.rowScale[k], ax = D.ax[k], ay = D.ay[k];
    const kb = kernelB[s];
    for (let j = 0; j < gridN; j++) {
      const y = half - (j + 0.5) / gridN * D.world;   // canvas row 0 = top
      const gy = (j + 0.5) / gridN * D.world;         // scenes: y up, row 0 top
      for (let i = 0; i < gridN; i++) {
        const x = (i + 0.5) / gridN * D.world - half;
        const p = j * gridN + i;
        kb[p] += Math.cos(ax * x + ay * y);
        const th = ax * ((i + 0.5) / gridN * D.world) + ay * (D.world - gy);
        const c = Math.cos(th), sn = Math.sin(th);
        for (let m = 0; m < D.scenes.length; m++)
          sceneB[m][s][p] += c * D.scenes[m].bRe[k] + sn * D.scenes[m].bIm[k];
      }
    }
  }
}

// ---- shared LUT painter ----------------------------------------------------
function paint(ctx, img, vals, norm, floor) {
  const px = img.data, f = floor || 0;
  for (let p = 0; p < nPix; p++) {
    const v = norm > 1e-12 ? (vals[p] / norm - f) / (1 - f) : 0;
    const li = Math.max(0, Math.min(255, (v * 255) | 0));
    const c = D.lut[li], q = p * 4;
    px[q] = c[0]; px[q + 1] = c[1]; px[q + 2] = c[2]; px[q + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const tmp = new Float32Array(nPix);
function combine(bset) {
  tmp.fill(0);
  let any = false;
  for (let s = 0; s < nS; s++) {
    if (gains[s] <= 0) continue;
    any = true;
    const b = bset[s], g = gains[s];
    for (let p = 0; p < nPix; p++) tmp[p] += g * b[p];
  }
  return any;
}

// ---- similarity map --------------------------------------------------------
const simCtx = document.getElementById("sim").getContext("2d");
const simImg = simCtx.createImageData(gridN, gridN);
function renderSim() {
  let wsum = 0;
  for (let s = 0; s < nS; s++) wsum += gains[s] * counts[s];
  combine(kernelB);
  paint(simCtx, simImg, tmp, wsum);
  simCtx.fillStyle = D.colors.agent;
  simCtx.beginPath();
  simCtx.arc(gridN / 2, gridN / 2, gridN * 0.014, 0, 7);
  simCtx.fill();
}

// ---- encoded scene maps ----------------------------------------------------
const mapCtx = [], mapImg = [];
for (let m = 0; m < D.scenes.length; m++) {
  mapCtx[m] = document.getElementById("map" + m).getContext("2d");
  mapImg[m] = mapCtx[m].createImageData(gridN, gridN);
}
function renderMap(m) {
  if (!combine(sceneB[m])) { paint(mapCtx[m], mapImg[m], tmp, 0); return; }
  // normalize by ~99.5th percentile so one hot spot cannot wash out the map
  const sub = [];
  for (let p = 0; p < nPix; p += 8) sub.push(tmp[p]);
  sub.sort((a, b) => a - b);
  paint(mapCtx[m], mapImg[m], tmp, sub[Math.floor(sub.length * 0.995)],
        D.simFloor);
}

// ---- encoding-vector arrows ------------------------------------------------
const arrCtx = document.getElementById("arr").getContext("2d");
function renderArrows() {
  const W = arrCtx.canvas.width, cx = W / 2, cy = W / 2, L = W * 0.44;
  arrCtx.clearRect(0, 0, W, W);
  arrCtx.fillStyle = D.colors.agent;
  arrCtx.beginPath(); arrCtx.arc(cx, cy, 4, 0, 7); arrCtx.fill();
  arrCtx.strokeStyle = D.colors.arrow;
  arrCtx.lineWidth = 2;
  for (const a of D.arrows) {
    const tx = cx + a.ux * L, ty = cy - a.uy * L;    // canvas y is down
    arrCtx.globalAlpha = 0.1 + 0.9 * gains[a.s];
    arrCtx.beginPath(); arrCtx.moveTo(cx, cy); arrCtx.lineTo(tx, ty); arrCtx.stroke();
    const ang = Math.atan2(ty - cy, tx - cx), h = 10;
    arrCtx.fillStyle = D.colors.arrow;
    arrCtx.beginPath();
    arrCtx.moveTo(tx + Math.cos(ang) * h * 0.45, ty + Math.sin(ang) * h * 0.45);
    arrCtx.lineTo(tx - Math.cos(ang - 0.45) * h, ty - Math.sin(ang - 0.45) * h);
    arrCtx.lineTo(tx - Math.cos(ang + 0.45) * h, ty - Math.sin(ang + 0.45) * h);
    arrCtx.fill();
  }
  arrCtx.globalAlpha = 1;
}

function renderAll() {
  renderSim(); renderArrows(); renderMap(0); renderMap(1);
}

// ---- per-scale cards: hex-grid thumbnail + gain slider ----------------------
const cardsDiv = document.getElementById("cards");
const sliders = [], vals = [], thumbs = [];
for (let o = nS - 1; o >= 0; o--) {                  // fine (top) -> coarse
  const s = o;
  const card = document.createElement("div");
  card.className = "card";
  const tag = s === nS - 1 ? " (fine)" : s === 0 ? " (coarse)" : "";
  card.innerHTML =
    '<canvas width="96" height="96"></canvas>' +
    '<div class="ctl"><div class="name">scale ' + D.scales[s].toFixed(2) + tag +
    '</div><input type="range" min="0" max="1" step="0.01" value="1">' +
    '<div class="val">g = 1.00</div></div>';
  cardsDiv.appendChild(card);
  const cv = card.querySelector("canvas"), ctx = cv.getContext("2d");
  const img = ctx.createImageData(96, 96);
  const rows = D.thumbIdx[s], half = D.world / 2;
  for (let j = 0; j < 96; j++) {
    const y = half - (j + 0.5) / 96 * D.world;
    for (let i = 0; i < 96; i++) {
      const x = (i + 0.5) / 96 * D.world - half;
      let v = 0;
      for (const k of rows) v += Math.cos(D.ax[k] * x + D.ay[k] * y);
      v /= rows.length;
      const li = Math.max(0, Math.min(255, (v * 255) | 0));
      const c = D.lut[li], q = (j * 96 + i) * 4;
      img.data[q] = c[0]; img.data[q + 1] = c[1]; img.data[q + 2] = c[2];
      img.data[q + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  sliders[s] = card.querySelector("input");
  vals[s] = card.querySelector(".val");
  thumbs[s] = cv;
  sliders[s].addEventListener("input", () => {
    gains[s] = parseFloat(sliders[s].value);
    presetSel.value = "custom";
    syncCard(s);
    renderAll();
  });
}
function syncCard(s) {
  vals[s].textContent = "g = " + gains[s].toFixed(2);
  thumbs[s].style.opacity = 0.25 + 0.75 * gains[s];
}

// ---- presets ----------------------------------------------------------------
const presetSel = document.getElementById("preset");
presetSel.addEventListener("change", () => {
  const g = D.presets[presetSel.value];
  if (!g) return;                                    // "custom": leave as-is
  for (let s = 0; s < nS; s++) {
    gains[s] = g[s];
    sliders[s].value = g[s];
    syncCard(s);
  }
  renderAll();
});

for (let s = 0; s < nS; s++) syncCard(s);
renderAll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=float, default=10.0)
    ap.add_argument("--n-scales", type=int, default=7)
    ap.add_argument("--n-rotates", type=int, default=15)
    ap.add_argument("--scale-min", type=float, default=np.pi / 27)
    ap.add_argument("--length-scale", type=float, default=1 / 3,
                    help="nominal scales stay <= pi; ls=1/3 shifts the "
                         "effective band to pi/9..3*pi rad/m")
    ap.add_argument("--grid-n", type=int, default=220)
    ap.add_argument("--sim-floor", type=float, default=0.25,
                    help="encoded-map display floor, fraction of the max")
    ap.add_argument("--arrow-rotates", type=int, default=3)
    ap.add_argument("--arrow-len-pow", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="results/widgets/ssp_gain_widget.html")
    args = ap.parse_args()

    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  scale_min=args.scale_min,
                                  scale_sampling="log",
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    n_free = (d - 1) // 2
    a_half = ssp_space.phase_matrix[1:n_free + 1] / ssp_space.length_scale.flatten()
    nv, ns, nr = ssp_space.grid_basis_dim, ssp_space.n_scales, ssp_space.n_rotates
    row_scale = ((np.arange(len(a_half)) // nv) % ns)

    # scene bundles + learned gain presets (same NNLS fit as scene_gain_fit)
    scenes = build_scenes(args.grid_n)
    gn = args.grid_n
    cc = (np.arange(gn) + 0.5) / gn * args.world
    X, Y = np.meshgrid(cc, cc)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    scene_data, presets = [], {"uniform": [1.0] * ns, "custom": None}
    for i, key in enumerate(("indoor", "outdoor")):
        occ = scenes[key]["occ"].ravel()
        if key == "indoor":   # walls: exact line-integral encoding
            b = segment_bundle_coeffs(a_half, wall_segments(scenes[key]["rects"]))
        else:
            b = bundle_coeffs(a_half, grid[occ])
        B = scale_basis_maps(a_half, row_scale, ns, b, grid)
        g, _ = fit_gains(B, occ)
        presets[f"map{i + 1}"] = [round(float(v), 3) for v in g / g.max()]
        scene_data.append({
            "name": key,
            "bRe": [round(float(v), 3) for v in b.real],
            "bIm": [round(float(v), 3) for v in b.imag],
        })
        print(f"{key}: {int(occ.sum())} pts, learned preset "
              f"{presets[f'map{i + 1}']}")

    # arrows: all scales, a few evenly spaced rotations, plus conjugates
    r_pick = np.floor(np.arange(args.arrow_rotates) * nr
                      / args.arrow_rotates).astype(int)
    idx = np.array([r * ns * nv + s * nv + v
                    for r in r_pick for s in range(ns) for v in range(nv)])
    rows_sel = a_half[idx]
    mags = np.linalg.norm(rows_sel, axis=1)
    disp = (mags / np.linalg.norm(a_half, axis=1).max()) ** args.arrow_len_pow
    uv = rows_sel / mags[:, None] * disp[:, None]
    arrows = [{"ux": round(float(sg * u[0]), 4), "uy": round(float(sg * u[1]), 4),
               "s": int(row_scale[k])}
              for u, k in zip(uv, idx) for sg in (1, -1)]

    lut = (plt.get_cmap("Blues")(np.linspace(0, 1, 256))[:, :3] * 255
           ).round().astype(int).tolist()

    data = {
        "world": args.world,
        "gridN": args.grid_n,
        "scales": [round(float(s), 4) for s in ssp_space.scales],
        "ax": [round(float(v), 6) for v in a_half[:, 0]],
        "ay": [round(float(v), 6) for v in a_half[:, 1]],
        "rowScale": row_scale.tolist(),
        # thumbnail = one hex group per scale (rotation 0, all 3 vertices)
        "thumbIdx": [[s * nv + v for v in range(nv)] for s in range(ns)],
        "arrows": arrows,
        "scenes": scene_data,
        "presets": presets,
        "simFloor": args.sim_floor,
        "lut": lut,
        "colors": COLORS,
    }

    html = (HTML
            .replace("__DATA__", json.dumps(data))
            .replace("__GRIDN__", str(args.grid_n))
            .replace("__BG__", COLORS["background"])
            .replace("__TEXT__", COLORS["text"])
            .replace("__EDGE__", COLORS["frame_edge"])
            .replace("__ARROW__", COLORS["arrow"]))
    out = FSPath(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, ssp_dim={d}, "
          f"{len(a_half)} rows, {len(arrows)} arrows)")


if __name__ == "__main__":
    main()
