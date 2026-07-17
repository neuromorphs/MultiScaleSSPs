# MultiScaleSSPs

MultiScaleSSPs studies **adaptive spatial resolution** in Spatial Semantic
Pointer (SSP) representations: instead of encoding an entire environment at
one fixed length scale, can a VSA-based spatial memory learn to spend more
of its representational capacity on regions/objects that need fine detail,
and less on ones that don't -- and how much of a trained memory can be
stripped away post-hoc while still decoding correctly?

## Introduction

An SSP encodes a continuous location as a high-dimensional vector whose
similarity to other locations falls off with distance at a rate set by a
single **length scale**. One length scale for a whole map is a compromise:
fine enough to resolve small objects wastes precision on large, coarse
regions; coarse enough for large regions can't tell nearby small objects
apart. MultiScaleSSPs builds synthetic multi-room/multi-object environments
and asks whether a *per-region* or *per-object* gain over the SSP's Fourier
scale bands -- learned by decoding through a bundled VSA memory -- can beat a
single shared scale, and which parts of the resulting representation (which
Fourier components, which spatial sample points) are actually load-bearing
for that decode, via post-hoc pruning.

## Background

This project builds on the Semantic Pointer (SP) / Spatial Semantic Pointer
(SSP) formalism (see Acknowledgements) and the `vsagym` Vector Symbolic
Architecture library. The core representation choices explored here:

- **Ground-truth environments** (`src/multiscalessps/envs/`): `RoomEnv`
  (flat region/object labels) and `BuildingEnv` (rooms containing furniture,
  so a point's label is simultaneously coarse "which room" and fine "which
  piece of furniture"). `scripts/indoor_env.py` is the main testbed -- a
  six-room layout with ModelNet10 furniture footprints, each room themed to
  need a different scale profile (large sparse items vs. dense clutter vs.
  structured rows).
- **Scale modulation**: each object/room is encoded with a learned
  non-negative gain per hexagonal scale block of the SSP spectrum, fit by
  decoding back through the bundled memory (`scripts/room_id_scale_modulation.py`
  and its quadrant/shape-world predecessors). Per-region gains are compared
  against a single shared profile and against learned kernel/window
  parameterizations of the same idea (`scripts/room_method_select.py`).
- **Component pruning** (`scripts/vsa_bin_pruning.py`,
  `examples/learnable_points/vsa_pruning.py`): once a room's memory is
  trained, how many of its Fourier bins (or spatial sample points, feeding
  each object's encoding before bundling) can be zeroed out while still
  decoding objects correctly? Random/magnitude/priority/learned-gate masking
  strategies are compared as compression curves against retained cosine
  similarity.
- **Real-world data loaders** (`src/multiscalessps/data/`): ModelNet10
  (CAD meshes), Robot@Home2 (indoor robot laser scans + room annotations),
  and SceneNN/Semantic3D (labeled point clouds) are included for eventually
  grounding these adaptive-scale ideas beyond synthetic worlds.

## Installation

Requires Python >= 3.10.

This project uses [uv](https://docs.astral.sh/uv/) and depends on a local editable clone of
[vsa-gym-wrapper](https://github.com/ctn-waterloo/vsa-gym-wrapper) (the `vsagym` package).

```bash
make setup
```

`make setup` clones `vsa-gym-wrapper` (only if it isn't already present) and then runs `uv sync`.
This creates `.venv` with `multiscalessps` and `vsagym` both installed in editable mode.
Run things with `uv run python ...` or activate the env with `source .venv/bin/activate`.

If you prefer to do it by hand:

```bash
git clone https://github.com/ctn-waterloo/vsa-gym-wrapper
uv sync
```

Alternatively, with plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./vsa-gym-wrapper -e .
```

## Usage

- Visualize a room environment (layout, dense sample data, and sampling behavior) and save the artifacts to a directory:

  ```bash
  python scripts/visualize_room.py
  ```

- Visualize the VSA baseline's decoded class maps and KL-vs-length-scale
  accuracy across a range of length scales, and save the artifacts to a
  directory:

  ```bash
  python scripts/visualize_vsa_baseline.py
  ```

## Members

- Shay Snyder
- Nicole Dumont
- Sven Krausse
- Lorin Achey
- Matthias Kampa

## Acknowledgements

The SP/SSP (Semantic Pointer / Spatial Semantic Pointer) representations in
`src/multiscalessps/ssps/` are adapted from the formalism developed in:

> Dumont, N. S.-Y. (2025). *Symbols, Dynamics, and Maps: A Neurosymbolic
> Approach to Spatial Cognition* (PhD Thesis). University of Waterloo,
> Waterloo, ON. https://hdl.handle.net/10012/21501

```bibtex
@phdthesis{dumont2025,
    title   = {Symbols, Dynamics, and Maps: A Neurosymbolic Approach to Spatial Cognition},
    author  = {Nicole Sandra-Yaffa Dumont},
    type    = {PhD Thesis},
    school  = {University of Waterloo},
    address = {Waterloo, ON},
    year    = {2025},
    url     = {https://hdl.handle.net/10012/21501}
}
```

This project was developed as part of the
[Telluride Neuromorphic Cognition Engineering Workshop 2026](https://sites.google.com/view/telluride-2026/home).
