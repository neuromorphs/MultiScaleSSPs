# MultiScaleSSPs

<!-- TODO: brief introduction of the project -->

## Introduction

<!-- TODO: brief overview of what MultiScaleSSPs is and what problem it addresses -->

## Background

<!-- TODO: motivation, related work, and context for the project -->

## Installation

Requires Python >= 3.9.

This project uses [uv](https://docs.astral.sh/uv/) and depends on a local editable clone of
[vsa-gym-wrapper](https://github.com/ctn-waterloo/vsa-gym-wrapper) (the `vsagym` package).

```bash
git clone https://github.com/ctn-waterloo/vsa-gym-wrapper
uv sync
```

This creates `.venv` with `multiscalessps` and `vsagym` both installed in editable mode.
Run things with `uv run python ...` or activate the env with `source .venv/bin/activate`.

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

<!-- TODO: list contributors -->

-
-
-

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
