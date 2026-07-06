# MultiScaleSSPs

<!-- TODO: brief introduction of the project -->

## Introduction

<!-- TODO: brief overview of what MultiScaleSSPs is and what problem it addresses -->

## Background

<!-- TODO: motivation, related work, and context for the project -->

## Installation

Requires Python >= 3.9.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

- Visualize a room environment (layout, dense sample data, and sampling behavior) and save the artifacts to a directory:

  ```bash
  python scripts/visualize_room.py --out-dir .scratch/room
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
