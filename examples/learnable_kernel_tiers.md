# Tiered vs. hierarchical encoding in `learnable_kernel_tiers.py`

This note explains two axes of variation in [learnable_kernel_tiers.py](learnable_kernel_tiers.py)
that are easy to conflate because both involve "levels": **tiered** (how many
separate memory vectors the building is split across) and **hierarchical**
(how containment — furniture belongs to a room — is encoded *inside* a single
memory vector). They are independent choices; the script's Tier 3 model can
be run in either a flat or a hierarchical mode.

## Shared notation

Every encoder here is an FHRR (Fourier Holographic Reduced Representation)
map: a position $x \in \mathbb{R}^2$ is turned into a $D$-dimensional unit
phasor via a random phase matrix $\Phi \in \mathbb{R}^{2 \times D}$ and a
per-class lengthscale $\ell_c$ ([learnable_kernel_tiers.py:349-355](learnable_kernel_tiers.py#L349-L355)):

$$
\mathrm{pos}_c(x) = \exp\!\Big(i \, \frac{x \Phi}{\ell_c}\Big) \in \mathbb{C}^D, \qquad |\mathrm{pos}_c(x)_d| = 1
$$

Binding ($\odot$, elementwise complex product) attaches a class identity
vector $v_c$ (fixed random unit phasor) and a learnable per-class gain
$g_c = e^{\log g_c}$:

$$
\mathrm{atom}(x, c) = g_c \cdot \mathrm{pos}_c(x) \odot v_c
$$

Bundling ($\oplus$, sum + renormalize) superposes many atoms into one memory
vector:

$$
M = \frac{\sum_i \mathrm{atom}(x_i, c_i)}{\big\| \sum_i \mathrm{atom}(x_i, c_i) \big\|}
$$

Querying class $c$ at position $x$ against memory $M$ unbinds and correlates
([learnable_kernel_tiers.py:357-368](learnable_kernel_tiers.py#L357-L368)):

$$
\mathrm{score}_c(x) = \frac{1}{D}\,\mathrm{Re}\Big\langle \overline{\mathrm{pos}_c(x)} \odot \overline{v_c},\; M \Big\rangle
$$

fed through a learned-temperature softmax for a class distribution. All
lengthscales $\ell_c$, gains $g_c$, and temperatures are learned jointly by
gradient descent on classification loss — that's the "learnable kernel"
part; this note is about what gets bundled into which memory.

## Axis 1: Tiered — how many memories, at what granularity

"Tiered" refers to splitting the *problem* into separate encoders at
different levels of granularity, each with its own memory vector(s):

| Tier | Memory count | Encodes | Code |
|---|---|---|---|
| **Tier 1** | 1 shared memory | room identity only (5 classes incl. wall) | [L650-672](learnable_kernel_tiers.py#L650-L672) |
| **Tier 2** | 4 memories (one per room) | floor + local furniture, per room | [L677-716](learnable_kernel_tiers.py#L677-L716) |
| **Tier 3** | 1 shared whole-building memory | rooms *and* furniture together | [L742-854](learnable_kernel_tiers.py#L742-L854) |

Tier 1's memory:

$$
M_1 = \bigoplus_{i \,:\, \text{room pts}} \mathrm{atom}(x_i,\, \mathrm{room}_i)
$$

Tier 2 builds one independent memory $M_2^{(r)}$ per room $r$, trained only
on that room's local furniture-class set.

The **cascade T1→T2** method ([L718-737](learnable_kernel_tiers.py#L718-L737))
composes these two *independently trained* memories post-hoc, by
marginalizing the Tier‑2 posterior over the Tier‑1 room posterior:

$$
P(\text{furn}=c \mid x) \;=\; \sum_{r} \underbrace{P(\text{room}=r \mid x;\, M_1)}_{\text{Tier 1 query}} \cdot \underbrace{P(\text{furn}=c \mid x;\, M_2^{(r)})}_{\text{Tier 2 query}}
$$

Nothing about *how* $M_1$ or $M_2^{(r)}$ was built knows about this
composition — it happens entirely in probability space at query time, across
vectors that were never bound or trained together.

## Axis 2: Flat vs. hierarchical — containment *inside* one memory

This choice only exists once you've committed to Tier 3's single shared
memory. Given room atoms and furniture atoms, how is "this furniture belongs
to this room" represented?

**Flat** ([`HouseMap.build_memory`](learnable_kernel_tiers.py#L421-L438),
`hierarchical=False`) — furniture atoms are bundled straight into the same
memory as room atoms, with no link between the two:

$$
M_{\text{flat}} \;=\; \Big(\bigoplus_i \mathrm{atom}(x_i, \mathrm{room}_i)\Big) \;\oplus\; \Big(\bigoplus_j \mathrm{atom}(x_j, \mathrm{furn}_j)\Big)
$$

Decoding furniture is a direct unbind against $M_{\text{flat}}$, identical in
form to Tier 1's query — no room information involved
([L450-465, `if not hierarchical`](learnable_kernel_tiers.py#L450-L465)):

$$
\mathrm{score}_c(x) = \tfrac{1}{D}\,\mathrm{Re}\big\langle \overline{\mathrm{pos}_c(x)} \odot \overline{v^{\text{furn}}_c},\; M_{\text{flat}} \big\rangle
$$

**Hierarchical** (`hierarchical=True`, [L432-434](learnable_kernel_tiers.py#L432-L434)) —
each furniture atom is *additionally bound* with its own room's identity
vector $v^{\text{room}}_{r(j)}$ before bundling, so containment is baked into
the vector itself:

$$
M_{\text{hier}} \;=\; \Big(\bigoplus_i \mathrm{atom}(x_i, \mathrm{room}_i)\Big) \;\oplus\; \Big(\bigoplus_j \mathrm{atom}(x_j, \mathrm{furn}_j) \odot v^{\text{room}}_{r(j)}\Big)
$$

Because binding with a different room vector $v^{\text{room}}_{r'}$ is
(approximately) orthogonal to binding with $v^{\text{room}}_r$, `bed` stored
under `bedroom` becomes near-orthogonal to a hypothetical `bed` under
`kitchen` *by construction* — not merely down-weighted after the fact.

Decoding now requires unbinding a room vector before the furniture vector
([L450-467](learnable_kernel_tiers.py#L450-L467)); the room to unbind can be
either the ground-truth room (`oracle`, used during training/calibration) or
one decoded from the same memory's room query, marginalized exactly like the
cascade's formula — but now over bindings *within one shared memory* instead
of across two separately-trained memories:

$$
\mathrm{score}_c(x, r) = \tfrac{1}{D}\,\mathrm{Re}\big\langle \overline{\mathrm{pos}_c(x)} \odot \overline{v^{\text{furn}}_c} \odot \overline{v^{\text{room}}_r},\; M_{\text{hier}} \big\rangle
$$

$$
P(\text{furn}=c \mid x) = \sum_r P(\text{room}=r \mid x;\, M_{\text{hier}}) \cdot P(\text{furn}=c \mid x, r;\, M_{\text{hier}})
$$

## Why the formulas look alike but aren't

The cascade formula (Axis 1) and the hierarchical-marginalization formula
(Axis 2) have the *same shape* — both are $\sum_r P(r\mid x)\,P(c\mid x, r)$ —
which is exactly why the two axes are easy to conflate. The difference is
what's on the right-hand side of the conditioning:

- **Cascade (tiered):** $P(c \mid x, r)$ comes from a *separate* memory
  $M_2^{(r)}$, trained in isolation on room $r$'s points only.
- **Hierarchical:** $P(c \mid x, r)$ comes from unbinding $v^{\text{room}}_r$
  out of the *one* jointly-trained memory $M_{\text{hier}}$, which also holds
  every other room's furniture.

Empirically ([`lk_results/summary.png`](lk_results/summary.png),
[`lk_results/furniture_decode.png`](lk_results/furniture_decode.png)), the
tiered/cascade split (separate memories, separate capacity) reconstructs
furniture far more cleanly than either Tier‑3 single-memory variant — flat or
hierarchical — because Tier 3 forces every class to share one memory's
interference budget, and hierarchical binding narrows cross-room confusion
but doesn't add capacity back.
