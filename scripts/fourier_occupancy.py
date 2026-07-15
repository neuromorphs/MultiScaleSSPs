"""3D shape occupancy with Fourier feature networks.

Adapted from the Fourier Feature Networks paper repository:
https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_shape_occupancy.ipynb
(Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions
in Low Dimensional Domains", NeurIPS 2020)

Modernized for current JAX (the notebook used the removed jax.experimental.stax
and jax.experimental.optimizers APIs; this uses plain JAX + optax) and for
current trimesh (embreex instead of pyembree for fast ray queries).

Trains an MLP to predict occupancy (inside/outside) of a mesh from 3D
coordinates, comparing against ground truth computed by ray casting. Reports
IoU on "easy" (uniform in bounding box) and "hard" (near-surface) test points,
and renders a normal map of the learned isosurface via ray marching.

Example:
    uv run python scripts/fourier_occupancy.py \
        --mesh data/meshes/dragon.obj --embedding gauss --iters 10000
"""

import argparse
import functools
import json
import os
import time

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import optax
import trimesh
from jax import jit, random
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def as_mesh(scene_or_mesh):
    """Convert a possible trimesh.Scene to a single Trimesh."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    assert isinstance(scene_or_mesh, trimesh.Trimesh)
    return scene_or_mesh


def recenter_mesh(mesh):
    """Normalize vertices to fit in [0, 1]^3, centered at 0.5."""
    mesh.vertices -= mesh.vertices.mean(0)
    mesh.vertices /= onp.max(onp.abs(mesh.vertices))
    mesh.vertices = 0.5 * (mesh.vertices + 1.0)


def load_mesh(mesh_file, verbose=True):
    mesh = as_mesh(trimesh.load(mesh_file))
    if verbose:
        print(f'{mesh_file}: {mesh.vertices.shape[0]} vertices, '
              f'{mesh.faces.shape[0]} faces, watertight={mesh.is_watertight}')
    recenter_mesh(mesh)
    c0 = mesh.vertices.min(0) - 1e-3
    c1 = mesh.vertices.max(0) + 1e-3
    return mesh, (c0, c1)


def gt_fn(mesh, queries):
    """Ground-truth occupancy of query points via ray casting."""
    queries = onp.asarray(queries)
    return mesh.ray.contains_points(
        queries.reshape([-1, 3])).reshape(queries.shape[:-1])


# ---------------------------------------------------------------------------
# Test point generation (easy = uniform in bbox, hard = near surface)
# ---------------------------------------------------------------------------

def uniform_bary(u):
    su0 = onp.sqrt(u[..., 0])
    b0 = 1.0 - su0
    b1 = u[..., 1] * su0
    return onp.stack([b0, b1, 1.0 - b0 - b1], -1)


def sample_surface_pts(mesh, n, rng):
    face_inds = rng.integers(0, mesh.faces.shape[0], [n])
    barys = uniform_bary(rng.uniform(size=[n, 2]))
    faces = mesh.faces[face_inds]
    pts = onp.sum(mesh.vertices[faces] * barys[..., None], 1)
    return pts


def make_test_pts(mesh, corners, rng, test_size):
    c0, c1 = corners
    test_easy = rng.uniform(size=[test_size, 3]) * (c1 - c0) + c0
    test_hard = (sample_surface_pts(mesh, test_size, rng)
                 + rng.normal(size=[test_size, 3]) * 0.01)
    return test_easy, test_hard


# ---------------------------------------------------------------------------
# Network (plain-JAX replacement for jax.experimental.stax)
# ---------------------------------------------------------------------------

def init_mlp(key, in_dim, num_layers, num_channels):
    sizes = [in_dim] + [num_channels] * (num_layers - 1) + [1]
    glorot = jax.nn.initializers.glorot_normal()
    params = []
    for i in range(len(sizes) - 1):
        key, sub = random.split(key)
        params.append({'W': glorot(sub, (sizes[i], sizes[i + 1])),
                       'b': jnp.zeros(sizes[i + 1])})
    return params


def apply_mlp(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer['W'] + layer['b'])
    return x @ params[-1]['W'] + params[-1]['b']


def input_encoder(x, avals, bvals):
    """Fourier feature mapping of x in [0,1]^3; identity if no bvals."""
    if bvals is None:
        return x * 2.0 - 1.0
    return jnp.concatenate(
        [avals * jnp.sin((2.0 * jnp.pi * x) @ bvals.T),
         avals * jnp.cos((2.0 * jnp.pi * x) @ bvals.T)],
        axis=-1) / jnp.linalg.norm(avals)


def make_bvals(embedding, embedding_size, scale, rng):
    if embedding == 'gauss':
        bvals = jnp.array(rng.normal(size=[embedding_size, 3]) * scale)
    elif embedding == 'posenc':
        bvals = 2.0 ** jnp.linspace(0, scale, embedding_size // 3) - 1
        bvals = jnp.reshape(jnp.eye(3) * bvals[:, None, None],
                            [len(bvals) * 3, 3])
    elif embedding == 'basic':
        bvals = jnp.eye(3)
    elif embedding == 'none':
        return None, None
    else:
        raise ValueError(f'unknown embedding {embedding}')
    return jnp.ones_like(bvals[:, 0]), bvals


# ---------------------------------------------------------------------------
# Rendering (ray-marched normal map of the learned isosurface)
# ---------------------------------------------------------------------------

trans_t = lambda t: jnp.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=jnp.float32)

rot_phi = lambda phi: jnp.array([
    [1, 0, 0, 0],
    [0, jnp.cos(phi), -jnp.sin(phi), 0],
    [0, jnp.sin(phi), jnp.cos(phi), 0],
    [0, 0, 0, 1]], dtype=jnp.float32)

rot_theta = lambda th: jnp.array([
    [jnp.cos(th), 0, -jnp.sin(th), 0],
    [0, 1, 0, 0],
    [jnp.sin(th), 0, jnp.cos(th), 0],
    [0, 0, 0, 1]], dtype=jnp.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * jnp.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * jnp.pi) @ c2w
    return c2w


def get_rays(H, W, focal, c2w):
    i, j = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing='xy')
    dirs = jnp.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -jnp.ones_like(i)], -1)
    rays_d = jnp.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return jnp.stack([rays_o, rays_d], 0)


@functools.partial(jit, static_argnums=(6, 7, 8))
def render_rays(params, ab, rays, corners, near, far,
                N_samples, N_samples_2, clip):
    """Two-pass hierarchical ray marching of the thresholded occupancy."""
    rays_o, rays_d = rays[0], rays[1]
    c0, c1 = corners
    th = 0.5

    def march(z_vals):
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        alpha = jax.nn.sigmoid(jnp.squeeze(
            apply_mlp(params, input_encoder(0.5 * (pts + 1), *ab))))
        if clip:
            mask = jnp.logical_or(jnp.any(0.5 * (pts + 1) < c0, -1),
                                  jnp.any(0.5 * (pts + 1) > c1, -1))
            alpha = jnp.where(mask, 0.0, alpha)
        alpha = jnp.where(alpha > th, 1.0, 0.0)
        trans = 1.0 - alpha + 1e-10
        trans = jnp.concatenate(
            [jnp.ones_like(trans[..., :1]), trans[..., :-1]], -1)
        weights = alpha * jnp.cumprod(trans, -1)
        depth_map = jnp.sum(weights * z_vals, -1)
        acc_map = jnp.sum(weights, -1)
        return depth_map, acc_map

    # Coarse pass over the full [near, far] range.
    depth_map, acc_map = march(jnp.linspace(near, far, N_samples))
    # Fine pass in a narrow band around the coarse surface estimate.
    z_vals = jnp.linspace(-1.0, 1.0, N_samples_2) * 0.01 + depth_map[..., None]
    return march(z_vals)


@jit
def make_normals(rays, depth_map):
    rays_o, rays_d = rays
    pts = rays_o + rays_d * depth_map[..., None]
    dx = pts - jnp.roll(pts, -1, axis=0)
    dy = pts - jnp.roll(pts, -1, axis=1)
    normal_map = jnp.cross(dx, dy)
    return normal_map / jnp.maximum(
        jnp.linalg.norm(normal_map, axis=-1, keepdims=True), 1e-5)


def render_mesh_normals(mesh, rays):
    """Ground-truth normal map by ray casting the actual mesh."""
    origins, dirs = onp.asarray(rays).reshape([2, -1, 3])
    origins = origins * 0.5 + 0.5
    dirs = dirs * 0.5
    z = mesh.ray.intersects_first(origins, dirs)
    pic = onp.zeros([origins.shape[0], 3])
    pic[z != -1] = mesh.face_normals[z[z != -1]]
    return pic.reshape(rays.shape[1:])


def render_normal_map(params, ab, rays, corners, near, far,
                      N_samples, N_samples_2, row_batch=16):
    """Render the network's normal map in batches of image rows."""
    H = rays.shape[1]
    rets = []
    for i in tqdm(range(0, H, row_batch), desc='render'):
        rets.append(render_rays(params, ab, rays[:, i:i + row_batch],
                                corners, near, far,
                                N_samples, N_samples_2, True))
    depth_map = jnp.concatenate([r[0] for r in rets], 0)
    return onp.asarray(make_normals(rays, depth_map))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def iou(pred, gt):
    pred, gt = pred > 0.5, gt > 0.5
    return onp.logical_and(pred, gt).sum() / onp.logical_or(pred, gt).sum()


def run_training(args, mesh, corners, test_pts, rng):
    c0, c1 = corners
    key = random.PRNGKey(args.seed)

    avals, bvals = make_bvals(args.embedding, args.embedding_size,
                              args.scale, rng)
    ab = (avals, bvals)
    in_dim = input_encoder(jnp.ones([1, 3]), *ab).shape[-1]
    print(f'embedding: {args.embedding}, encoded input dim: {in_dim}')

    params = init_mlp(key, in_dim, args.layers, args.channels)
    schedule = optax.exponential_decay(args.lr, 5000, 0.1)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    @jit
    def network_pred(params, inputs):
        return jax.nn.sigmoid(jnp.squeeze(
            apply_mlp(params, input_encoder(inputs, *ab))))

    @jit
    def loss_fn(params, inputs, z):
        x = jnp.squeeze(apply_mlp(params, input_encoder(inputs, *ab))[..., 0])
        # numerically stable sigmoid binary cross-entropy on logits
        return jnp.mean(jnp.maximum(x, 0) - x * z
                        + jnp.log(1 + jnp.exp(-jnp.abs(x))))

    @jit
    def step_fn(params, opt_state, inputs, outputs):
        loss, g = jax.value_and_grad(loss_fn)(params, inputs, outputs)
        updates, opt_state = optimizer.update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    history = {'iter': [], 'loss': [], 'iou_easy': [], 'iou_hard': []}
    gt_test = [gt_fn(mesh, t) for t in test_pts]

    t0 = time.time()
    for i in tqdm(range(args.iters + 1), desc='train'):
        inputs = rng.uniform(size=[args.batch_size, 3]) * (c1 - c0) + c0
        params, opt_state, loss = step_fn(
            params, opt_state, jnp.array(inputs), jnp.array(gt_fn(mesh, inputs)))

        if i % args.eval_every == 0:
            ious = [iou(onp.asarray(network_pred(params, t)), g)
                    for t, g in zip(test_pts, gt_test)]
            history['iter'].append(i)
            history['loss'].append(float(loss))
            history['iou_easy'].append(float(ious[0]))
            history['iou_hard'].append(float(ious[1]))
            tqdm.write(f'iter {i:6d}  loss {float(loss):.5f}  '
                       f'IoU easy {ious[0]:.4f}  hard {ious[1]:.4f}')
    print(f'training took {time.time() - t0:.1f}s')

    return params, ab, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Fourier feature network 3D occupancy')
    p.add_argument('--mesh', default='data/meshes/dragon.obj')
    p.add_argument('--embedding', default='gauss',
                   choices=['gauss', 'posenc', 'basic', 'none'])
    p.add_argument('--embedding-size', type=int, default=256)
    p.add_argument('--scale', type=float, default=12.0,
                   help='gauss: stdev of frequencies; posenc: max log2 freq')
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64 * 64 * 2 * 4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--test-size', type=int, default=2 ** 18)
    p.add_argument('--render-res', type=int, default=512)
    p.add_argument('--render-samples', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--outdir', default='occupancy_logs')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = onp.random.default_rng(args.seed)
    name = (f'{os.path.splitext(os.path.basename(args.mesh))[0]}'
            f'_{args.embedding}_{args.scale:g}')

    mesh, corners = load_mesh(args.mesh)
    test_pts = make_test_pts(mesh, corners, rng, args.test_size)

    params, ab, history = run_training(args, mesh, corners, test_pts, rng)

    # Final IoU on freshly drawn test points.
    final_pts = make_test_pts(mesh, corners, rng, args.test_size)
    pred_fn = jit(lambda x: jax.nn.sigmoid(jnp.squeeze(
        apply_mlp(params, input_encoder(x, *ab)))))
    scores = {}
    for label, pts in zip(['easy', 'hard'], final_pts):
        scores[label] = float(iou(onp.asarray(pred_fn(jnp.array(pts))),
                                  gt_fn(mesh, pts)))
    print(f'final IoU: {scores}')
    with open(os.path.join(args.outdir, name + '_scores.json'), 'w') as f:
        json.dump({'scores': scores, 'history': history, 'args': vars(args)},
                  f, indent=2)

    # Render learned and ground-truth normal maps.
    R = 2.0
    c2w = pose_spherical(90.0 + 10 + 45, -30.0, R)
    H = W = args.render_res
    focal = H * 0.9
    rays = get_rays(H, W, focal, c2w[:3, :4])
    corners_j = (jnp.array(corners[0]), jnp.array(corners[1]))

    normal_map = render_normal_map(params, ab, rays, corners_j, R - 1, R + 1,
                                   args.render_samples, args.render_samples)
    gt_normal_map = render_mesh_normals(mesh, rays)

    for label, img in [('pred', normal_map), ('gt', gt_normal_map)]:
        out = (255 * onp.clip(0.5 * img + 0.5, 0, 1)).astype(onp.uint8)
        path = os.path.join(args.outdir, f'{name}_normals_{label}.png')
        imageio.imsave(path, out)
        print(f'wrote {path}')

    # Cross-section figure: GT vs predicted occupancy on the z=0.5 slice.
    N = 256
    x = onp.linspace(0.0, 1.0, N, endpoint=False)
    grid = onp.stack(onp.meshgrid(x, x, indexing='ij'), -1)
    queries = onp.concatenate(
        [grid, 0.5 + onp.zeros_like(grid[..., :1])], -1)
    slice_gt = gt_fn(mesh, queries)
    slice_pred = onp.asarray(pred_fn(jnp.array(queries.reshape(-1, 3)))
                             ).reshape(N, N)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (img, title) in zip(axes, [
            (slice_gt, 'ground truth'), (slice_pred, 'prediction'),
            (onp.abs(slice_pred - slice_gt), 'error')]):
        im = ax.imshow(img)
        ax.set_title(f'{title} (z=0.5 slice)')
        fig.colorbar(im, ax=ax)
    path = os.path.join(args.outdir, name + '_slice.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
