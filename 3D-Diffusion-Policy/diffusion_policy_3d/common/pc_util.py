import torch
import numpy as np
from torch_cluster import fps, knn, radius
from typing import Optional, Tuple, Dict, Union, Literal

# Adapted from https://github.com/charlesq34/pointnet/blob/master/part_seg/test.py#L82
def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc


def depth2pc(
    depth: np.ndarray,
    camera_intrinsic_matrix: np.ndarray,
    bounding_box: Optional[Dict] = None,
    seg_mask: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if len(depth.shape) == 3:
        depth = depth.squeeze(-1)
    height, width = depth.shape
    fx = camera_intrinsic_matrix[0, 0]
    fy = camera_intrinsic_matrix[1, 1]
    cx = camera_intrinsic_matrix[0, 2]
    cy = camera_intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pc = np.stack([x, y, z], axis=-1)
    # Apply bounding box mask
    bb_mask = np.ones_like(z, dtype=bool)
    if bounding_box is not None:
        bb_mask = (
            (np.all(pc[..., :3] > bounding_box['lower_bound'], axis=-1))
            & (np.all(pc[..., :3] < bounding_box['upper_bound'], axis=-1))
        )
    pc = pc[bb_mask]
    if seg_mask is not None:
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.squeeze(-1)
        seg_mask = seg_mask[bb_mask]
    return pc, seg_mask


def random_point_sample(
    pc: torch.Tensor,
    ratio: Optional[float] = 0.3,
    mask: Union[torch.Tensor, None] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        pc: point cloud xyz (Shape: [B, N, 3]).
        ratio: the percentage of points to be sampled.
        mask: point cloud mask (Shape: [B, N]). If given, fps will be conducted
            in the region where mask==1.

    Returns:
        batch_idx: the batch indices of the sampled points (Shape: [N',]).
        point_idx: the point indices of the sampled points (Shape: [N',]).
        pc_sample: the sampled points flattened along batch dimension (Shape: [N', 3]).
    """
    batch_size, num_points = pc.shape[:2]
    device = pc.device
    batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_points).to(device)
    point_idx = torch.arange(num_points)[None, :].expand(batch_size, num_points).to(device)
    if mask is None:
        pc_flatten = pc.flatten(end_dim=1)
        batch_idx = batch_idx.reshape(-1)
        point_idx = point_idx.reshape(-1)
    else:
        mask = mask.bool()
        pc_flatten = pc[mask]
        batch_idx = batch_idx[mask]
        point_idx = point_idx[mask]
    # Count sample points per batch
    point_counts = torch.bincount(batch_idx, minlength=batch_size)
    num_samples = (point_counts.float() * ratio).long().clamp(min=1)
    # Random shuffle inside each batch
    rand = torch.rand_like(batch_idx.float())
    LARGE = batch_idx.numel() + 1
    order = torch.argsort(batch_idx * LARGE + rand)
    batch_idx_sort = batch_idx[order]
    # Compute batch rank
    batch_start = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
    batch_start[1:] = torch.cumsum(point_counts, dim=0)
    start_idx = batch_start[batch_idx_sort]
    batch_rank = torch.arange(batch_idx_sort.numel(), device=device) - start_idx
    # Sample points inside valid regions
    idx = order[batch_rank <= num_samples[batch_idx_sort]]
    batch_idx = batch_idx[idx]
    point_idx = point_idx[idx]
    pc_sample = pc_flatten[idx]
    return batch_idx, point_idx, pc_sample


def farthest_point_sample(
    pc: torch.Tensor,
    ratio: Optional[float] = 0.3,
    mask: Union[torch.Tensor, None] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        pc: point cloud xyz (Shape: [B, N, 3]).
        ratio: the percentage of points to be sampled.
        mask: point cloud mask (Shape: [B, N]). If given, fps will be conducted
            in the region where mask==1.

    Returns:
        batch_idx: the batch indices of the sampled points (Shape: [N',]).
        point_idx: the point indices of the sampled points (Shape: [N',]).
        pc_sample: the sampled points flattened along batch dimension (Shape: [N', 3]).
    """
    batch_size, num_points = pc.shape[:2]
    device = pc.device
    batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_points).to(device)
    point_idx = torch.arange(num_points)[None, :].expand(batch_size, num_points).to(device)
    if mask is None:
        pc_flatten = pc.flatten(end_dim=1)
        batch_idx = batch_idx.reshape(-1)
        point_idx = point_idx.reshape(-1)
    else:
        mask = mask.bool()
        pc_flatten = pc[mask]
        batch_idx = batch_idx[mask]
        point_idx = point_idx[mask]
    if ratio >= 1.0:
        pc_sample = pc_flatten
    else:
        # The returned fps indices range from 0 to B*N
        idx = fps(pc_flatten, batch_idx, ratio)
        batch_idx = batch_idx[idx]
        point_idx = point_idx[idx]
        # Gather sampled points
        pc_sample = pc[batch_idx, point_idx]
    return batch_idx, point_idx, pc_sample


def k_nearest_neighbors(
    pc_query: torch.Tensor,
    pc: torch.Tensor,
    batch_idx_query: Union[torch.Tensor, None] = None,
    mask: Union[torch.Tensor, None] = None,
    num_neighbors: Optional[int] = 5
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """
    Finds @num_neighbors neighbor points for each element in @pc_query.

    Args:
        pc_query: query points (Shape: [B, M, 3] or [N', 3]).
        pc: searching space (Shape: [B, N, 3]).
        batch_idx_query: the batch indices of the query points, used when pc_query.shape==[N', 3].
            Note that the indices must match the batch size of @pc.
        mask: point cloud mask (Shape: [B, N]). If given, neighbor points will be searched
            in the region where mask==1.

    Returns:
        batch_idx_query: batch indices of the query points (Shape: [N',]).
        point_idx_neighbor: point indices of the neighbor points (Shape: [N', @max_num_neighbors]).
    """
    batch_size, num_points = pc.shape[:2]
    device = pc.device
    batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_points).to(device)
    point_idx = torch.arange(num_points)[None, :].expand(batch_size, num_points).to(device)
    if mask is None:
        pc_flatten = pc.flatten(end_dim=1)
        batch_idx = batch_idx.reshape(-1)
        point_idx = point_idx.reshape(-1)
    else:
        mask = mask.bool()
        pc_flatten = pc[mask]
        batch_idx = batch_idx[mask]
        point_idx = point_idx[mask]

    if len(pc_query.shape) == 2:
        assert batch_idx_query is not None, "@batch_idx_query not provided in the arguments!"
        pc_query_flatten = pc_query
        batch_idx_query = batch_idx_query
    else:
        pc_query_flatten = pc_query.flatten(end_dim=1)
        batch_idx_query = torch.arange(batch_size).repeat_interleave(pc_query.shape[1]).to(device)
    idx = knn(
        x=pc_flatten,
        y=pc_query_flatten,
        k=num_neighbors,
        batch_x=batch_idx,
        batch_y=batch_idx_query
    )
    point_idx_neighbor = point_idx[idx[1]].reshape(-1, num_neighbors)
    return batch_idx_query, point_idx_neighbor


def ball_query(
    pc_query: torch.Tensor,
    pc: torch.Tensor,
    batch_idx_query: Union[torch.Tensor, None] = None,
    mask: Union[torch.Tensor, None] = None,
    ball_radius: Optional[float] = 0.2,
    max_num_neighbors: Optional[int] = 5
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """
    Finds for each element in @pc_query all points in @pc within distance @ball_radius and
    filters out those query points with neighbor points less than @max_num_neighbors

    Args:
        pc_query: query points (Shape: [B, M, 3] or [N', 3]).
        pc: searching space (Shape: [B, N, 3]).
        batch_idx_query: the batch indices of the query points, used when pc_query.shape==[N', 3].
            Note that the indices must match the batch size of @pc.
        mask: point cloud mask (Shape: [B, N]). If given, neighbor points will be searched
            in the region where mask==1.

    Returns:
        query_idx: global indices for the flattened @pc_query after filtering (Shape: [M',], M' < N').
        batch_idx_query: batch indices of the filtered query points (Shape: [M',]).
        point_idx_neighbor: point indices of the neighbor points (Shape: [M', @max_num_neighbors]).
    """
    batch_size, num_points = pc.shape[:2]
    device = pc.device
    batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_points).to(device)
    point_idx = torch.arange(num_points)[None, :].expand(batch_size, num_points).to(device)
    if mask is None:
        pc_flatten = pc.flatten(end_dim=1)
        batch_idx = batch_idx.reshape(-1)
        point_idx = point_idx.reshape(-1)
    else:
        mask = mask.bool()
        pc_flatten = pc[mask]
        batch_idx = batch_idx[mask]
        point_idx = point_idx[mask]

    if len(pc_query.shape) == 2:
        assert batch_idx_query is not None, "@batch_idx_query not provided in the arguments!"
        pc_query_flatten = pc_query
        batch_idx_query = batch_idx_query
    else:
        pc_query_flatten = pc_query.flatten(end_dim=1)
        batch_idx_query = torch.arange(batch_size).repeat_interleave(pc_query.shape[1]).to(device)
    
    idx = radius(
        x=pc_flatten,
        y=pc_query_flatten,
        r=ball_radius,
        batch_x=batch_idx,
        batch_y=batch_idx_query,
        max_num_neighbors=max_num_neighbors
    )
    query_idx, neighbor_idx = idx
    # Count neighbors per center point
    neighbor_counts = torch.bincount(query_idx, minlength=pc_query_flatten.shape[0])
    # Only those that have exactly @max_num_neighbors neighbors are kept
    is_valid = (neighbor_counts == max_num_neighbors)
    edge_mask = is_valid[query_idx]
    query_idx = query_idx[edge_mask]
    neighbor_idx = neighbor_idx[edge_mask]
    if query_idx.numel() == 0:
        print("WARNING: No query points have @max_num_neighbors neighbors!")
        return None
    # Sort so that neighbors of the same query are contiguous
    sort_order = torch.argsort(query_idx)
    query_idx = query_idx[sort_order]
    neighbor_idx = neighbor_idx[sort_order]
    # Remove repeated values in query indices
    assert query_idx.numel() % max_num_neighbors == 0
    query_idx = query_idx[::max_num_neighbors]
    # Obtain batch and point indices for neighbor points
    batch_idx_query = batch_idx_query[query_idx]
    point_idx_neighbor = point_idx[neighbor_idx].reshape(-1, max_num_neighbors)
    return query_idx, batch_idx_query, point_idx_neighbor


# def sample_and_group(
#     pc_xyz: torch.Tensor,
#     pc_feat: torch.Tensor,
#     pc_mask: Union[torch.Tensor, None] = None,
#     sample_method: Literal["random", "fps"] = "fps",
#     sample_ratio: Optional[float] = 0.3,
#     group_method: Literal["knn", "ball_query"] = "ball_query",
#     num_neighbors: Optional[int] = 10
# ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
#     """
#     Sample features in pc_mask==1 regions and look for neighbors in pc_mask==0 regions

#     Returns:
#         batch_idx_query: batch indices of @pc_feat_query (Shape: [M',]).
#         pc_feat_query: feature query (Shape: [M', D]).
#         pc_feat_neighbor: neighbor features of @pc_feat_query (Shape: [M', @num_neighbors, D]).
#     """
#     pos_key_mask = pc_mask
#     neg_key_mask = None if pc_mask is None else ~pc_mask
#     # Sample query points & features
#     if sample_method == "random":
#         batch_idx, point_idx, pc_xyz_query = random_point_sample(pc_xyz, sample_ratio, pos_key_mask)
#         pc_feat_query = pc_feat[batch_idx, point_idx]
#     elif sample_method == "fps":
#         batch_idx, point_idx, pc_xyz_query = farthest_point_sample(pc_xyz, sample_ratio, pos_key_mask)
#         pc_feat_query = pc_feat[batch_idx, point_idx]
#     else:
#         raise ValueError(f"Unsupported sampling method {sample_method}! Must be either 'random' or 'fps'.")
#     # Look for neighbor points & features in the "dictionary"
#     if group_method == "knn":
#         batch_idx_query, point_idx_neighbor = k_nearest_neighbors(pc_xyz_query, pc_xyz, batch_idx, neg_key_mask, num_neighbors)
#         pc_feat_neighbor = pc_feat[batch_idx_query[:, None], point_idx_neighbor]
#     elif group_method == "ball_query":
#         ret = ball_query(pc_xyz_query, pc_xyz, batch_idx, neg_key_mask, max_num_neighbors=num_neighbors)
#         if ret is None:
#             return ret
#         query_idx, batch_idx_query, point_idx_neighbor = ret
#         pc_feat_query = pc_feat_query[query_idx]
#         pc_feat_neighbor = pc_feat[batch_idx_query[:, None], point_idx_neighbor]
#     else:
#         raise ValueError(f"Unsupported grouping method {group_method}! Must be either 'knn' or 'ball_query'.")
    
#     return batch_idx_query, pc_feat_query, pc_feat_neighbor


def sample_and_group(
    pc_xyz: torch.Tensor,
    pc_feat: torch.Tensor,
    pc_mask: Union[torch.Tensor, None] = None,
    sample_method: Literal["random", "fps"] = "fps",
    sample_ratio: Optional[float] = 0.3,
    group_method: Literal["knn"] = "knn",
    num_pos_keys: Optional[int] = 1,
    num_neg_keys: Optional[int] = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample feature query in pc_mask==1 regions and look for positive and negative
    neighbors in pc_mask==1 and pc_mask==0 regions, respectively.

    Returns:
        batch_idx_query: (N',)
        pc_feat_query: (N', D)
        pc_feat_pos_keys: (N', @num_pos_keys, D)
        pc_feat_neg_keys: (N', @num_neg_keys, D)
    """
    pos_key_mask = pc_mask
    neg_key_mask = None if pc_mask is None else ~pc_mask
    # Sample query points & features
    if sample_method == "random":
        batch_idx, point_idx, pc_xyz_query = random_point_sample(pc_xyz, sample_ratio, pos_key_mask)
        pc_feat_query = pc_feat[batch_idx, point_idx]
    elif sample_method == "fps":
        batch_idx, point_idx, pc_xyz_query = farthest_point_sample(pc_xyz, sample_ratio, pos_key_mask)
        pc_feat_query = pc_feat[batch_idx, point_idx]
    else:
        raise ValueError(f"Unsupported sampling method {sample_method}! Must be either 'random' or 'fps'.")
    # Look for neighbor points & features in the "dictionary"
    if group_method == "knn":
        batch_idx_query, point_idx_neighbor = k_nearest_neighbors(pc_xyz_query, pc_xyz, batch_idx, pos_key_mask, num_pos_keys)
        pc_feat_pos_keys = pc_feat[batch_idx_query[:, None], point_idx_neighbor]
        batch_idx_query, point_idx_neighbor = k_nearest_neighbors(pc_xyz_query, pc_xyz, batch_idx, neg_key_mask, num_neg_keys)
        pc_feat_neg_keys = pc_feat[batch_idx_query[:, None], point_idx_neighbor]
    else:
        raise ValueError(f"Unsupported grouping method {group_method}!")
    
    return batch_idx_query, pc_feat_query, pc_feat_pos_keys, pc_feat_neg_keys


def visualize_in_original_pc(
    pc: torch.Tensor,          # (T, N, 3) or (B, N, 3)
    pc_query: torch.Tensor,    # (Q, 3)
    neighbors: torch.Tensor,   # (Q, K, 3)
    frame_id: int = 0,
    sphere_radius: float = 0.005
):
    """
    Visualize query points and neighbors overlaid on the original point cloud.
    """
    import open3d as o3d
    # -------------------------
    # Original point cloud (gray)
    # -------------------------
    pc_frame = pc[frame_id].detach().cpu().numpy()   # (N, 3)

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pc_frame)
    pcd_all.paint_uniform_color([0.7, 0.7, 0.7])  # gray

    # -------------------------
    # Neighbor points (green)
    # -------------------------
    neighbors_flat = neighbors.reshape(-1, 3).detach().cpu().numpy()

    pcd_neighbors = o3d.geometry.PointCloud()
    pcd_neighbors.points = o3d.utility.Vector3dVector(neighbors_flat)
    pcd_neighbors.paint_uniform_color([0.0, 1.0, 0.0])  # green

    # -------------------------
    # Query points (red spheres)
    # -------------------------
    pc_query_np = pc_query.detach().cpu().numpy()

    query_spheres = []
    for p in pc_query_np:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(p)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
        sphere.compute_vertex_normals()
        query_spheres.append(sphere)

    # -------------------------
    # Visualize
    # -------------------------
    o3d.visualization.draw_geometries(
        [pcd_all, pcd_neighbors, *query_spheres],
        window_name="Original PC (Gray) + Queries (Red) + Neighbors (Green)"
    )


# if __name__ == "__main__":
#     import zarr
#     with zarr.open("/Users/wangjl/Desktop/Projects/sim_demo_collector/demos/robosuite_stack.zarr", 'r') as f:
#         frame_id = 700
#         batch_size = 2
#         camera = "frontview"
#         pc = f[f'data/{camera}_pc'][()][frame_id: frame_id + batch_size]
#         pc_mask = f[f'data/{camera}_pc_mask'][()][frame_id: frame_id + batch_size]
#         pc = torch.from_numpy(pc).float()
#         pc_mask = torch.from_numpy(pc_mask).bool()

#     batch_idx, pc_query, neighbors = sample_and_group(pc, pc, pc_mask, sample_method="random", group_method="ball_query", num_neighbors=5)
#     visualize_in_original_pc(pc, pc_query[:3], neighbors[:3], frame_id=0)

if __name__ == "__main__":
    import zarr
    with zarr.open("/Users/wangjl/Desktop/Projects/sim_demo_collector/demos/robosuite_stack.zarr", 'r') as f:
        frame_id = 700
        batch_size = 2
        camera = "frontview"
        pc = f[f'data/{camera}_pc'][()][frame_id: frame_id + batch_size]
        pc_mask = f[f'data/{camera}_pc_mask'][()][frame_id: frame_id + batch_size]
        pc = torch.from_numpy(pc).float()
        pc_mask = torch.from_numpy(pc_mask).bool()

    batch_idx, query, pos_keys, neg_keys = sample_and_group(pc, pc, pc_mask)