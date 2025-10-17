import torch
from utils import geo_utils, general_utils, sparse_utils, plot_utils
from utils.Phases import Phases
import numpy as np
import networkx as nx



def is_valid_sample(data, min_pts_per_cam=10, phase=Phases.TRAINING):
    '''
    data : SceneData object, min_pts_per_cam: minimum number of points per camera
    phase : current phase (optimization/training/validation/testing/fine-tuning/short-optimization)
    returns True if the sample is valid, False otherwise
    '''
    if phase is Phases.TRAINING:
        return data.x.pts_per_cam.min().item() >= min_pts_per_cam
    else:
        return True


def divide_indices_to_train_test(N, n_val, n_test=0):
    perm = np.random.permutation(N)
    test_indices = perm[:n_test] if n_test>0 else []
    val_indices = perm[n_test:n_test+n_val]
    train_indices = perm[n_test+n_val:]
    return train_indices, val_indices, test_indices


def sample_indices(N, num_samples, adjacent):
    if num_samples == 1:  # Return all the data
        indices = np.arange(N)
    else:
        if num_samples < 1:  # fraction
            num_samples = int(np.ceil(num_samples * N))
        num_samples = max(2, num_samples)
        if num_samples >= N:
            return np.arange(N)
        if adjacent:
            start_ind = np.random.randint(0,N-num_samples+1)
            end_ind = start_ind+num_samples
            indices = np.arange(start_ind, end_ind)
        else:
            indices = np.random.choice(N,num_samples,replace=False)
    return indices


def save_cameras(outputs, conf, curr_epoch, phase):
    xs = outputs['xs']
    M = geo_utils.xs_to_M(xs)
    general_utils.save_camera_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)

def save_outliers(outputs, conf, curr_epoch, phase):
    if curr_epoch is None:
        general_utils.save_outliers_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)

# def save_metrics(outputs, conf, curr_epoch, phase):
#     general_utils.save_outliers_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)


def get_data_statistics(all_data, outputs=None):
    valid_pts = all_data.valid_pts
    valid_pts_stat = valid_pts.sum(dim=0).float()
    stats = {"Max_2d_pt": all_data.M.max().item(), "Num_2d_pts": valid_pts.sum().item(), "n_pts": all_data.M.shape[-1],
             "pts_per_cam_mean":  valid_pts.sum(dim=1).float().mean().item(), "Cameras_per_pts_mean": valid_pts_stat.mean().item(), "Cameras_per_pts_std": valid_pts_stat.std().item(),
             "Num of cameras": all_data.y.shape[0]}

    if outputs is not None:
        dict_info = all_data.dict_info.copy()
        dict_info.pop("outliers_pred", None)  # safely remove if exists
        stats.update(dict_info)

    return stats


def correct_matches_global(M, Ps, Ns):
    """
    This function corrects the matches using global triangulation.
    Args:
        M (torch.Tensor): A tensor of shape (N, 2), where N is the number of matches. ## correct this line
        Ps (torch.Tensor): A tensor of shape (N, 3, 4), where N is the number of cameras.
        Ns (torch.Tensor): A tensor of shape (N, 3, 3), where N is the number of cameras.
    Returns:
        xs (torch.Tensor): A tensor of shape M
    """

    # First, we get the invalid points in M.

    M_invalid_pts = np.logical_not(get_M_valid_points(M))

    # Next, we perform global triangulation to get the corrected matches.

    Xs = geo_utils.n_view_triangulation(Ps, M, Ns)
    xs = geo_utils.batch_pflat((Ps @ Xs))[:, 0:2, :]

    # Finally, we remove the invalid points from xs.

    xs[np.isnan(xs)] = 0
    # xs[np.stack((M_invalid_pts, M_invalid_pts), axis=1)] = 0
    xs = xs.reshape(M.shape)
    xs = torch.tensor(xs)
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] // 2, 2).transpose(0, 1)
    xs[M_invalid_pts] = 0
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] * 2).transpose(0, 1)

    return xs.numpy()



def get_M_valid_points(M):
    """
    Get a boolean matrix indicating valid points in M.
    A valid point is defined as one that is observed by at least two cameras.
    Args:
        M (torch.Tensor or np.ndarray): A tensor or array of shape (2 * num_views, num_points).
    Returns:
        M_valid_pts (torch.Tensor or np.ndarray): A boolean tensor or array of shape (num_views, num_points),
        where True indicates that the point is valid (observed by at least two cameras).
    """
    n_pts = M.shape[-1]

    if type(M) is torch.Tensor:
        M_valid_pts = torch.abs(M.reshape(-1, 2, n_pts)).sum(dim=1) != 0 # zero point
        M_valid_pts[:, M_valid_pts.sum(dim=0) < 2] = False  # mask out point tracks that contain only 1 point (viewed only from one camera-f)
    else:
        M_valid_pts = np.abs(M.reshape(-1, 2, n_pts)).sum(axis=1) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < 2] = False

    return M_valid_pts





def M2sparse(M, normalize=False, Ns=None, M_original=None, features=None):
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)  # [n_pts, 1]
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)  # [n_cams, 1]
    mat_indices = torch.nonzero(valid_pts).T  # [2, the number of points in the scene]
    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        mat_vals = norm_M[mat_indices[0], mat_indices[1], :]
    else:
        mat_vals = M.reshape(n_cams, 2, n_pts).transpose(1, 2)[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    
    return sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape)

# Descriptor functionality added:
def MandS2sparse(M, S, normalize=False, Ns=None, M_original=None, features=None):
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)  # [n_pts, 1]
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)  # [n_cams, 1]
    mat_indices = torch.nonzero(valid_pts).T  # [2, the number of points in the scene]
    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        mat_vals = norm_M[mat_indices[0], mat_indices[1], :]
    else:
        mat_vals = M.reshape(n_cams, 2, n_pts).transpose(1, 2)[mat_indices[0], mat_indices[1], :]
    
    s_values = S[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    s_shape = (n_cams, n_pts, S.shape[-1])
    
    m = sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape) # A sparse representation of M
    s = sparse_utils.SparseMat(s_values, mat_indices, cam_per_pts, pts_per_cam, s_shape) # A sparse representation of S
    return m, s

def get_M_view_adjacency(M):
    """
    Calculates the view adjacency matrix from  M.

    Args:
        M: A tracks tensor (2 * num_views, num_points).

    Returns:
        view_graph_adj: A torch.Tensor of shape (num_views, num_views) representing the view adjacency matrix,
                       where view_graph_adj[i, j] is the number of shared visible points between views i and j.
    """
    view_graph_adj = torch.zeros([M.shape[0] // 2, M.shape[0] // 2], dtype=torch.int32, device=M.device)
    M_valid_pts = get_M_valid_points(M)
    for i in range(M_valid_pts.shape[0]):
        for j in range(i + 1, M_valid_pts.shape[0]):
            num_shared_points = torch.logical_and(M_valid_pts[i], M_valid_pts[j]).sum()
            view_graph_adj[i, j] = num_shared_points
            view_graph_adj[j, i] = num_shared_points

    return view_graph_adj


def check_if_M_connected(M, thr=1, return_largest_component=False, returnAll=False):
    """
    Check connectivity of the camera-point view graph derived from the visibility matrix M.

    Args:
        M (Tensor): [2m, n] binary visibility matrix.
        thr (int): Minimum number of shared points to consider a connection between views.
        return_largest_component (bool): If True, return the largest connected component.
        returnAll (bool): If True, return all connected components.

    Returns:
        bool or (bool, List[int]) or List[Set[int]] depending on flags:
            - If no flags: returns is_connected (bool)
            - If return_largest_component: returns (is_connected, largest_component)
            - If returnAll: returns list of all components
    """
    import networkx as nx
    import numpy as np

    # Get adjacency matrix of M
    view_graph_adj = get_M_view_adjacency(M) # [m, m] tensor
    view_graph_adj = view_graph_adj.detach().cpu().numpy()

    # Create binary adjacency graph based on threshold
    view_graph = nx.from_numpy_array((view_graph_adj >= thr).astype(int))

    # Check overall connectivity
    connected = nx.is_connected(view_graph)

    # Extract connected and biconnected components
    components = sorted(nx.connected_components(view_graph), key=len, reverse=True)
    #biconnected_components = sorted(nx.biconnected_components(view_graph), key=len, reverse=True)

    # print(f"Component sizes (sorted): {[len(comp) for comp in components]}")
    # print(f"The graph has {len(components)} components")


    if returnAll:
        return components
    if return_largest_component:
        largest_cc = components[0] if components else []
        return connected, list(largest_cc)

    return connected

