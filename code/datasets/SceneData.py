import torch
from utils import geo_utils, dataset_utils, sparse_utils
from datasets import  Euclidean
import os.path
from pyhocon import ConfigFactory
import numpy as np
import warnings

# from time import time



class SceneData:
    def __init__(self, M, Ns, Ps_gt, scan_name, dilute_M=False, outliers=None, dict_info=None, nameslist=None, M_original=None, S=None, s=None, s_dict=None):
        '''
        M is [2m, n], Ns is [m, 3, 3], Ps_gt is [m, 3, 4] (torch.tensor)s
        M is the Veiws-Poitns matrix
        Ns is the intrinsics matrices
        Ps_gt is the ground truth camera matrices
        scan_name is the name of the scene (string)

        dilute_M: whether to dilute M by removing points which are seen by small number of cameras
        this removes columns of M that have less than # of non-zero entries

        outliers is a list of length m, each element is a 1D tensor of outlier indices for that camera
        dict_info is a dictionary of additional information about the scene
        nameslist is a list of image names of length m
        M_original is the original M before any filtering
        
        This class holds the data for a single scene
        It also prepares some additional useful attributes:
        self.M : the Veiws-Poitns matrix (torch.tensor of shape [2m, n])
        self.Ns_invT : inverse transpose of Ns (torch.tensor of shape [m, 3, 3])
        self.dict_info : dictionary of additional information about the scene (dict)
        self.img_list : ndarray of image names as strings (ndarray of shape (m,))
        self.norm_M : normalized M - with metric coordinates rather than pixel coordinates (torch.tensor of shape [2m, n])
        self.outlier_indices : list of outlier indices for each camera (torch.tensor of shape [m, n])
        self.scan_name : name of the scene (string)
        self.valid_pts : a boolean matrix indicating valid points in M (torch.tensor of shape [m, n])
        self.x : a sparse matrix representation of M (sparse_utils.SparseMat of shape [m, n, 2])
        self.y : the ground truth camera matrices (torch.tensor of shape [m, 3, 4])

        Added Properties:
        self.S : A Descriptors (SIFT in this case) matrix for each point (torch.tensor of shape [m, n, 128])
        '''
        
        if M_original is None:
            M_original = M.detach().clone()

        # Dilute M
        if dilute_M:
            M = geo_utils.dilutePoint(M)

        n_images = Ps_gt.shape[0]

        # Set attribute
        self.scan_name = scan_name
        self.y = Ps_gt
        self.M = M
        self.M_original = M_original
        self.Ns = Ns
        self.outlier_indices = outliers

        # Descriptor functionality added:
        if S is not None:
            self.x, self.s = dataset_utils.MandS2sparse(M, S.to_dense(), normalize=True, Ns=Ns, M_original=M_original)
            self.S = None
        else:
            self.S = None
            # M to sparse matrix
            self.x = dataset_utils.M2sparse(M, normalize=True, Ns=Ns, M_original=M_original)

        if s_dict is not None:
            indices_to_keep = list(s_dict.values())
            self.s = sparse_utils.SparseMat(s.values[indices_to_keep], self.x.indices, self.x.cam_per_pts, self.x.pts_per_cam, self.x.shape)

        # Get image list
        if nameslist is None:
            self.img_list = torch.arange(n_images)
        else:
            self.img_list = nameslist

        # Prepare Ns inverse transpose
        self.Ns_invT = torch.transpose(torch.inverse(Ns), 1, 2)

        # Get valid points
        self.valid_pts = dataset_utils.get_M_valid_points(M)

        # Normalize M
        self.norm_M = geo_utils.normalize_M(M, Ns, self.valid_pts).transpose(1, 2).reshape(n_images * 2, -1)

        # Stats of the scene
        self.dict_info = dict_info



    def to(self, *args, **kwargs):
        for key in self.__dict__:
            if not key.startswith('__'):
                attr = getattr(self, key)
                if isinstance(attr, sparse_utils.SparseMat) or torch.is_tensor(attr):
                    setattr(self, key, attr.to(*args, **kwargs))

        return self


def create_scene_data(conf, phase=None):
    # Init
    scan = conf.get_string('dataset.scan')
    calibrated = conf.get_bool('dataset.calibrated')
    dilute_M = conf.get_bool('dataset.diluteM', default=False)


    # Get raw data
    if calibrated:
        M, Ns, Ps_gt, outliers, dict_info, namesList, M_original, S = Euclidean.get_raw_data(conf, scan, phase)
    else:
        raise ValueError("The code doesn't support the uncalibrated case")

    return SceneData(M, Ns, Ps_gt, scan, dilute_M, outliers=outliers, dict_info=dict_info, nameslist=namesList, M_original=M_original, S=S)


def sample_data(data:SceneData, num_samples, adjacent=True):
    """For a given scene, randomly sample num_samples cameras (rows), adjacent or not.
    Note: when the requested num_samples is more than available cameras, all cameras will be returned"""

    # Get indices
    indices = dataset_utils.sample_indices(len(data.y), num_samples, adjacent=adjacent)
    M_indices = np.sort(np.concatenate((2 * indices, 2 * indices + 1)))

    indices = torch.from_numpy(indices).squeeze() # an indices tensor of shape (num_samples,)
    M_indices = torch.from_numpy(M_indices).squeeze() # an indices tensor of shape (2*num_samples,)

    # Get sampled data
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    outlier_indices = data.outlier_indices[indices]
    outlier_indices = outlier_indices[:, (M > 0).sum(dim=0) > 2]

    # Taking only 3D points which are seen by at least two views
    PointsMask = (M > 0).sum(dim=0) > 2
    M = M[:, PointsMask]

    point_indices = torch.nonzero(PointsMask, as_tuple=True)[0]
    s_dict = filter_s_by_indices(data.s, indices, point_indices)

    # Descriptor functionality added:
    # if data.S is not None:
    #     beg_time = time()
    #     S = data.S.to_dense()
    #     end_time = time()
    #     print(f'{end_time-beg_time:.1f}')
    #     S = S[indices]
    #     S = S[:, PointsMask,:]
    #     S = S.to_sparse()
    # else:
    #     S = None

    sampled_data = SceneData(M, Ns, y, data.scan_name,outliers=outlier_indices, nameslist=data.img_list[indices], s = data.s, s_dict = s_dict)
    if (sampled_data.x.pts_per_cam == 0).any():
        warnings.warn('Cameras with no points for dataset '+ data.scan_name)

    return sampled_data


def create_scene_data_from_list(scan_names_list, conf):
    data_list = []
    for scan_name in scan_names_list:
        conf["dataset"]["scan"] = scan_name
        data = create_scene_data(conf)
        data_list.append(data)

    return data_list


def test_data(data:SceneData, conf):
    import loss_functions

    # Test Losses of GT and random on data
    repLoss = loss_functions.ESFMLoss(conf)
    cams_gt = prepare_cameras_for_loss_func(data.y, data)
    cams_rand = prepare_cameras_for_loss_func(torch.rand(data.y.shape), data)

    print("Loss for GT: Reprojection = {}".format(repLoss(cams_gt, data)))
    print("Loss for rand: Reprojection = {}".format(repLoss(cams_rand, data)))


def prepare_cameras_for_loss_func(Ps, data):
    Vs_invT = Ps[:, 0:3, 0:3]
    Vs = torch.inverse(Vs_invT).transpose(1, 2)
    ts = torch.bmm(-Vs.transpose(1, 2), Ps[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
    pts_3D = torch.from_numpy(geo_utils.n_view_triangulation(Ps.numpy(), data.M.numpy(), data.Ns.numpy())).float()
    return {"Ps": torch.bmm(data.Ns, Ps), "pts3D": pts_3D}


def get_subset(data, subset_size):
    # Get subset indices
    valid_pts = dataset_utils.get_M_valid_points(data.M)
    n_cams = valid_pts.shape[0]

    first_idx = valid_pts.sum(dim=1).argmax().item()
    curr_pts = valid_pts[first_idx].clone()
    valid_pts[first_idx] = False
    indices = [first_idx]

    for i in range(subset_size - 1):
        shared_pts = curr_pts.expand(n_cams, -1) & valid_pts
        next_idx = shared_pts.sum(dim=1).argmax().item()
        curr_pts = curr_pts | valid_pts[next_idx]
        valid_pts[next_idx] = False
        indices.append(next_idx)

    print("Cameras are:")
    print(indices)

    indices = torch.sort(torch.tensor(indices))[0]
    M_indices = torch.sort(torch.cat((2 * indices, 2 * indices + 1)))[0]
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    M = M[:, (M > 0).sum(dim=0) > 2]
    return SceneData(M, Ns, y, data.scan_name + "_{}".format(subset_size), outliers=data.outlier_indices[indices], dict_info=data.dict_info, nameslist=data.img_list[indices])


def filter_s_by_x(s,x):
    
    # beg_time = time()

    keys_s = list(zip(s.indices[0].tolist(), s.indices[1].tolist()))
    keys_x = list(zip(x.indices[0].tolist(), x.indices[1].tolist()))
    dict_s = {key: index for index, key in enumerate(keys_s)}
    dict_x = {key: index for index, key in enumerate(keys_x)}
    keys_s = set(dict_s.keys())
    keys_x = set(dict_x.keys())
    common_keys = set(dict_s.keys()) & set(dict_x.keys())
    common_indices = torch.tensor([dict_s[key] for key in common_keys])
    s = sparse_utils.SparseMat(s.values[common_indices], x.indices,  x.cam_per_pts, x.pts_per_cam, x.shape)

    # print(f"{time()-beg_time:.3f}")

    return s


def filter_s_by_indices(s, m_indices, n_indices):
    
    # beg_time = time()

    keys_s = list(zip(s.indices[0].tolist(), s.indices[1].tolist()))
    dict_s = {key: index for index, key in enumerate(keys_s)}
    
    m_indices_set = set(m_indices.tolist())
    n_indices_set = set(n_indices.tolist())
    
    dict_s = {key: value for key, value in dict_s.items() if key[0] in m_indices_set}
    dict_s = {key: value for key, value in dict_s.items() if key[1] in n_indices_set}

    # print(f"{time()-beg_time:.3f}")

    return dict_s
    

# if __name__ == "__main__":
    # test_dataset()

