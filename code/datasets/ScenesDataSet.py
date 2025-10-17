import random

from datasets import SceneData
from utils import dataset_utils
import numpy as np


class ScenesDataSet:
    def __init__(self, data_list, return_all, min_sample_size=10, max_sample_size=30, phase=None):
        '''
        A dataset class that has an getitem method
            which returns either the full scene or a random
            sample of it (rows of M and corresponding entries of Ps, Ns, outliers, etc)

            Note:
                sample size is not a batch size!
                Batch size petantaially contains multiple scenes is handled by the dataloader.
        
        data_list : a list of SceneData objects
        return_all : if True, return the full scene, otherwise return a random sample
        min_sample_size, max_sample_size : if return_all is False, the sample size will be between these two values
        phase : current phase (optimization/training/validation/testing/fine-tuning/short-optimization)
        '''
        
        super().__init__()
        self.data_list = data_list # list of SceneData objects
        self.return_all = return_all # if True, return the full scene, otherwise return a random sample
        self.min_sample_size = min_sample_size # can be int (number of cameras) or float (fraction of cameras)
        self.max_sample_size = max_sample_size # can be int (number of cameras) or float (fraction of cameras)
        self.phase = phase # current phase (optimization/training/validation/testing/fine-tuning/short-optimization)

    def __getitem__(self, item):
        # Get scene according to the index 'item' and sample it if needed.
        # If sampling size > 1.0 - it is considered as number of cameras
        # If sampling size <= 1.0 - it is considered as fraction of cameras
        
        current_data = self.data_list[item]
        if self.return_all:
            return current_data
        else:
            if self.max_sample_size > 1.0:
                max_sample = min(self.max_sample_size, len(current_data.y))
                if self.min_sample_size >= max_sample:
                    sample_fraction = max_sample
                else:
                    sample_fraction = np.random.randint(self.min_sample_size, max_sample + 1)
            else:
                if len(current_data.y) < 50: # if there not enough views (50), use these fractions to ensure enough data is sampled.
                    self.max_sample_size = 1.0
                    self.min_sample_size = 0.4

                if self.min_sample_size > 1.0:
                    sample_fraction = int(random.uniform(self.min_sample_size,  min(self.max_sample_size * len(current_data.y),100)))
                else:
                    sample_fraction = int(random.uniform(self.min_sample_size * len(current_data.y),  min(self.max_sample_size * len(current_data.y), 100)))
            
            # Note: sample_fraction is an integer number of cameras to sample, not really a fraction.

            counter = 0
            while 1:
                data = SceneData.sample_data(current_data, sample_fraction)
                if dataset_utils.is_valid_sample(data, min_pts_per_cam=3, phase=self.phase) or counter > 0:
                    # Maybe to avoid infinite loop? Very wierd way to do it, might be an error.
                    return data
                counter += 1


    def __len__(self):
        return len(self.data_list)


def collate_fn(data):
    """
       default collate function for the dataset
    """
    return data


