# %% - Imports

import cv2  # DO NOT REMOVE
import torch


from utils import general_utils_test, dataset_utils, path_utils
# from utils.general_utils_test import init_exp
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet, collate_fn
from datasets import SceneData
# from single_scene_optimization_Tests import train_single_model
import train
import copy
from time import time

# from lightning.fabric import Fabric

# fabric = Fabric(accelerator="cuda", devices="auto", strategy='ddp_notebook')
# fabric.launch()


def print_memory():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        
        # Total memory
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        
        # Used memory
        used_memory = torch.cuda.memory_allocated()
        
        # Reserved memory (may include free memory)
        reserved_memory = torch.cuda.memory_reserved()
        
        # Free memory can be calculated as:
        free_memory = reserved_memory - used_memory

        # Convert to megabytes for easier readability
        total_memory_mb = total_memory / (1024 ** 2)
        used_memory_mb = used_memory / (1024 ** 2)
        reserved_memory_mb = reserved_memory / (1024 ** 2)
        free_memory_mb = free_memory / (1024 ** 2)

        print(f"Total GPU memory: {total_memory_mb:.2f} MB")
        print(f"Used GPU memory: {used_memory_mb:.2f} MB")
        print(f"Reserved GPU memory: {reserved_memory_mb:.2f} MB")
        print(f"Free GPU memory: {free_memory_mb:.2f} MB")
    else:
        print("CUDA is not available. Please check your GPU setup.")

print('\n-------------------------------------')
print('--------- Test file Started ---------')
print('-------------------------------------\n')

# %% - Configuration initialization form conf file

# Init Experiment
conf, device, phase = general_utils_test.init_exp(Phases.TRAINING.name)
general_utils_test.log_code(conf) # Log code to the experiment folder (you can comment this line if you don't want to log the code)`

# Set device
# Get configuration
min_sample_size = conf.get_float('dataset.min_sample_size')
max_sample_size = conf.get_float('dataset.max_sample_size')
batch_size = conf.get_int('dataset.batch_size')
optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
optimization_lr = conf.get_float('train.optimization_lr')

# %% - Test: Creating SceneData obj - Changes Passed this Test Block
scene_to_scan = '0007_300 new'
conf["dataset"]["scan"] = scene_to_scan
SceneData_0007 = SceneData.create_scene_data(conf, phase=None)
# print(SceneData_0007)

# %% - Test: Creating list of SceneData obj's - Changes Passed this Test Block
data_scenes_list = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)
# print(data_scenes_list)

# %% - Test: Creating a ScenesDataSet obj
train_set = ScenesDataSet(data_scenes_list, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size, phase=Phases.TRAINING)
Sliced_SceneData_0007 = train_set[0]

# %% - Test: Feeding to SetOfSetBlock
from models import SetOfSet
d_in = 2 ; d_out = 256
Equiv_Block = SetOfSet.SetOfSetBlock(d_in, d_out, conf)
Equiv_Block(Sliced_SceneData_0007.x)


print('\n-------------------------------------')
print('---------- Test file Ended ----------')
print('-------------------------------------\n')

# %% - Test: GPU Memory printing
current_device = torch.cuda.current_device()
total_memory = torch.cuda.get_device_properties(current_device).total_memory
total_memory_mb = total_memory / (1024 ** 2)  # Convert bytes to megabytes
print(f"Total GPU memory: {total_memory_mb:.2f} MB")

# %% - Test: Filtering using missing indices

# Example tensor I_1 with integers
I_1 = torch.randint(0, 10000, (2, 100000), dtype=torch.int)
I_2 = torch.randint(0, 481, (2, 1012), dtype=torch.int)

beg_time = time()
# Step 1: Extract the two rows and create tuples
keys_1 = list(zip(I_1[0].tolist(), I_1[1].tolist()))
keys_2 = list(zip(I_2[0].tolist(), I_2[1].tolist()))

# Step 2: Create the dictionary using a dictionary comprehension
dict_1 = {key: index for index, key in enumerate(keys_1)}
dict_2 = {key: index for index, key in enumerate(keys_2)}

keys_1 = set(dict_1.keys())
keys_2 = set(dict_2.keys())

# Step 2: Find missing keys in dict_b
missing_keys = keys_1 - keys_2

# Step 3: Collect missing values from dict_a
# missing_values = [dict_1[key] for key in missing_keys]

# Step 1: Identify the common keys
common_keys = set(dict_1.keys()) & set(dict_2.keys())

# Step 2: Collect values from dict_a for the common keys
common_values = [dict_1[key] for key in common_keys]

print("Common values in dict_a for keys also in dict_b:", common_values)

# print("Resulting dictionary:", result_dict)
print(f"{time()-beg_time:.3f}")

# --------- Side Tests --------------
# %% - Indexing test:
'''
# Example tensors
Tens = torch.randn(10, 10, 4)  # Shape: (100, 2000, 300)
Ind = torch.randint(0, 5, (5,))  # Example indices tensor of shape (50,)

# Use advanced indexing to sample Tens according to Ind
sampled_tensor = Tens[Ind]  # This will give you a tensor of shape (50, 2000, 300)

# Check the shape of the resulting tensor
print("Sampled tensor shape:", sampled_tensor.shape)  # Output: (50, 2000, 300)
'''
# %%
'''
if phase is not Phases.FINE_TUNE:
    # Train model
    model = general_utils_test.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    print(f'Number of parameters: {sum([x.numel() for x in model.parameters()])}')
    print(f'Number of trainable parameters:: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Create train, test and validation sets
    test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
    validation_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.validation_set'), conf)
    train_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)

    train_set = ScenesDataSet(train_scenes, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size, phase=Phases.TRAINING)
    validation_set = ScenesDataSet(validation_scenes, return_all=True)
    test_set = ScenesDataSet(test_scenes, return_all=True)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
'''
# %%

