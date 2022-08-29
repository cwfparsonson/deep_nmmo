import numpy as np
import random
import torch

import importlib

import pathlib
import glob


def seed_stochastic_modules_globally(default_seed=0, 
                                     numpy_seed=None, 
                                     random_seed=None,
                                     torch_seed=None,
                                     dgl_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed

    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    if torch_seed is None:
        torch_seed = default_seed
    if dgl_seed is None:
        dgl_seed = default_seed

    np.random.seed(numpy_seed)

    random.seed(random_seed)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_class_from_path(path):
    '''
    Path must be the path to the class **without** the .py extension.

    E.g. deep_nmmo.module_name.ModuleClass
    '''
    ClassName = path.split('.')[-1]
    path_to_class = '.'.join(path.split('.')[:-1])
    module = __import__(path_to_class, fromlist=[ClassName])
    return getattr(module, ClassName)

def get_module_from_path(path):
    '''
    Path must be the path to the module **without** the .py extension.

    E.g. deep_nmmo.module_name
    '''
    return importlib.import_module(path)

def gen_unique_experiment_folder(path_to_save, experiment_name):
    # init highest level folder
    path = path_to_save + '/' + experiment_name + '/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # init folder for this experiment
    path_items = glob.glob(path+'*')
    ids = sorted([int(el.split('_')[-1]) for el in path_items])
    if len(ids) > 0:
        _id = ids[-1] + 1
    else:
        _id = 0
    foldername = f'{experiment_name}_{_id}/'
    pathlib.Path(path+foldername).mkdir(parents=True, exist_ok=False)

    return path + foldername
