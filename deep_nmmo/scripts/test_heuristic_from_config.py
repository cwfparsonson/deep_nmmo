import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from deep_nmmo.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
import os

import time
import pickle
import gzip


@hydra.main(config_path='configs', config_name='heuristic_config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' in cfg.experiment:
        seed_stochastic_modules_globally(cfg.experiment.seed)
        print(f'Seeded with seed={cfg.experiment.seed}')

    # create dir for saving data
    save_dir = gen_unique_experiment_folder(path_to_save=cfg.experiment.path_to_save, experiment_name=cfg.experiment.name)
    cfg['experiment']['save_dir'] = save_dir
    print(f'Created save directory {save_dir}')

    # init weights and biases
    if 'wandb' in cfg:
        if cfg.wandb is not None:
            import wandb
            cfg.wandb.init.dir = cfg.experiment.path_to_save
            wandb.init(**cfg.wandb.init)
            wandb.confg = cfg
        else:
            wandb = None
    else:
        wandb = None

    # save copy of config to the save dir
    OmegaConf.save(config=cfg, f=save_dir+'heuristic_config.yaml')

    # print info
    print('\n\n\n')
    print(f'~'*100)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*100)

    env_loop = hydra.utils.instantiate(cfg.env_loop, wandb=wandb)
    print(f'Initialised {env_loop}.')

    start_time = time.time()
    print(f'Launching run of {cfg.experiment.num_runs} episode(s)...')
    results = env_loop.run(verbose=False, seed=cfg.experiment.env_seed, num_episodes=cfg.experiment.num_runs)
    print(f'Finished run of {cfg.experiment.num_runs} episode(s) in {time.time() - start_time:.3f} s')
    update_log_start_time = time.time()
    for result in results:
        env_loop.update_log(**result)
        time.sleep(0.25)
    update_log_time = time.time() - update_log_start_time
    print(f'Updated log in {update_log_time:.3f} s')

if __name__ == '__main__':
    run()
