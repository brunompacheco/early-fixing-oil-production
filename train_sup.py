import numpy as np
import torch
import torch.nn

from src.net import Fixer
from src.trainer import SupervisedEarlyFixingTrainer
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        import random
        seed = 33
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)

        wandb_project = None  # avoid logging run

        torch.autograd.set_detect_anomaly(True)
    else:
        seed = None
        wandb_project = 'gef-fs'

    for _ in range(5):
        SupervisedEarlyFixingTrainer(
            Fixer(layers=[25, 25]).double(),
            'ef_objs.pkl',
            lr=.001,
            epochs=100,
            wandb_project=wandb_project,
            wandb_group='Supervised',
            random_seed=seed,
            device=device,
        ).run()
