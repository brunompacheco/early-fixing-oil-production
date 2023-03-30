import numpy as np
import torch
import torch.nn

from src.net import ObjSurrogate, Fixer
from src.trainer import ObjectiveSurrogateTrainer, EarlyFixingTrainer
from src.utils import debugger_is_active, load_from_wandb


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
        surrogate = ObjectiveSurrogateTrainer(
            ObjSurrogate(layers=[10, 10, 10]).double(),
            'reduced_ef_objs.pkl',
            lr=.001,
            epochs=20,
            wandb_project=wandb_project,
            wandb_group='Semi-Supervised - Surrogate',
            random_seed=seed,
            device=device,
        ).run()

        surrogate.eval()
        fixer = EarlyFixingTrainer(
            Fixer(layers=[25, 25]).double(),
            surrogate,
            'ef_objs.pkl',
            lr=.01,
            epochs=100,
            wandb_project=wandb_project,
            wandb_group='Semi-Supervised - Fixer',
            random_seed=seed,
            device=device,
        ).run()
