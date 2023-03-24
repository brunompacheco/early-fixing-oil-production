import numpy as np
import torch
import torch.nn

from src.net import InstanceGCN, ObjSurrogate, Fixer
from src.trainer import FeasibilityTrainer, ObjectiveSurrogateTrainer, EarlyFixingTrainer
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

    # net = ObjSurrogate(layers=[40, 20, 10, 10, 10]).double()
    # FeasibilityTrainer(
    #     net,
    #     'ef_objs.pkl',
    #     lr=.01,
    #     epochs=25,
    #     wandb_project=wandb_project,
    #     random_seed=seed,
    #     device=device,
    # ).run()

    # net = ObjSurrogate(layers=[40, 20, 20, 20, 20])
    # # net = load_from_wandb(net, 'anfmnk7e', 'gef-fs')
    # ObjectiveSurrogateTrainer(
    #     net.double(),
    #     'ef_objs.pkl',
    #     lr=.001,
    #     epochs=50,
    #     wandb_project=wandb_project,
    #     random_seed=seed,
    #     device=device,
    # ).run()

    surrogate = load_from_wandb(ObjSurrogate(layers=[40, 20, 20, 20, 20]), 's8whau88', 'gef-fs').double()
    EarlyFixingTrainer(
        Fixer(layers=[100, 100, 100, 100, 100]).double(),
        surrogate,
        'ef_objs.pkl',
        lr=.01,
        epochs=100000,
        wandb_project=wandb_project,
        random_seed=seed,
        device=device,
    ).run()
