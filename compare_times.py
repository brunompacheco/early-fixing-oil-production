import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
from tqdm import tqdm

from src.model import decode_fixing, fix_c_gl, get_C_GL, get_model, warm_start_c_gl
from src.net import Fixer
from src.utils import load_from_wandb


if __name__ == '__main__':
    with open('ef_objs.pkl', 'rb') as f:
        ef_objs = pickle.load(f)

    wells = list(ef_objs.keys())

    np.random.seed(42)
    test_is = np.random.choice(len(wells), int(0.2 * len(wells)), replace=False)
    test_wells = [wells[i] for i in test_is]
    ef_objs_test = {w: ef_objs[w] for w in test_wells}

    fixer = load_from_wandb(Fixer(layers=[25, 25]), 'm95c8k43', 'gef-fs')
    fixer.eval()

    baseline_times = list()
    fix_times = list()
    ef_times = list()
    for well_name in tqdm(list(ef_objs_test.keys())):
        with open(Path('data/raw')/(well_name + '.pkl'), 'rb') as f:
            well = pickle.load(f)

        q_gl_maxs = {q_gl_max for (_, _, _, _, q_gl_max), _ in ef_objs_test[well_name].items()}
        for q_gl_max in q_gl_maxs:
            model = get_model([well,], q_gl_max=q_gl_max)
            fixed_model = model.copy()

            start = time()
            model.optimize()
            baseline_times.append(time() - start)

            with torch.no_grad():
                bsw = torch.Tensor([well['bsw'],])
                gor = torch.Tensor([well['gor'],])
                q_gl_max_ = torch.Tensor([q_gl_max,])

                start = time()
                y_hat = torch.softmax(fixer(bsw, gor, q_gl_max_), -1)
                fix_times.append(time() - start)

                c_mbd_hat, gl_mbd_hat = (y_hat > .5).numpy().astype(int)[0]

            cs_hat, gls_hat = decode_fixing(c_mbd_hat, gl_mbd_hat, well)

            fixed_model = fix_c_gl(fixed_model, cs_hat, gls_hat)
            # fixed_model = warm_start_c_gl(fixed_model, cs_hat, gls_hat)
            fixed_model.update()

            start = time()
            fixed_model.optimize()
            ef_times.append(time() - start)

    print('Baseline average time = ', sum(baseline_times) / len(baseline_times), '(\sigma = ', np.std(baseline_times),')')
    print('Model average runtime = ', sum(fix_times) / len(fix_times), '(\sigma = ', np.std(fix_times),')')
    print('Early-fixed average time = ', sum(ef_times) / len(ef_times), '(\sigma = ', np.std(ef_times),')')

    # with open('reduced_ef_objs.pkl', 'wb') as f:
    #     pickle.dump(ef_objs, f)
