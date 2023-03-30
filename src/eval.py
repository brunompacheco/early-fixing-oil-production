import pickle
import numpy as np
import torch
from tqdm import tqdm

from src.model import decode_fixing, get_model, fix_c_gl


def get_accuracy_gaps(fixer, ds):
    logits = list()
    q_gl_maxs = list()
    well_is = list()
    objs = list()
    z_cs = list()
    z_gls = list()
    for (bsw, gor, q_gl_max), (z_c, z_gl, obj, well_i) in tqdm(ds):
        with torch.no_grad():
            bsw = torch.Tensor([bsw,])
            gor = torch.Tensor([gor,])
            q_gl_max = torch.Tensor([q_gl_max,])

            logits.append(fixer(bsw, gor, q_gl_max))

            z_cs.append(z_c)
            z_gls.append(z_gl)
            q_gl_maxs.append(q_gl_max)
            well_is.append(well_i)
            objs.append(obj)
    logit = torch.vstack(logits)
    y_hat = torch.softmax(logit, -1).numpy()
    y_pred = torch.argmax(logit, -1).numpy()

    z_c = np.stack(z_cs)
    z_gl = np.stack(z_gls)
    q_gl_max = np.stack(q_gl_maxs)
    well_i = np.stack(well_is)
    obj = np.stack(objs)

    y_true = np.vstack([
        np.argmax(z_c, -1),
        np.argmax(z_gl, -1),
    ]).T

    hits = (y_true == y_pred).all(-1)
    acc = hits.sum() / len(hits)

    failed_mbds = (y_hat[~hits] > 0.5).astype(int)

    failed_q_gl_max = q_gl_max[~hits]
    failed_well_is = well_i[~hits]
    actual_obj = obj[~hits]

    failed_wells = [ds.wells[i] for i in failed_well_is]

    gaps = list()
    for k in range(len(failed_wells)):
        with open('../data/raw/'+failed_wells[k]+'.pkl', 'rb') as f:
            failed_well = pickle.load(f)
        failed_well_model = get_model([failed_well,], q_gl_max=failed_q_gl_max[k])

        c_mbd, gl_mbd = failed_mbds[k]

        c_pair, gl_pair = decode_fixing(c_mbd, gl_mbd, failed_well)
        failed_well_model = fix_c_gl(failed_well_model, c_pair, gl_pair)

        failed_well_model.optimize()

        if failed_well_model.status != 2:
            gaps.append(-1)
        else:
            gaps.append(actual_obj[k] - failed_well_model.ObjVal)

    return acc, gaps

def get_mae_feas(surrogate, ds):
    y_hats = list()
    bsws = list()
    gors = list()
    q_gl_maxs = list()
    well_is = list()
    ys = list()
    z_cs = list()
    z_gls = list()
    for (bsw, gor, z_c, z_gl, q_gl_max), y in tqdm(ds):
        with torch.no_grad():
            z_cs.append(z_c)
            z_gls.append(z_gl)
            bsws.append(bsw)
            gors.append(gor)
            q_gl_maxs.append(q_gl_max)
            ys.append(y)

            bsw = torch.Tensor([bsw,])
            gor = torch.Tensor([gor,])
            z_c = torch.Tensor([z_c,])
            z_gl = torch.Tensor([z_gl,])
            q_gl_max = torch.Tensor([q_gl_max,])

            y_hats.append(surrogate(bsw, gor, z_c, z_gl, q_gl_max))
    y_hat = torch.vstack(y_hats).squeeze().numpy()

    z_c = np.stack(z_cs)
    z_gl = np.stack(z_gls)
    bsw = np.stack(bsws)
    gor = np.stack(gors)
    q_gl_max = np.stack(q_gl_maxs)
    y = np.stack(ys)

    error = y_hat - y
    mae = np.abs(error[y >= 0]).mean()

    hits = (y_hat < 0) == (y < 0)
    acc = hits.sum() / len(hits)

    return mae, acc
