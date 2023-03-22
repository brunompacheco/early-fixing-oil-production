from pathlib import Path
import pickle
from warnings import warn
import dgl
import numpy as np
from torch.utils.data import Dataset
import torch

from dgl.data import DGLDataset

from src.model import encode_fixing, get_C_GL, get_model
from src.wells import Q_LIQ_N_FUN, bsw_n, gor_n


class EarlyFixingDataset(Dataset):
    def __init__(self, ef_objs, wells_dir='/home/bruno/gef-fs/data/raw') -> None:
        wells_dir = Path(wells_dir)

        super().__init__()

        self.wells = list(ef_objs.keys())

        self.well_data = list()
        self.objs = list()
        self.samples_per_well = dict()
        self.targets = dict()
        for well_name in ef_objs.keys():
            with open(wells_dir/(well_name + '.pkl'), 'rb') as f:
                well = pickle.load(f)

            # define ordering for samples
            max_obj = -1
            for (c1, c2, gl1, gl2), obj in ef_objs[well_name].items():
                if obj > max_obj:
                    max_obj = obj

                    c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), well_name)

                    self.targets[well_name] = (c_mbd, gl_mbd, obj)
            if well_name not in self.targets.keys():
                warn('Well '+well_name+' had no feasible points')
                self.wells.remove(well_name)
                continue

            C, GL = get_C_GL(well_name)

            q_liq_fun = -1 * np.ones((len(C), len(GL)))
            for i in range(len(C)):
                for j in range(len(GL)):
                    try:
                        q_liq_fun[i,j] = well['curve'][C[i], GL[j]]
                    except KeyError:
                        q_liq_fun[i,j] = -1.

            self.well_data.append((q_liq_fun, well['bsw'], well['gor']))

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, idx):
        well_name = self.wells[idx]

        q_liq_fun, bsw, gor = self.well_data[idx]
        c_mbd, gl_mbd, obj = self.targets[well_name]

        return (q_liq_fun, bsw, gor), (c_mbd, gl_mbd, obj)

class WellObjDataset(Dataset):
    def __init__(self, ef_objs, wells_dir='/home/bruno/gef-fs/data/raw') -> None:
        wells_dir = Path(wells_dir)
        super().__init__()

        self.well_data = list()
        self.samples_per_well = dict()
        self.ef_objs = list()
        for well_name in ef_objs.keys():
            with open(wells_dir/(well_name + '.pkl'), 'rb') as f:
                well = pickle.load(f)

            # define ordering for samples
            well_ef_objs = {'x': list(), 'y': list(),}
            for (c1, c2, gl1, gl2), y in ef_objs[well_name].items():
                c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), well_name)

                well_ef_objs['x'].append((c_mbd, gl_mbd))
                # well_ef_objs['x'].append((np.array([c1, c2]), np.array([gl1, gl2])))
                well_ef_objs['y'].append(y)
            self.ef_objs.append(well_ef_objs)

            C, GL = get_C_GL(well_name)

            q_liq_fun = -1 * np.ones((len(C), len(GL)))
            for i in range(len(C)):
                for j in range(len(GL)):
                    try:
                        q_liq_fun[i,j] = well['curve'][C[i], GL[j]]
                    except KeyError:
                        q_liq_fun[i,j] = -1.

            self.well_data.append((q_liq_fun, well['bsw'], well['gor']))

        self.wells = list(ef_objs.keys())

    def __len__(self):
        return sum(len(n_ef_objs['y']) for n_ef_objs in self.ef_objs)

    def __getitem__(self, idx):
        if idx < 0:
            i = len(self) + idx
        else:
            i = idx  # sample index

        for n in range(len(self.wells)):
            n_samples = len(self.ef_objs[n]['y'])
            if i < n_samples:
                break
            else:
                i -= n_samples

        q_liq_fun, bsw, gor = self.well_data[n]

        c_mbd, gl_mbd = self.ef_objs[n]['x'][i]
        y = self.ef_objs[n]['y'][i]

        return (q_liq_fun, bsw, gor, c_mbd, gl_mbd), y

class EarlyFixingGraphDataset(DGLDataset):
    def __init__(self, ef_objs) -> None:
        super().__init__(name='EarlyFixing')

        self.gs = dict()
        self.samples_per_well = dict()
        self.ef_objs = dict()
        self.wells = list()
        for n in ef_objs.keys():
            self.wells.append(n)

            # get model's graph
            model = get_model([n,], avoid_infeasible=True)

            eta_vars = np.array([v.varname.startswith('eta_n') for v in model.getVars()])

            A = model.getA()
            z_cons = A[:,eta_vars].T
            x_cons = A[:,~eta_vars].T

            eta_c_vars = np.array([v.varname.startswith('eta_n_c') for v in model.getVars()])
            eta_c_vars = eta_c_vars[eta_vars]
            eta_gl_vars = np.array([v.varname.startswith('eta_n_gl') for v in model.getVars()])
            eta_gl_vars = eta_gl_vars[eta_vars]

            self.ef_objs[n] = {'g': list(), 'y': list()}

            for (c1, c2, gl1, gl2), y in ef_objs[n].items():
                # self.ef_objs[n]['x'].append((np.array([c1, c2]), np.array([gl1, gl2])))

                c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), n)
                # self.ef_objs[n]['x'].append((c_mbd, gl_mbd))

                c_mbd = np.pad(c_mbd, (0,1))
                c_mbd += np.roll(c_mbd, 1)

                gl_mbd = np.pad(gl_mbd, (0,1))
                gl_mbd += np.roll(gl_mbd, 1)

                g = dgl.heterograph({
                    ('z', 'z2c', 'c'): z_cons.nonzero(),
                    ('x', 'x2c', 'c'): x_cons.nonzero(),
                })
                eweights = {
                    'z2c': torch.Tensor(z_cons.data),
                    'x2c': torch.Tensor(x_cons.data) / 1e5,
                }
                nfeats = {
                    'z': torch.Tensor(np.array(model.getAttr('obj'))[eta_vars]),
                    'x': torch.Tensor(np.array(model.getAttr('obj'))[~eta_vars]),
                    'c': torch.Tensor(model.getAttr('rhs')) / 1e5,
                }

                fixing_feats = torch.zeros_like(nfeats['z'])
                fixing_feats[eta_c_vars] = torch.Tensor(c_mbd)
                fixing_feats[eta_gl_vars] = torch.Tensor(gl_mbd)

                nfeats['z'] = torch.vstack([
                    nfeats['z'],
                    fixing_feats,
                ]).T

                g.ndata['feat'] = nfeats
                g.edata['weight'] = eweights

                self.ef_objs[n]['g'].append(g)
                self.ef_objs[n]['y'].append(y)

    def __len__(self):
        return sum(len(n_ef_objs['y']) for n_ef_objs in self.ef_objs.values())

    def __getitem__(self, idx):
        if idx < 0:
            i = len(self) + idx
        else:
            i = idx  # sample index

        for n in self.wells:
            n_samples = len(self.ef_objs[n]['y'])
            if i < n_samples:
                break
            else:
                i -= n_samples

        g = self.ef_objs[n]['g'][i]
        y = self.ef_objs[n]['y'][i]

        return g, y
