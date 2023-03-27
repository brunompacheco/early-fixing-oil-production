from pathlib import Path
import pickle
from warnings import warn
import dgl
import numpy as np
from torch.utils.data import Dataset
import torch

from dgl.data import DGLDataset

from src.model import encode_fixing, get_C_GL, get_model


class EarlyFixingDataset(Dataset):
    def __init__(self, ef_objs, wells_dir='/home/bruno/gef-fs/data/raw') -> None:
        wells_dir = Path(wells_dir)

        super().__init__()

        self.wells = list(ef_objs.keys())

        self.q_gl_maxs = dict()
        self.well_data = dict()
        self.objs = list()
        self.samples_per_well = dict()
        self.targets = dict()
        for well_name in ef_objs.keys():
            with open(wells_dir/(well_name + '.pkl'), 'rb') as f:
                well = pickle.load(f)

            self.q_gl_maxs[well_name] = list({q_gl_max for (_, _, _, _, q_gl_max), _ in ef_objs[well_name].items()})

            self.targets[well_name] = dict()

            # define ordering for samples
            max_objs = {q_gl_max: -1 for q_gl_max in self.q_gl_maxs[well_name]}
            for (c1, c2, gl1, gl2, q_gl_max), obj in ef_objs[well_name].items():
                if obj > max_objs[q_gl_max]:
                    max_objs[q_gl_max] = obj

                    c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), well_name)

                    self.targets[well_name][q_gl_max] = (c_mbd, gl_mbd, obj)
            
            # sync q_gl_maxs
            self.q_gl_maxs[well_name] = list(self.targets[well_name].keys())

            if len(self.targets[well_name].keys()) == 0:
                warn('Well '+well_name+' had no feasible points')
                self.targets.pop(well_name, None)
                self.wells.remove(well_name)
                continue

            C, GL = get_C_GL(well_name)

            # q_liq_fun = -1 * np.ones((len(C), len(GL)))
            # for i in range(len(C)):
            #     for j in range(len(GL)):
            #         try:
            #             q_liq_fun[i,j] = well['curve'][C[i], GL[j]]
            #         except KeyError:
            #             q_liq_fun[i,j] = -1.

            # self.well_data[well_name] = (q_liq_fun, well['bsw'], well['gor'])
            self.well_data[well_name] = (well['bsw'], well['gor'])

    def __len__(self):
        return sum(len(q_gl_maxs) for q_gl_maxs in self.q_gl_maxs.values())

    def __getitem__(self, idx):
        if idx < 0:
            i = len(self) + idx
        else:
            i = idx  # sample index

        for well_name in self.wells:
            n_q_gl_maxs = len(self.targets[well_name].values())
            if i < n_q_gl_maxs:
                break
            else:
                i -= n_q_gl_maxs
        
        well_i = self.wells.index(well_name)
        
        q_gl_max = self.q_gl_maxs[well_name][i]

        # q_liq_fun, bsw, gor = self.well_data[well_name]
        bsw, gor = self.well_data[well_name]
        c_mbd, gl_mbd, obj = self.targets[well_name][q_gl_max]

        # return (q_liq_fun, bsw, gor, q_gl_max), (c_mbd, gl_mbd, obj, well_i)
        return (bsw, gor, q_gl_max), (c_mbd, gl_mbd, obj, well_i)

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
            for (c1, c2, gl1, gl2, q_gl_max), y in ef_objs[well_name].items():
                c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), well_name)

                well_ef_objs['x'].append((c_mbd, gl_mbd, q_gl_max))
                # well_ef_objs['x'].append((np.array([c1, c2]), np.array([gl1, gl2])))
                well_ef_objs['y'].append(y)
            self.ef_objs.append(well_ef_objs)

            C, GL = get_C_GL(well_name)

            # q_liq_fun = -1 * np.ones((len(C), len(GL)))
            # for i in range(len(C)):
            #     for j in range(len(GL)):
            #         try:
            #             q_liq_fun[i,j] = well['curve'][C[i], GL[j]]
            #         except KeyError:
            #             q_liq_fun[i,j] = -1.

            # self.well_data.append((q_liq_fun, well['bsw'], well['gor']))
            self.well_data.append((well['bsw'], well['gor']))

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

        # q_liq_fun, bsw, gor = self.well_data[n]
        bsw, gor = self.well_data[n]

        c_mbd, gl_mbd, q_gl_max = self.ef_objs[n]['x'][i]
        y = self.ef_objs[n]['y'][i]

        # return (q_liq_fun, bsw, gor, c_mbd, gl_mbd, q_gl_max), y
        return (bsw, gor, c_mbd, gl_mbd, q_gl_max), y
