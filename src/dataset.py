import numpy as np
from torch.utils.data import Dataset

from src.model import encode_fixing, get_model


class EarlyFixingDataset(Dataset):
    def __init__(self, ef_objs) -> None:
        super().__init__()

        self.Ab = dict()
        self.samples_per_well = dict()
        self.ef_objs = dict()
        for n in ef_objs.keys():
            # define ordering for samples
            self.ef_objs[n] = {'x': list(), 'y': list(),}

            for (c1, c2, gl1, gl2), y in ef_objs[n].items():
                c_mbd, gl_mbd = encode_fixing((c1,c2), (gl1, gl2), n)

                self.ef_objs[n]['x'].append((c_mbd, gl_mbd))
                self.ef_objs[n]['y'].append(y)

            # get A, b from wells
            model = get_model([n,])

            A = model.getA().toarray()
            b = np.array(model.getAttr('rhs'))

            self.Ab[n] = (A,b)

    def __len__(self):
        return sum(len(n_ef_objs['y']) for n_ef_objs in self.ef_objs.values())

    def __getitem__(self, idx):
        wells = list(self.ef_objs.keys())

        if idx < 0:
            i = len(self) + idx
        else:
            i = idx  # sample index

        for n in wells:
            n_samples = len(self.ef_objs[n]['y'])
            if i < n_samples:
                break
            else:
                i -= n_samples

        A, b = self.Ab[n]
        c_mbd, gl_mbd = self.ef_objs[n]['x'][i]
        y = self.ef_objs[n]['y'][i]

        return (A, b, c_mbd, gl_mbd), y
