import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.model import fix_c_gl, get_model, get_C_GL


if __name__ == '__main__':
    wells_fps = Path('data/raw').glob('*.pkl')

    # Prepara estruturas de dados para armazenamento dos valores das curvas de produção e RGL de poço
    C, GL = get_C_GL(None)

    ef_objs = dict()

    for well_fp in tqdm(list(wells_fps)):
        with open(well_fp, 'rb') as f:
            well = pickle.load(f)

        well_name = well_fp.name.removesuffix('.pkl')

        ef_objs[well_name] = dict()

        q_gl_maxs = 1e5 + np.random.rand(100) * 2e5
        for q_gl_max in q_gl_maxs:
            try:
                model = get_model([well,], q_gl_max=q_gl_max)
            except ZeroDivisionError:
                print('Zero division errror in ', well_name)
                ef_objs.pop(well_name, None)
                break

            for i in range(len(C) - 1):
                for j in range(len(GL) - 1):
                    cs_to_fix = [C[i], C[i+1]]
                    gls_to_fix = [GL[j], GL[j+1]]

                    fixed_model = fix_c_gl(model, cs_to_fix, gls_to_fix)
                    fixed_model.setParam("TimeLimit", 300)
                    fixed_model.update()
                    fixed_model.optimize()

                    if fixed_model.status != 2:
                        ef_objs[well_name][tuple(cs_to_fix+gls_to_fix+[q_gl_max,])] = -1
                    else:
                        ef_objs[well_name][tuple(cs_to_fix+gls_to_fix+[q_gl_max,])] = fixed_model.ObjVal

    with open('ef_objs.pkl', 'wb') as f:
        pickle.dump(ef_objs, f)
