import pickle

from src.model import fix_c_gl, get_model
from src.wells import Q_LIQ_N_FUN, WELLS


if __name__ == '__main__':
    # Prepara estruturas de dados para armazenamento dos valores das curvas de produção e RGL de poço
    C = {n: set() for n in WELLS}          # CKP
    GL = {n: set() for n in WELLS}         # Gás Lift

    # Percorre as curvas de produção e preenche as estruturas criadas para armazenamento de cada grandeza
    for n in WELLS:
        for (ckp, gl), ql in Q_LIQ_N_FUN[n].items():
            C[n].add(ckp)
            GL[n].add(gl)
            # Set_Qliq_n[n].add(ql)
        # Converte estruturas em listas ordenadas
        C[n] = sorted(list(C[n]))
        GL[n] = sorted(list(GL[n]))
        # Set_Qliq_n[n] = sorted(list(Set_Qliq_n[n]))

    ef_objs = dict()

    for n in WELLS:
        print(n)

        model = get_model([n,])

        ef_objs[n] = dict()

        for i in range(len(C[n]) - 1):
            cs = [C[n][i], C[n][i+1]]
            for j in range(len(GL[n]) - 1):
                gls = [GL[n][j], GL[n][j+1]]

                fixed_model = fix_c_gl(model, cs, gls)
                fixed_model.setParam("TimeLimit", 300)
                fixed_model.update()
                fixed_model.optimize()

                if fixed_model.status != 2:
                    ef_objs[n][tuple(cs+gls)] = -1
                else:
                    ef_objs[n][tuple(cs+gls)] = fixed_model.ObjVal

    with open('ef_objs.pkl', 'wb') as f:
        pickle.dump(ef_objs, f)
