from typing import List

import gurobipy as gp
from gurobipy import GRB

from src.wells import *


def get_model(wells: List[str], avoid_infeasible=True):
    N = wells
    M = [0,]  # manifolds

    ###############################################################################
    ### MANIPULAÇÕES DOS DADOS DE ENTRADA

    # Inicializa conjuntos de valores possíveis nas chaves da curva de queda de pressão
    B = set()
    L = set()
    P = {m: set() for m in M}
    Q = {m: set() for m in M}

    # Prepara estruturas de dados para armazenamento dos valores das curvas de produção e RGL de poço
    C = dict()          # CKP
    GL = dict()         # Gás Lift

    # Percorre as curvas de produção e preenche as estruturas criadas para armazenamento de cada grandeza
    for n in N:
        C[n], GL[n] = get_C_GL(n)

    # Prepara os conjuntos com breakpoints de Qliq e LGR
    Set_Qliq_n = {n: [] for n in N}      # Conjunto de breakpoints de Qliq
    Set_LGR_n = {n: [] for n in N}       # Conjunto de breakpoints de LGR      #[10 * k for k in range(51)]
    for n in N:
        min_lgr, max_lgr, max_ql = 10000, 0, 0
        for (ckp, gl), ql in Q_LIQ_N_FUN[n].items():
            gas_prod = ql * (1 - bsw_n[n]) * gor_n[n]
            gas_tot = gl + gas_prod
            lgr = gas_tot / ql

            min_lgr = min(min_lgr, lgr)
            max_lgr = max(max_lgr, lgr)
            max_ql = max(max_ql, ql)
        num_pontos = 48
        Set_Qliq_n[n] = [max_ql * k * 1.0/num_pontos for k in range(num_pontos+1)]
        lgr_step = (max_lgr - min_lgr)/num_pontos
        Set_LGR_n[n] = [min_lgr + k * lgr_step for k in range(num_pontos+1)]

    # Insere valor -1 nos pontos espúrios (chaves faltantes) das curvas de produção
    for n in N:
        expected_keys = [(c, gl) for c in C[n] for gl in GL[n]]
        for key in expected_keys:
            if key not in Q_LIQ_N_FUN[n]:
                Q_LIQ_N_FUN[n][key] = -1

    # inserve valor -1 nos pontos espúrios (chaves faltantes) das curvas de queda de pressão
    for m in M:
        expected_keys = [(p, r, b, q) for p in P[m] for r in L for b in B for q in Q[m]]
        # for key in expected_keys:
        #     if key not in DELTA_P_FUN[m]:
        #         DELTA_P_FUN[m][key] = -1

    ### FIM - MANIPULAÇÕES DOS DADOS DE ENTRADA
    ###############################################################################

    ###############################################################################
    ### MODELO MATEMÁTICO: DEFINIÇÃO DE VARIÁVEIS E CONJUNTOS

    # Create a new model
    model = gp.Model("flow_splitting")

    # Create variables
    ckp_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="ckp_n")
    q_oil_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="q_oil_n")
    q_water_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="q_water_n")
    q_gas_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="q_gas_n")
    q_gl_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="q_gl_n")
    q_liq_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="q_liq_n")
    #lgr_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="lgr_n")

    q_liq_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="q_liq_m")
    q_oil_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="q_oil_m")
    q_water_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="q_water_m")
    q_gas_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="q_gas_m")
    #bsw_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="bsw_m")
    #lgr_m = model.addVars(M, vtype=GRB.CONTINUOUS, name="lgr_m")


    #q_gas_n_m = model.addVars(N, M, vtype=GRB.CONTINUOUS, name="q_gas_n_m")

    # Indexing sets
    lmbd_n_c_gl_index = [(n, c, gl) for n in N for c in C[n] for gl in GL[n]]
    lmbd_n_c_gl = model.addVars(lmbd_n_c_gl_index, lb=0.0, vtype=GRB.CONTINUOUS, name='lmbd_n_c_gl')


    eta_n_c_index = [(n, c) for n in N for c in C[n]]
    eta_n_c = model.addVars(eta_n_c_index, vtype=GRB.CONTINUOUS, name='eta_n_c')

    eta_n_gl_index = [(n, gl) for n in N for gl in GL[n]]
    eta_n_gl = model.addVars(eta_n_gl_index, lb=0.0, vtype=GRB.CONTINUOUS, name='eta_n_gl')


    ### FIM - MODELO MATEMÁTICO: DEFINIÇÃO DE VARIÁVEIS E CONJUNTOS
    ###############################################################################

    ###############################################################################
    ### MODELO MATEMÁTICO: Restrições do modelo

    # PRODUÇÃO DOS POÇOS
    for n in N:
        # Ckp compatível com variáveis de combinação convexa
        model.addConstr(ckp_n[n] == gp.quicksum(lmbd_n_c_gl[n, c, gl] * c for c in C[n] for gl in GL[n])) 
        # Q_gl compatível com variáveis de combinação convexa
        model.addConstr(q_gl_n[n] == gp.quicksum(lmbd_n_c_gl[n, c, gl] * gl for c in C[n] for gl in GL[n]))  # 21b
        model.addConstr(q_liq_n[n] == gp.quicksum(lmbd_n_c_gl[n, c, gl] * Q_LIQ_N_FUN[n][c, gl] for c in C[n] for gl in GL[n]))  # 21c

    for n in N:
        for ckp in C[n]:
            for gl in GL[n]:            # !!! gl
                if avoid_infeasible:
                    if Q_LIQ_N_FUN[n][ckp, gl] == -1:
                        # Proibição de uso de pontos espúrios da curva de produção
                        model.addConstr(lmbd_n_c_gl[n, ckp, gl] <= 0)
                    else:
                        model.addConstr(lmbd_n_c_gl[n, ckp, gl] <= 1e9)
        for ckp in C[n]:
            # Acoplamento lambda eta
            model.addConstr(eta_n_c[n, ckp] == gp.quicksum(lmbd_n_c_gl[n, ckp, gl] for gl in GL[n]))  # 21g
        for gl in GL[n]:            # !!! gl
            # Acoplamento lambda eta
            model.addConstr(eta_n_gl[n, gl] == gp.quicksum(lmbd_n_c_gl[n, c, gl] for c in C[n]))  # 21h

        model.addSOS(GRB.SOS_TYPE2, [eta_n_c[n, c] for c in C[n]])  # 21i
        model.addSOS(GRB.SOS_TYPE2, [eta_n_gl[n, gl] for gl in GL[n]])  # 21j

        model.addConstr(gp.quicksum(eta_n_c[n, c] for c in C[n]) == 1.0)
        model.addConstr(gp.quicksum(eta_n_gl[n, gl] for gl in GL[n]) == 1.0)

    for n in N:
        # Associando fluidos ao q_liq através de bsw e rgo
        model.addConstr(q_oil_n[n] == q_liq_n[n] * (1 - bsw_n[n]))  # 22a
        model.addConstr(q_water_n[n] == q_liq_n[n] * bsw_n[n])  # 22b
        model.addConstr(q_gas_n[n] == q_liq_n[n] * (1 - bsw_n[n]) * gor_n[n])  # 22c


    for n in N:
        for m in M:
            model.addConstr(q_liq_m[m] <= q_liq_max)  # #todo: adicionar no modelo
            model.addConstr(q_gas_m[m] <= q_gas_max)  # #todo: adicionar no modelo

    # DIVISÃO DE FLUXOS
    for m in M:
        model.addConstr(q_gas_m[m] == gp.quicksum(q_gas_n[n] for n in N))  # 25f
        model.addConstr(q_liq_m[m] == gp.quicksum(q_liq_n[n] for n in N))  # 25g
        model.addConstr(q_oil_m[m] == gp.quicksum(q_liq_n[n] * (1 - bsw_n[n]) for n in N))  # 25h
        model.addConstr(q_water_m[m] == gp.quicksum(q_water_n[n] for n in N))  # 25i

    ### FIM - MODELO MATEMÁTICO: Restrições do modelo
    ###############################################################################

    ## Set objective
    model.setObjective(gp.quicksum(q_oil_n[n] for n in N), GRB.MAXIMIZE)

    model.setParam("IntFeasTol", 1e-9)
    # model.setParam("MipGap", 1e-4)
    # model.setParam("TimeLimit", 300)

    model.update()

    return model

def fix_c_gl(model, cs, gls):
    assert len(cs) <= 2
    assert len(gls) <= 2

    model_ = model.copy()

    cs = [float(c) for c in cs]
    gls = [float(gl) for gl in gls]

    for var in model_.getVars():
        if var.VarName.startswith('eta_n_c'):
            c = float(var.VarName.split(',')[-1].rstrip(']'))
            if c not in cs:
                var.ub = 0.0
        elif var.VarName.startswith('eta_n_gl'):
            gl = float(var.VarName.split(',')[-1].rstrip(']'))
            if gl not in gls:
                var.ub = 0.0

    model_.update()

    return model_

def get_C_GL(well):
    # Prepara estruturas de dados para armazenamento dos valores das curvas de produção e RGL de poço
    C = set()          # CKP
    GL = set()         # Gás Lift

    # Percorre as curvas de produção e preenche as estruturas criadas para armazenamento de cada grandeza
    for (ckp, gl), ql in Q_LIQ_N_FUN[well].items():
        C.add(ckp)
        GL.add(gl)
        # Set_Qliq_n[n].add(ql)
    # Converte estruturas em listas ordenadas
    C = sorted(list(C))
    GL = sorted(list(GL))
    # Set_Qliq_n[n] = sorted(list(Set_Qliq_n[n]))

    return C, GL
