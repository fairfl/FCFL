import numpy as np

from .hco_lp import HCO_LP

def hco_search(multi_obj_fg, x = None, deltas = None, args = None):

    factor_delta = args.factor_delta
    lr_delta = args.lr_delta
    eps = [args.eps_g, args.eps_delta_l, args.eps_delta_g]
    max_iters = args.max_epoch_stage1
    n_dim = args.n_dim
    step_size = args.step_size
    store_xs = args.store_xs
    deltas = np.array([0.0,0.0])
    deltas[0] = args.delta_l 
    deltas[1] = args.delta_g 


    # r = [0.98, 0.15]
    x = np.random.randn(n_dim) if x is None else x
           # number of objectives
    lp = HCO_LP( n_dim, eps) # eps [eps_disparity,]
    lss, gammas, d_nds = [], [], []
    if store_xs:
        xs = [x]

    # find the Pareto optimal solution
    desc, asce = 0, 0
    for t in range(max_iters):
        x = x.reshape(-1)
        ls, d_ls = multi_obj_fg(x)
        alpha, deltas = lp.get_alpha(ls, d_ls, deltas, factor_delta, lr_delta) 
        if lp.last_move == "dom":
            desc += 1
        else:
            asce += 1
        lss.append(ls)
        gammas.append(lp.gamma)
        d_nd = alpha @ d_ls
        d_nds.append(np.linalg.norm(d_nd, ord=np.inf)) 


        if np.linalg.norm(d_nd, ord=np.inf) < eps[0] and deltas[0] < eps[1] and deltas[1] < eps[2]:
            print('converged, ', end=',')
            break
        x = x - 10. * max(ls[1], 0.1) * step_size * d_nd
        if store_xs:
            xs.append(x)

    print(f'# iterations={asce+desc}; {100. * desc/(desc+asce)} % descent')
    res = {'ls': np.stack(lss),
           'gammas': np.stack(gammas)}
    if store_xs:
        res['xs': xs]
    return x, res
