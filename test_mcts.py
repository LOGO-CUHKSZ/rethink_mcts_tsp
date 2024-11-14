import numpy as np
import os
from mcts_tsp import parallel_mcts_solve

def read_default_heatmap(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    N = int(lines[0].strip())
    assert len(lines) == N + 1
    
    heatmap = np.zeros((N, N))
    for i, line in enumerate(lines[1:]):
        heatmap[i] = np.array([float(x) for x in line.split()])

    return heatmap

def get_heatmap_list(name, num_nodes, num_instances, desc=None, topk=None):
    folder = "testset/all_heatmap"
    if name == 'zero':
        heatmap_list = np.zeros((num_instances, num_nodes, num_nodes))
    else:
        try:
            heatmap_list = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, name, f"heatmap_{num_nodes}.npy"))[:num_instances]
        except:
            heatmap_list = np.zeros((num_instances, num_nodes, num_nodes))
            if desc is None and topk is None:
                folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, name, "heatmap", f"tsp{num_nodes}")
            else:
                folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, name, "heatmap", f"tsp{num_nodes}_{desc}_top{topk}")
            for i in range(num_instances):
                heatmap_list[i] = read_default_heatmap(os.path.join(folder_name, f"heatmaptsp{num_nodes}_{i}.txt"))
    return heatmap_list


def read_concorde_file(file_name, num_nodes, num_instances):
    pos = np.zeros((num_instances, num_nodes, 2))
    opt_sols = np.zeros((num_instances, num_nodes + 1))
    with open(file_name, "r") as file:
        lines = file.readlines()
    assert len(lines) == num_instances
    for i, line in enumerate(lines):
        parts = line.strip().split("output")
        coords = np.array([float(x) for x in parts[0].strip().split()])
        pos[i] = coords.reshape(-1, 2)
        opt_sols[i] = np.array([int(x) for x in parts[1].strip().split(" ")])
    return pos, opt_sols


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MCTS Search')

    # problem setting
    parser.add_argument('--num_of_nodes', type=int, default=1000, help='Number of nodes in the TSP instances')
    parser.add_argument('--method', type=str, default='gt')
    parser.add_argument('--desc', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--load_T', type=float, default=0.1)
    parser.add_argument('--T', type=float, default=0.1)

    # mcts settings
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--param_h', type=float, default=None)
    parser.add_argument('--max_candidate_num', type=int, default=None)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--use_default', action='store_true')
    parser.add_argument('--max_threads', type=int, default=None)

    opts = parser.parse_args()

    test_instance_num = {
        500: 128,
        1000: 128,
        10000: 16
    }

    N = opts.num_of_nodes

    all_mcts_params_500 = {'attgcn': {500: {0.1: {'alpha': 0,
        'beta': 150,
        'param_h': 5,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.05: {'alpha': 1,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 200},
    0.01: {'alpha': 2,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.005: {'alpha': 2,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100}}},
    'difusco-p': {500: {0.1: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.05: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 50,
        'candidate_use_heatmap': 1,
        'max_depth': 50},
    0.01: {'alpha': 0,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 20,
        'candidate_use_heatmap': 1,
        'max_depth': 50},
    0.005: {'alpha': 0,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 50,
        'candidate_use_heatmap': 1,
        'max_depth': 50}}},
    'difusco-r': {500: {0.1: {'alpha': 0,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 20,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.05: {'alpha': 0,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 20,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.01: {'alpha': 2,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 20,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.005: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 20,
        'candidate_use_heatmap': 0,
        'max_depth': 10}}},
    'dimes': {500: {0.1: {'alpha': 0,
        'beta': 100,
        'param_h': 5,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 200},
    0.05: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.01: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.005: {'alpha': 2,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50}}},
    'gt': {500: {0.1: {'alpha': 0,
        'beta': 10,
        'param_h': 5,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.05: {'alpha': 2,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.01: {'alpha': 0,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.005: {'alpha': 0,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 1,
        'max_depth': 200}}},
    'softdist': {500: {0.1: {'alpha': 1,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': N,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.05: {'alpha': 1,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': N,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.01: {'alpha': 2,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 50,
        'candidate_use_heatmap': 1,
        'max_depth': 200},
    0.005: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': N,
        'candidate_use_heatmap': 1,
        'max_depth': 100}}},
    'utsp': {500: {0.1: {'alpha': 0,
        'beta': 100,
        'param_h': 5,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.05: {'alpha': 0,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.01: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50},
    0.005: {'alpha': 1,
        'beta': 150,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50}}},
    'zero': {500: {0.1: {'alpha': 2,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.05: {'alpha': 2,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.01: {'alpha': 2,
        'beta': 10,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 100},
    0.005: {'alpha': 2,
        'beta': 100,
        'param_h': 2,
        'max_candidate_num': 5,
        'candidate_use_heatmap': 0,
        'max_depth': 50}}}}

    all_mcts_params_1000 = {'attgcn': {1000: {0.1: {'alpha': 0,
    'beta': 150,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 200},
   0.05: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 200},
   0.01: {'alpha': 0,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 100},
   0.005: {'alpha': 1,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50}}},
 'difusco-p': {1000: {0.1: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 100},
   0.05: {'alpha': 1,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 50,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.01: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 50},
   0.005: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 50}}},
 'dimes': {1000: {0.1: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 200},
   0.05: {'alpha': 1,
    'beta': 100,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 100},
   0.01: {'alpha': 1,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.005: {'alpha': 1,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50}}},
 'gt': {1000: {0.1: {'alpha': 1,
    'beta': 10,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.05: {'alpha': 0,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.01: {'alpha': 0,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.005: {'alpha': 1,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 1,
    'max_depth': 50}}},
 'softdist': {1000: {0.1: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 20,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.05: {'alpha': 0,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 20,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.01: {'alpha': 0,
    'beta': 150,
    'param_h': 2,
    'max_candidate_num': 50,
    'candidate_use_heatmap': 1,
    'max_depth': 200},
   0.005: {'alpha': 1,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 1000,
    'candidate_use_heatmap': 1,
    'max_depth': 50}}},
 'utsp': {1000: {0.1: {'alpha': 1,
    'beta': 100,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.05: {'alpha': 0,
    'beta': 150,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.01: {'alpha': 1,
    'beta': 100,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.005: {'alpha': 0,
    'beta': 100,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50}}},
 'zero': {1000: {0.1: {'alpha': 1,
    'beta': 100,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 100},
   0.05: {'alpha': 1,
    'beta': 100,
    'param_h': 5,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.01: {'alpha': 1,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 5,
    'candidate_use_heatmap': 0,
    'max_depth': 50},
   0.005: {'alpha': 0,
    'beta': 10,
    'param_h': 2,
    'max_candidate_num': 20,
    'candidate_use_heatmap': 0,
    'max_depth': 10}}}}
    
    all_mcts_params_10000 = {
        'attgcn': {10000: {0.01: {'alpha': 1,
            'beta': 150,
            'param_h': 2,
            'max_candidate_num': 5,
            'candidate_use_heatmap': 1,
            'max_depth': 50},
        0.005: {'alpha': 2,
            'beta': 150,
            'param_h': 10,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 200}}},
        'difusco-p': {10000: {0.01: {'alpha': 0,
            'beta': 100,
            'param_h': 5,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 50},
        0.005: {'alpha': 2,
            'beta': 150,
            'param_h': 10,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 200}}},
        'difusco-r': {10000: {0.01: {'alpha': 2,
            'beta': 10,
            'param_h': 2,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 10},
        0.005: {'alpha': 1,
            'beta': 10,
            'param_h': 2,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 10}}},
        'dimes': {10000: {0.01: {'alpha': 1,
            'beta': 100,
            'param_h': 2,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 10},
        0.005: {'alpha': 2,
            'beta': 150,
            'param_h': 10,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 200}}},
        'gt': {10000: {0.01: 
            #            {'alpha': 1,
            # 'beta': 100,
            # 'param_h': 10,
            # 'max_candidate_num': 1000,
            # 'candidate_use_heatmap': 1,
            # 'max_depth': 100},
             {'alpha': 0,
                                'beta': 10,
                                'param_h': 10,
                                'max_candidate_num': 20,
                                'candidate_use_heatmap': 0,
                                'max_depth': 200},
        0.005: {'alpha': 1,
            'beta': 100,
            'param_h': 5,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 1,
            'max_depth': 100}}},
        'softdist': {10000: {0.01: 
                            #  {'alpha': 2,
                            #     'beta': 100,
                            #     'param_h': 5,
                            #     'max_candidate_num': 20,
                            #     'candidate_use_heatmap': 0,
                            #     'max_depth': 10},
                            {'alpha': 0,
                                'beta': 10,
                                'param_h': 10,
                                'max_candidate_num': 20,
                                'candidate_use_heatmap': 0,
                                'max_depth': 200},
        0.005: {'alpha': 2,
            'beta': 150,
            'param_h': 10,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 200}}},
        #  'utsp': {10000: {}},
        'zero': {10000: {0.01: {'alpha': 0,
            'beta': 100,
            'param_h': 2,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 10},
        0.005: {'alpha': 2,
            'beta': 150,
            'param_h': 10,
            'max_candidate_num': 20,
            'candidate_use_heatmap': 0,
            'max_depth': 200}}}}

    if N == 500:
        all_mcts_params = all_mcts_params_500
    elif N == 1000:
        all_mcts_params = all_mcts_params_1000
    else:
        all_mcts_params = all_mcts_params_10000

    if opts.use_default:
        if opts.method == 'zero':
            mcts_params = {'alpha': 1.0, 'beta': 10, 'param_h': 10,  'max_candidate_num': 1000, 'candidate_use_heatmap': 0, 'max_depth': 10}
        else:
            mcts_params = {'alpha': 1.0, 'beta': 10, 'param_h': 10,  'max_candidate_num': 1000, 'candidate_use_heatmap': 1, 'max_depth': 10}
    else:
        mcts_params = all_mcts_params[opts.method][N][opts.load_T]

    if opts.alpha is not None:
        mcts_params['alpha'] = opts.alpha
    if opts.beta is not None:
        mcts_params['beta'] = opts.beta
    if opts.param_h is not None:
        mcts_params['param_h'] = opts.param_h
    if opts.max_candidate_num is not None:
        mcts_params['max_candidate_num'] = opts.max_candidate_num
    if opts.max_depth is not None:
        mcts_params['max_depth'] = opts.max_depth

    if opts.max_threads is None:
        max_threads = test_instance_num[N]
    else:
        max_threads = opts.max_threads

    pos, opt_sols = read_concorde_file("testset/tsp{}_test_concorde.txt".format(N), N, test_instance_num[N])
    if N != 10000:
        opt_sols = opt_sols[:, :-1]-1
    else:
        opt_sols =  np.stack([np.arange(N) for i in range(test_instance_num[N])], 0)

    heatmap_list = get_heatmap_list(opts.method, N, test_instance_num[N], desc=opts.desc, topk=opts.topk)

    _, total_mcts, total_gap, total_time, _, _, soln_info = parallel_mcts_solve(
                        city_num=N, 
                        coordinates_list=pos, 
                        opt_solutions=opt_sols, 
                        heatmaps=heatmap_list,
                        num_threads=max_threads, 
                        param_t=opts.T,
                        **mcts_params
                        )
    
    if opts.use_default:
        default_suffix = '_default'
    else:
        default_suffix = ''
    
    # save as pkl file
    import pickle
    if not os.path.exists("tune_mcts_results_loadT-{}".format(opts.load_T)):
        os.makedirs("tune_mcts_results_loadT-{}".format(opts.load_T))
    file_name = f"tune_mcts_results_loadT-{opts.load_T}/{opts.method}_tsp{N}_testT-{opts.T}{default_suffix}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump([total_mcts, soln_info], f)

    print(f"{opts.method}_{N}_{opts.T}_load-{opts.load_T}: \t{np.average(total_mcts)} \t{np.average(total_gap)}")