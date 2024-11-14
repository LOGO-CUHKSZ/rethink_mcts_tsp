import numpy as np
import logging
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from datetime import datetime
import os
import shutil
from mcts_tsp import parallel_mcts_solve

def save_main_script(destination_dir):
    current_script_path = os.path.abspath(__file__)
    os.makedirs(destination_dir, exist_ok=True)
    destination_file = os.path.join(destination_dir, os.path.basename(current_script_path))
    shutil.copyfile(current_script_path, destination_file)  


def get_custom_logger(name: str,descriptions: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{}/output.log'.format(descriptions))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def softmax(x, axis=-1):
    e_x = np.exp(x) 
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Model:
    def __init__(self, opts):        
        # Configure logging
        out_dir = './tune_log/Tune-MCTS'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        desc = timestamp + '_tsp{}_{}_num_inst{}_epochs{}'.format(opts.num_of_nodes, opts.method, opts.num_instances, opts.n_trials)
        save_dir = os.path.join(out_dir, desc)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_main_script(save_dir)
        self.logger = get_custom_logger('Tune_MCTS_Logger', save_dir)
        self.save_dir = save_dir
        self.max_threads = min(opts.max_threads, opts.num_instances)

        tsp_instances = np.load('./trainset/train_tsp_%d.npy' % opts.num_of_nodes)[:opts.num_instances]
        self.tsp_instances = tsp_instances
        self.train_sols = np.stack([np.arange(opts.num_of_nodes)+1 for i in range(opts.num_instances)], 0)
        if opts.method == 'zero':
            self.heatmap = np.zeros((opts.num_instances, opts.num_of_nodes, opts.num_of_nodes))
        elif opts.method == 'difusco-r':
            self.heatmap = np.load('./trainset/heatmap/{}/heatmap_{}.npy'.format('difusco-raw', opts.num_of_nodes))[:opts.num_instances]
        elif opts.method == 'difusco-p':
            self.heatmap = np.load('./trainset/heatmap/{}/heatmap_{}.npy'.format('difusco-processed', opts.num_of_nodes))[:opts.num_instances]
        else:
            self.heatmap = np.load('./trainset/heatmap/{}/heatmap_{}.npy'.format(opts.method, opts.num_of_nodes))[:opts.num_instances]
        self.num_of_nodes = opts.num_of_nodes

        default_mcts_params = {
            500:{
                'alpha': 1, 'beta': 10, 'H': 2,  'max_candidate_num': opts.num_of_nodes, 'use_heatmap': 1, 'max_depth': 100
            },
            
            1000:{
                'alpha': 1, 'beta': 10, 'H': 10,  'max_candidate_num': opts.num_of_nodes, 'use_heatmap': 1, 'max_depth': 10
            },
            10000:{
                'alpha': 1, 'beta': 10, 'H': 2,  'max_candidate_num': opts.num_of_nodes, 'use_heatmap': 1, 'max_depth': 100
            },
        }

        self.alpha = default_mcts_params[opts.num_of_nodes]['alpha']
        self.beta = default_mcts_params[opts.num_of_nodes]['beta']
        self.H = default_mcts_params[opts.num_of_nodes]['H']
        self.max_candidate_num = default_mcts_params[opts.num_of_nodes]['max_candidate_num']
        self.use_heatmap = default_mcts_params[opts.num_of_nodes]['use_heatmap']
        self.max_depth = default_mcts_params[opts.num_of_nodes]['max_depth']
        self.T = opts.T

        self.best_info = None

        for key, value in vars(opts).items():
            self.logger.info('%s: %s', key, value)
        
        self.logger.info('*'*100)
        self.logger.info('*'*100)


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        use_heatmap = Categorical("use_heatmap", [0, 1], default=self.use_heatmap)
        H = Categorical("H", [2, 5, 10], default=self.H)
        max_candidate_num = Categorical("max_candidate_num", list(set([5, 20, 50, self.num_of_nodes])), default=self.num_of_nodes)
        max_depth = Categorical('max_depth', [10, 50, 100, 200], default=self.max_depth) 
        cs.add([use_heatmap, H, max_candidate_num, max_depth])
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        alpha = self.alpha
        beta = self.beta
        H = config["H"]
        max_candidate_num = config["max_candidate_num"]
        use_heatmap =  config["use_heatmap"]
        max_depth = config["max_depth"]

        _, obj, gap, _, _, _ = parallel_mcts_solve(city_num=self.num_of_nodes, 
                            coordinates_list=self.tsp_instances,
                            opt_solutions=self.train_sols,
                            heatmaps=self.heatmap,
                            num_threads=self.max_threads,
                            alpha=alpha,
                            beta=beta,
                            param_h=H, 
                            param_t=self.T,
                            max_candidate_num=max_candidate_num,
                            candidate_use_heatmap=use_heatmap,
                            max_depth=max_depth)

        if self.best_info is None:
            self.best_info = {'config':[alpha, beta,use_heatmap, H, max_candidate_num, max_depth],'obj': np.mean(obj)}
        else:
            if self.best_info['obj'] > np.mean(obj):
                self.best_info['config'] = [alpha, beta, use_heatmap, H, max_candidate_num, max_depth]
                self.best_info['obj'] = np.mean(obj)

        metric = np.mean(obj)

        self.logger.info('Parameters: %s', [alpha, beta, use_heatmap, H, max_candidate_num, max_depth])
        self.logger.info('Mean obj: %f', np.mean(obj))
        return metric
    

class Tune_MCTS:
    def __init__(self, opts):
        self.model = Model(opts)
                    # Create Scenario with custom output directory
        scenario = Scenario(self.model.configspace, 
                            deterministic=True, 
                            n_trials=opts.n_trials, 
                            output_directory=self.model.save_dir,
                            )

        # Now we use SMAC to find the best hyperparameters
        self.smac = HPOFacade(
            scenario,
            self.model.train,
            overwrite=True,
        )

    def run(self,):
        incumbent = self.smac.optimize()

        # Get cost of default configuration
        default_cost = self.smac.validate(self.model.configspace.get_default_configuration())
        self.model.logger.info(f"Default cost: {default_cost}")

        # Calculate the cost of the incumbent
        incumbent_cost = self.smac.validate(incumbent)
        self.model.logger.info(f"Incumbent cost: {incumbent_cost}")

        # save model.best_info
        self.model.logger.info('='*100)
        self.model.logger.info('Best Info: %s', self.model.best_info)


if __name__ == "__main__":
    import argparse
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Tune MCTS Hyperparameters')
    parser.add_argument('--num_of_nodes', type=int, default=10000, help='Number of nodes in the TSP instances')
    parser.add_argument('--num_instances', type=int, default=16, help='Number of TSP instances')
    parser.add_argument('--max_threads', type=int, default=200, help='Number of threads')
    parser.add_argument('--method', type=str, choices=['attgcn', 'dimes', 'softdist','gt','utsp', 'zero', 'difusco-r', 'difusco-p'])
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--T', type=float, default=0.1)
    opts = parser.parse_args()
    tuner = Tune_MCTS(opts)
    tuner.run()
