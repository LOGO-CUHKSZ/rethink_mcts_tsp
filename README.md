# Rethinking the "Heatmap + Monte Carlo Tree Search" Paradigm for Large Scale TSP

Implementation of our paper [Rethinking the "Heatmap + Monte Carlo Tree Search" Paradigm for Large Scale TSP]().

## Installation

### Environment Setup
```bash
conda create -n mcts_tsp python=3.10
conda activate mcts_tsp
```

### Dependencies
1. Install system dependencies:
```bash
conda install gxx_linux-64 gcc_linux-64 swig
```

2. Install Python packages:
```bash
pip install smac fire
# Install PyTorch (CPU version), which is sufficient for running the code
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/neo-pan/mcts_tsp.git
```

## Data Preparation

1. Extract test instances:
```bash
cd testset
unzip testdata.zip
```

2. Generate GT-Prior heatmap:
```bash
cd testset/all_heatmap/gt/
python batch_generate_heatmap.py
```

For other baseline heatmaps, download from [xyfffff/rethink_mcts_for_tsp](https://github.com/xyfffff/rethink_mcts_for_tsp) and place in `testset/all_heatmap/`.

## Usage

### Testing
Run `test_mcts.py` to evaluate different configurations:

```bash
python test_mcts.py --num_of_nodes <size> \
                    --method <heatmap_type> \
                    --use_default \
                    --max_threads <thread_count>
```

Key Parameters:
- `num_of_nodes`: Instance size
- `method`: Heatmap source
- `use_default`: Use default hyperparameters or tuned hyperparameters
- `max_threads`: Hardware-dependent parallelization
- See `--help` for additional parameters

### Hyperparameter Tuning
Use `tune_mcts.py` to optimize parameters with SMAC3:

```bash
python tune_mcts.py --num_of_nodes <size> \
                    --num_instances <count> \
                    --method <heatmap_type> \
                    --n_trials <optimization_iterations> \
                    --max_threads <thread_count>
```

Key Parameters:
- `num_instances`: Number of training instances
- `n_trials`: SMAC optimization iterations
- `method`: Choice of heatmap type (attgcn/dimes/softdist/gt/utsp/zero/difusco-r/difusco-p)