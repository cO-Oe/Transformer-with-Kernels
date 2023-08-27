# Transformers with Kernels

Kernel implemented extension of [Attention, Learn to Solve Routing Problems! (ICLR 2019)](https://openreview.net/forum?id=ByxBFsRqYm)

We extended [Kool's work](https://github.com/wouterkool/attention-learn-to-route) with Kernels mechaniques integrated into the attention-based model on Traveling Salesman Problem (TSP) and compared the results on [TSPLIB95](https://pypi.org/project/tsplib95/) library over 59 tours

Example plot from Kool
![TSP100](images/tsp.gif)

## Paper
For more details, please see our paper [Kernels with Transformers (link currently under progress)]() 

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)
* tsplib95==0.7.1

Please check environment_server.yml for detailed environment setup

## Repository Overview

There are several directories in this repo:

* [nets/](nets) contains the implementation of attention-model (from [Kool's work](https://github.com/wouterkool/attention-learn-to-route)) and the [compatability layer](nets/compatability_layer.py) (our work) for kernel methods
* [pretrained/](pretrained) contains the pretrained model from [Kool's](https://github.com/wouterkool/attention-learn-to-route/tree/master/pretrained) and the [our Cauchy-kernel model](pretrained/cauchy_tsp_100)
* [TSPLIB/](TSPLIB) contains the tsp problems from tsplib95 library



## Quick start

For training TSP instances with 100 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 100 --baseline rollout --run_name 'tsp100_rollout'
```

## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper of Kool's and ours) for all problems:
```bash
python generate_data.py --problem all --name validation --seed 4321
python generate_data.py --problem all --name test --seed 1234
```

### Training

For training TSP instances with 100 nodes and using rollout as REINFORCE baseline and using the generated validation set:
```bash
python run.py --graph_size 100 --baseline rollout --run_name 'tsp100_rollout' --val_dataset data/tsp/tsp100_validation_seed4321.pkl
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 100 --load_path pretrained/tsp_100/epoch-99.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 100 --load_path 'outputs/tsp_100/tsp100_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure timing and save the results:
```bash
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.

#### Sampling
To report the best of 1280 sampled solutions, use
```bash
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy sample --width 1280 --eval_batch_size 1
```
Beam Search (not in the paper) is also recently added and can be used using `--decode_strategy bs --width {beam_size}`.

#### To run baselines
Baselines for different problems are within the corresponding folders and can be ran (on multiple datasets at once) as follows
```bash
python -m problems.tsp.tsp_baseline farthest_insertion data/tsp/tsp20_test_seed1234.pkl data/tsp/tsp50_test_seed1234.pkl data/tsp/tsp100_test_seed1234.pkl
```
To run baselines, you need to install [Compass](https://github.com/bcamath-ds/compass) by running the `install_compass.sh` script from within the `problems/op` directory and [Concorde](http://www.math.uwaterloo.ca/tsp/concorde.html) using the `install_concorde.sh` script from within `problems/tsp`. [LKH3](http://akira.ruc.dk/~keld/research/LKH-3/) should be automatically downloaded and installed when required. To use [Gurobi](http://www.gurobi.com), obtain a ([free academic](http://www.gurobi.com/registration/academic-license-reg)) license and follow the [installation instructions](https://www.gurobi.com/documentation/8.1/quickstart_windows/installing_the_anaconda_py.html).

### Generate Result

For generating results of running TSPLIB problems
```bash
python tsplib_run.py
```

### Other options and help
```bash
python run.py -h
python eval.py -h
```

## Citation
```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```

## Acknowledgements
TBD