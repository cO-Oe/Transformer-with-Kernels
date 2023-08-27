# When Transformer meets Kernel

Kernel implemented extension of [Attention, Learn to Solve Routing Problems! (ICLR 2019)](https://openreview.net/forum?id=ByxBFsRqYm)

We propose using kernel method to extract more meaningful features. We experimented with the Attention Model (AM), a version of the Transformer architecture, and modified it to enhance its performance with fewer parameters

We extended [Kool's work](https://github.com/wouterkool/attention-learn-to-route) with Kernels mechaniques integrated into the attention-based model on Traveling Salesman Problem (TSP) and compared the results on [TSPLIB95](https://pypi.org/project/tsplib95/) library over 69 tours

The Kernel method is implemented in [nets/compatibility_layer.py](nets/compatibility_layer.py), with Compatibility as the parent class and different kernel methods as child class, including: Cauchy kernel, RBF kernel, Scaled-dot product kernel, and Polynomial Kernel. We mainly discuss Cauchy kernel in our paper as it has the best performance amongst other kernels. For detailed implementation please kindly refer to the code.

## Paper
For more details, please see our paper [When Transformer meets Kernel (link currently under progress)]() 

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

* [/]() the root directory contains the main driver programs for training and evaluations, please check [below](#quick-start) for more informations
* [nets/](nets) contains the implementation of attention-model (from [Kool's work](https://github.com/wouterkool/attention-learn-to-route)) and the [compatibility layer](nets/compatibility_layer.py) (our work) for kernel methods
* [pretrained/](pretrained) contains the pretrained model from [Kool's](https://github.com/wouterkool/attention-learn-to-route/tree/master/pretrained) and the [our Cauchy-kernel model](pretrained/cauchy_tsp_100)
* [TSPLIB/](TSPLIB) contains the tsp problems from tsplib95 library for our testing and result comparisons
* [problems/](problems) contains the tsp problems generated randomly from [generate_data.py](generate_data.py)


## Quick start

For training TSP instances with 100 nodes and using Cauchy kernel and rollout as REINFORCE baseline:
```bash
python run.py --graph_size 100 --baseline rollout --kernel cauchy --run_name 'tsp100_rollout'
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
With Cauchy kernel, for example
```bash
python run.py --graph_size 100 --baseline rollout --kernel cauchy --run_name 'tsp100_rollout' --val_dataset data/tsp/tsp100_validation_seed4321.pkl
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

### Generate Result

For generating results to csv file of running TSPLIB problems
```bash
python tsplib_run.py
```
The output will be generated as tsp_output.csv in your local root directory

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