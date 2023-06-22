## A Circuit Domain Generalization Framework for Efficient Logic Synthesis in Chip Design

This is our code for our manuscript *A Circuit Domain Generalization Framework for Efficient Logic Synthesis in Chip Design*.

### Installation of Packages

#### ABC Installation

In this repository, we provide the executable file "abc" and the static library "libabc.a". We will also release the source code of our modified abc once the manuscript is accepted.

#### Python Environment Installation

The python environment requires:

* A GPU equipped machine
* Python 3.7.12
* Pytorch 1.13.1
* (*) PyTorch Geometric 2.2.0 (see explanations below)
* See *requirements.txt* for all required packages

*Note*: Generally, different versions of required packages might change slightly. Thus, we have to specify the versions of torch_geometric. Other versions may also work, but we have not tested.

This python environment can be installed easily and thus we omit the detailed instructional codes.

#### Python ABC Interface

We use the Python interface Abc_Py for logic synthesis framework ABC provided in the paper "Exploring Logic Optimizations with Reinforcement Learning and Graph Convolutional Network". Actually, we implement a slight variant of the interface to make it compilable and add some additional functions.

##### Preliminary Packages

To install the Abc_Py, we need to install the pybind11, which is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.

```bash
# (*) Step1: On Linux you’ll need to install the python-dev or python3-dev packages as well as cmake.

# (*) Step2: Download the pybind11 repository
git clone https://github.com/pybind/pybind11.git

# Step3: Install the pybind11 package
mkdir build
cd build
cmake ..
make check -j 4
```

##### Install Abc_Py

```bash
# (*) Step1: set PYBIND11_DIR to the directory of pybind11
# Generally, the directory is ../pybind11.

# (*) Step2: set ABC_DIR to the directory of ABC with the static library libabc.a
# This directory usually is ../libabc.

# Step3: python ABC interface installation
cd ./code/abc_py
python setup.py install
```

### Data Generation

Here we provide an example data generation script of the training datasets in our paper. Moreover, we have released the training datasets in ../data in this repository.

For example, to generate the datasets on the Log2 circuit under the Mfs2 operator:

```bash
cd ./code/data_generation
python o1_data_collector_mfs2.py --blif_data_path [DIR to the Log2 blif file]
```

The above codes will generate the Circuit Dataset on the Log2 circuit under the Mfs2 operator.

### Training and Evaluation

There are mainly two python scripts to launch our codes:

* ./train/o4_pipeline.py: This code is used to train COG classifiers.
* ./test/run_mfs2.py and ./test/run_resub.py: This code is used to evaluate the PruneX operators with COG classifiers. Note that here we apply our PruneX with COG classifiers to the Mfs2 and Resub operators, respectively.

For example, to train a COG classifier:

```bash
# train COG classifier using the Log2 dataset from the Mfs2 operator
python ./train/o4_pipeline.py +config_groups=multi_domain_train_mfs2 config_groups.exp_kwargs.seed=1 config_groups.exp_kwargs.base_log_dir=runs/exp_multi_domain_mfs2_log2 config_groups.train_data_loader.kwargs.npy_data_path=../data/mfs2/open_source_benchmarks/evaluation_strategy_1/epfl/log2_test/multi_domains/train config_groups.test_data_loader.kwargs.npy_data_path=../data/mfs2/open_source_benchmarks/evaluation_strategy_1/epfl/log2_test/multi_domains/test config_groups.policy.kwargs.out_size=2 config_groups.trainer.kwargs.epochs=2001
# train COG classifier using the Log2 dataset from the Resub operator
python o4_pipeline.py +config_groups=multi_domain_train_resub config_groups.exp_kwargs.seed=1 config_groups.exp_kwargs.base_log_dir=runs/exp_multi_domain_resub_log2 config_groups.train_data_loader.kwargs.npy_data_path=../data/resub/open_source_benchmarks/evaluation_strategy_1/epfl/log2_test/multi_domains/train config_groups.test_data_loader.kwargs.npy_data_path=../data/resub/open_source_benchmarks/evaluation_strategy_1/epfl/log2_test/multi_domains/test config_groups.trainer.kwargs.epochs=2001
```

Then it will train the model with one GPU and one processing. All hyperparameters used for training are set in the config file */configs/config_groups/xxx.yaml, e.g.,

* exp_kwargs: basic kwargs for the experiment, such as the base_log_dir and seed
* trainer: kwargs for training, such as the learning rate (lr) and evaluate_freq
* policy: kwargs for our GCNN model, such as the emebedding size and out size
* train_data_loader: kwargs for the train data loader, such as the npy_data_path and max_batch_size
* test_data_loader: kwargs for the test data loader, such as the npy_data_path and max_batch_size

By default, the logdir for training is outputs/YY-MM-DD/HH-MM-SS/xxx. The logs include:

* .hydra: stores all the configs used in this experiment
* models: stores the checkpoints of trained GCNN model
* *runs* stores detailed information at each training step in the tensorboard event file. We mainly care about the following information.
  * top 50% accuracy
  * train loss

#### Evaluation

To compare our PruneX with the default operator in ABC, you have to specify the logdir of the model saved during training:

```bash
# (*) Step1: add the directory of libcall python to the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./test/libs/libcall_python_inference.so:$LD_LIBRARY_PATH
# (*) Step2: cp the evaluator python files to the directory /datasets/ai4eda/preliminary_expers/ai4mfs2
mkdir /datasets/ai4eda/preliminary_expers/ai4mfs2
cp ./test/libs/evaluator/* /datasets/ai4eda/preliminary_expers/ai4mfs2
# Step3: run evaluation
python run_mfs2.py -sel_percents 0.5 --model_path ./models/mfs2/epfl/log2/seed1/itr_2000.pkl --save_dir epfl_log2 --test_blif seed1 --test_blif_path ../data/mfs2/open_source_benchmarks/evaluation_strategy_1/epfl/log2_test/multi_domains/test/log2.blif
```
