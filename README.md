**Awesome Open Source MLOps**

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re) [![Gitter](https://badges.gitter.im/open-source-mlops/community.svg)](https://gitter.im/open-source-mlops/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

An awesome & curated list of best open source MLOps tools for data scientists.

**Contribute**

Contributions are most welcome, please adhere to the [contribution guidelines](contributing.md).

**Community**

You can join our [gitter](https://gitter.im/open-source-mlops/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) channel to discuss.

Table of Contents
=================

* [Table of Contents](#table-of-contents)
* [Training](#training)
   * [IDEs and Workspaces](#ides-and-workspaces)
   * [Frameworks for Training](#frameworks-for-training)
   * [Experiment Tracking](#experiment-tracking)
   * [Visualization](#visualization)
* [Model](#model)
   * [Model Management](#model-management)
   * [Pretrained Model](#pretrained-model)
* [Serving](#serving)
   * [Frameworks for Serving](#frameworks-for-serving)
   * [Optimizations](#optimizations)
   * [Observability](#observability)
* [Large Scale Deployment](#large-scale-deployment)
   * [ML Platforms](#ml-platforms)
   * [Workflow](#workflow)
   * [Scheduling](#scheduling)
* [AutoML](#automl)
* [Data](#data)
   * [Data Management](#data-management)
   * [Data Ingestion](#data-ingestion)
   * [Data Storage](#data-storage)
   * [Data Transformation](#data-transformation)
   * [Feature Engineering](#feature-engineering)
* [Performance](#performance)
   * [ML Compiler](#ml-compiler)
   * [Profiling](#profiling)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

# Training

## IDEs and Workspaces

- [code server](https://github.com/coder/code-server) ![](https://img.shields.io/github/stars/coder/code-server.svg?style=social) - Run VS Code on any machine anywhere and access it in the browser.
- [conda](https://github.com/conda/conda) ![](https://img.shields.io/github/stars/conda/conda.svg?style=social) -  OS-agnostic, system-level binary package manager and ecosystem.
- [Docker](https://github.com/moby/moby) ![](https://img.shields.io/github/stars/moby/moby.svg?style=social) - Moby is an open-source project created by Docker to enable and accelerate software containerization.
- [Jupyter Notebooks](https://github.com/jupyter/notebook) ![](https://img.shields.io/github/stars/jupyter/notebook.svg?style=social) - The Jupyter notebook is a web-based notebook environment for interactive computing.

## Frameworks for Training

- [Caffe](https://github.com/BVLC/caffe) ![](https://img.shields.io/github/stars/BVLC/caffe.svg?style=social) - A fast open framework for deep learning. 
- [ColossalAI](https://github.com/hpcaitech/ColossalAI) ![](https://img.shields.io/github/stars/hpcaitech/ColossalAI.svg?style=social) - An integrated large-scale model training system with efficient parallelization techniques.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) - DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [Horovod](https://github.com/horovod/horovod) ![](https://img.shields.io/github/stars/horovod/horovod.svg?style=social) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
- [Kedro](https://github.com/kedro-org/kedro) ![](https://img.shields.io/github/stars/kedro-org/kedro.svg?style=social) - Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code.
- [LightGBM](https://github.com/microsoft/LightGBM) ![](https://img.shields.io/github/stars/microsoft/LightGBM.svg?style=social) - A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
- [MegEngine](https://github.com/MegEngine/MegEngine) ![](https://img.shields.io/github/stars/MegEngine/MegEngine.svg?style=social) - MegEngine is a fast, scalable and easy-to-use deep learning framework, with auto-differentiation.
- [MindSpore](https://github.com/mindspore-ai/mindspore) ![](https://img.shields.io/github/stars/mindspore-ai/mindspore.svg?style=social) - MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios.
- [MXNet](https://github.com/apache/incubator-mxnet) ![](https://img.shields.io/github/stars/apache/incubator-mxnet.svg?style=social) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler.
- [Oneflow](https://github.com/Oneflow-Inc/oneflow) ![](https://img.shields.io/github/stars/Oneflow-Inc/oneflow.svg?style=social) - OneFlow is a performance-centered and open-source deep learning framework.
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ![](https://img.shields.io/github/stars/PaddlePaddle/Paddle.svg?style=social) - Machine Learning Framework from Industrial Practice.
- [PyTorch](https://github.com/pytorch/pytorch) ![](https://img.shields.io/github/stars/pytorch/pytorch.svg?style=social) -  Tensors and Dynamic neural networks in Python with strong GPU acceleration.
- [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) ![](https://img.shields.io/github/stars/PyTorchLightning/pytorch-lightning.svg?style=social) - The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.
- [XGBoost](https://github.com/dmlc/xgboost) ![](https://img.shields.io/github/stars/dmlc/xgboost.svg?style=social) - Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library.
- [TenorFlow](https://github.com/tensorflow/tensorflow) ![](https://img.shields.io/github/stars/tensorflow/tensorflow.svg?style=social) - An Open Source Machine Learning Framework for Everyone.

## Experiment Tracking

- [Aim](https://github.com/aimhubio/aim) ![](https://img.shields.io/github/stars/aimhubio/aim.svg?style=social) - an easy-to-use and performant open-source experiment tracker.
- [Guild AI](https://github.com/guildai/guildai) ![](https://img.shields.io/github/stars/guildai/guildai.svg?style=social) - Experiment tracking, ML developer tools.
- [MLRun](https://github.com/mlrun/mlrun) ![](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=social) - Machine Learning automation and tracking.
-[Kedro-Viz](https://github.com/kedro-org/kedro-viz) - Kedro-Viz is an interactive development tool for building data science pipelines with Kedro. Kedro-Viz also allows users to view and compare different runs in the Kedro project.
- [LabNotebook](https://github.com/henripal/labnotebook) ![](https://img.shields.io/github/stars/henripal/labnotebook.svg?style=social) - LabNotebook is a tool that allows you to flexibly monitor, record, save, and query all your machine learning experiments.
- [Sacred](https://github.com/IDSIA/sacred) ![](https://img.shields.io/github/stars/IDSIA/sacred.svg?style=social) - Sacred is a tool to help you configure, organize, log and reproduce experiments.

## Visualization

- [Maniford](https://github.com/uber/manifold) ![](https://img.shields.io/github/stars/uber/manifold.svg?style=social) - A model-agnostic visual debugging tool for machine learning.
- [netron](https://github.com/lutzroeder/netron) ![](https://img.shields.io/github/stars/lutzroeder/netron.svg?style=social) - Visualizer for neural network, deep learning, and machine learning models.
- [TensorBoard](https://github.com/tensorflow/tensorboard) ![](https://img.shields.io/github/stars/tensorflow/tensorboard.svg?style=social) - TensorFlow's Visualization Toolkit.
- [TensorSpace](https://github.com/tensorspace-team/tensorspace) ![](https://img.shields.io/github/stars/tensorspace-team/tensorspace.svg?style=social) - Neural network 3D visualization framework, build interactive and intuitive model in browsers, support pre-trained deep learning models from TensorFlow, Keras, TensorFlow.js.
- [dtreeviz](https://github.com/parrt/dtreeviz) ![](https://img.shields.io/github/stars/parrt/dtreeviz.svg?style=social) - A python library for decision tree visualization and model interpretation.
- [Zetane Viewer](https://github.com/zetane/viewer) ![](https://img.shields.io/github/stars/zetane/viewer.svg?style=social) - ML models and internal tensors 3D visualizer.

# Model

## Model Management

## Pretrained Model

# Serving

## Frameworks for Serving

## Optimizations

- [Forward](https://github.com/Tencent/Forward) ![](https://img.shields.io/github/stars/Tencent/Forward.svg?style=social) - A library for high performance deep learning inference on NVIDIA GPUs.
- [PocketFlow](https://github.com/Tencent/PocketFlow) ![](https://img.shields.io/github/stars/Tencent/PocketFlow.svg?style=social) - use AutoML to do model compression.

## Observability

# Large Scale Deployment

## ML Platforms

- [ClearML](https://github.com/allegroai/clearml) ![](https://img.shields.io/github/stars/allegroai/clearmlsvg?style=social) - Auto-Magical CI/CD to streamline your ML workflow. Experiment Manager, MLOps and Data-Management.
- [MLflow](https://github.com/mlflow/mlflow) ![](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=social) - Open source platform for the machine learning lifecycle.
- [Kubeflow](https://github.com/kubeflow/kubeflow) ![](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=social) - Machine Learning Toolkit for Kubernetes.
- [PAI](https://github.com/microsoft/pai) ![](https://img.shields.io/github/stars/microsoft/pai.svg?style=social) - Resource scheduling and cluster management for AI.
- [Polyaxon](https://github.com/polyaxon/polyaxon) ![](https://img.shields.io/github/stars/polyaxon/polyaxon.svg?style=social) - Machine Learning Management & Orchestration Platform.

## Workflow

- [Keras](https://github.com/keras-team/keras) ![](https://img.shields.io/github/stars/keras-team/keras.svg?style=social) - Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow.

## Scheduling

# AutoML

- [Adanet](https://github.com/tensorflow/adanet) ![](https://img.shields.io/github/stars/tensorflow/adanet.svg?style=social) - Tensorflow package for AdaNet.
- [Advisor](https://github.com/tobegit3hub/advisor) ![](https://img.shields.io/github/stars/tobegit3hub/advisor.svg?style=social) - open-source implementation of Google Vizier for hyper parameters tuning.
- [Archai](https://github.com/microsoft/archai) ![](https://img.shields.io/github/stars/microsoft/archai.svg?style=social) - a platform for Neural Network Search (NAS) that allows you to generate efficient deep networks for your applications.
- [auptimizer](https://github.com/LGE-ARC-AdvancedAI/auptimizer) ![](https://img.shields.io/github/stars/LGE-ARC-AdvancedAI/auptimizer.svg?style=social) - An automatic ML model optimization tool.
- [autoai](https://github.com/blobcity/autoai) ![](https://img.shields.io/github/stars/blobcity/autoai.svg?style=social) - A framework to find the best performing AI/ML model for any AI problem.
- [AutoGL](https://github.com/THUMNLab/AutoGL) ![](https://img.shields.io/github/stars/THUMNLab/AutoGL.svg?style=social) - An autoML framework & toolkit for machine learning on graphs
- [AutoGluon](https://github.com/awslabs/autogluon) ![](https://img.shields.io/github/stars/awslabs/autogluon.svg?style=social) - AutoML for Image, Text, and Tabular Data.
- [automl-gs](https://github.com/minimaxir/automl-gs) ![](https://img.shields.io/github/stars/minimaxir/automl-gs.svg?style=social) - Provide an input CSV and a target field to predict, generate a model + code to run it.
- [autokeras](https://github.com/keras-team/autokeras) ![](https://img.shields.io/github/stars/keras-team/autokeras.svg?style=social) - AutoML library for deep learning.
- [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) ![](https://img.shields.io/github/stars/automl/Auto-PyTorch.svg?style=social) - Automatic architecture search and hyperparameter optimization for PyTorch.
- [auto-sklearn](https://github.com/automl/auto-sklearn) ![](https://img.shields.io/github/stars/automl/auto-sklearn.svg?style=social) - an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.
- [AutoWeka](https://github.com/automl/autoweka) ![](https://img.shields.io/github/stars/automl/autoweka.svg?style=social) - hyperparameter search for Weka.
- [Chocolate](https://github.com/AIworx-Labs/chocolate) ![](https://img.shields.io/github/stars/AIworx-Labs/chocolate.svg?style=social) - A fully decentralized hyperparameter optimization framework.
- [Dragonfly](https://github.com/dragonfly/dragonfly) ![](https://img.shields.io/github/stars/dragonfly/dragonfly.svg?style=social) - An open source python library for scalable Bayesian optimisation.
- [Determined](https://github.com/determined-ai/determined) ![](https://img.shields.io/github/stars/determined-ai/determined.svg?style=social) - scalable deep learning training platform with integrated hyperparameter tuning support; includes Hyperband, PBT, and other search methods.
- [DEvol (DeepEvolution)](https://github.com/joeddav/devol) ![](https://img.shields.io/github/stars/joeddav/devol.svg?style=social) - a basic proof of concept for genetic architecture search in Keras.
- [EvalML](https://github.com/alteryx/evalml) ![](https://img.shields.io/github/stars/alteryx/evalml.svg?style=social) - An open source python library for AutoML.
- [FEDOT](https://github.com/nccr-itmo/FEDOT) ![](https://img.shields.io/github/stars/nccr-itmo/FEDOT.svg?style=social) - AutoML framework for the design of composite pipelines.
- [FLAML](https://github.com/microsoft/FLAML) ![](https://img.shields.io/github/stars/microsoft/FLAML.svg?style=social) - Fast and lightweight AutoML ([paper](https://www.microsoft.com/en-us/research/publication/flaml-a-fast-and-lightweight-automl-library/)).
- [Goptuna](https://github.com/c-bata/goptuna) ![](https://img.shields.io/github/stars/c-bata/goptuna.svg?style=social) - A hyperparameter optimization framework, inspired by Optuna.
- [HpBandSter](https://github.com/automl/HpBandSter) ![](https://img.shields.io/github/stars/automl/HpBandSter.svg?style=social) - a framework for distributed hyperparameter optimization.
- [HPOlib2](https://github.com/automl/HPOlib2) ![](https://img.shields.io/github/stars/automl/HPOlib2.svg?style=social) - a library for hyperparameter optimization and black box optimization benchmarks.
- [Hyperband](https://github.com/zygmuntz/hyperband) ![](https://img.shields.io/github/stars/zygmuntz/hyperband.svg?style=social) - open source code for tuning hyperparams with Hyperband.
- [Hypernets](https://github.com/DataCanvasIO/Hypernets) ![](https://img.shields.io/github/stars/DataCanvasIO/Hypernets.svg?style=social) - A General Automated Machine Learning Framework.
- [Hyperopt](https://github.com/hyperopt/hyperopt) ![](https://img.shields.io/github/stars/hyperopt/hyperopt.svg?style=social) - Distributed Asynchronous Hyperparameter Optimization in Python.
- [hyperunity](https://github.com/gdikov/hypertunity) ![](https://img.shields.io/github/stars/gdikov/hypertunity.svg?style=social) - A toolset for black-box hyperparameter optimisation.
- [Katib](https://github.com/kubeflow/katib) ![](https://img.shields.io/github/stars/kubeflow/katib.svg?style=social) - Katib is a Kubernetes-native project for automated machine learning (AutoML).
- [Keras Tuner](https://github.com/keras-team/keras-tuner) ![](https://img.shields.io/github/stars/keras-team/keras-tuner.svg?style=social) - Hyperparameter tuning for humans.
- [learn2learn](https://github.com/learnables/learn2learn) ![](https://img.shields.io/github/stars/learnables/learn2learn.svg?style=social) - PyTorch Meta-learning Framework for Researchers.
- [Ludwig](https://github.com/uber/ludwig) ![](https://img.shields.io/github/stars/uber/ludwig.svg?style=social) - a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code.
- [MOE](https://github.com/Yelp/MOE) ![](https://img.shields.io/github/stars/Yelp/MOE.svg?style=social) - a global, black box optimization engine for real world metric optimization by Yelp.
- [Model Search](https://github.com/google/model_search) ![](https://img.shields.io/github/stars/google/model_search.svg?style=social) - a framework that implements AutoML algorithms for model architecture search at scale.
- [NASGym](https://github.com/gomerudo/nas-env) ![](https://img.shields.io/github/stars/gomerudo/nas-env.svg?style=social) - a proof-of-concept OpenAI Gym environment for Neural Architecture Search (NAS).
- [NNI](https://github.com/Microsoft/nni) ![](https://img.shields.io/github/stars/Microsoft/nni.svg?style=social) - An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning.
- [Optuna](https://github.com/optuna/optuna) ![](https://img.shields.io/github/stars/optuna/optuna.svg?style=social) - A hyperparameter optimization framework.
- [Ray Tune](github.com/ray-project/ray) ![](https://img.shields.io/github/stars/ect/ray.svg?style=social) - Scalable Hyperparameter Tuning.
- [REMBO](https://github.com/ziyuw/rembo) ![](https://img.shields.io/github/stars/ziyuw/rembo.svg?style=social) - Bayesian optimization in high-dimensions via random embedding.
- [RoBO](https://github.com/automl/RoBO) ![](https://img.shields.io/github/stars/automl/RoBO.svg?style=social) - a Robust Bayesian Optimization framework.
- [scikit-optimize(skopt)](https://github.com/scikit-optimize/scikit-optimize) ![](https://img.shields.io/github/stars/scikit-optimize/scikit-optimize.svg?style=social) - Sequential model-based optimization with a `scipy.optimize` interface.
- [Spearmint](https://github.com/HIPS/Spearmint) ![](https://img.shields.io/github/stars/HIPS/Spearmint.svg?style=social) - a software package to perform Bayesian optimization.
- [TPOT](http://automl.info/tpot/) ![](https://img.shields.io/github/stars/tpot/.svg?style=social) - one of the very first AutoML methods and open-source software packages.
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta) ![](https://img.shields.io/github/stars/tristandeleu/pytorch-meta.svg?style=social) - A Meta-Learning library for PyTorch.
- [Vegas](https://github.com/huawei-noah/vega) ![](https://img.shields.io/github/stars/huawei-noah/vega.svg?style=social) - an AutoML algorithm tool chain by Huawei Noah's Arb Lab.

# Data

## Data Management

- [Dolt](https://github.com/dolthub/dolt) ![](https://img.shields.io/github/stars/dolthub/dolt.svg?style=social) - Git for Data.
- [DVC](https://github.com/iterative/dvc) ![](https://img.shields.io/github/stars/iterative/dvc.svg?style=social) - Data Version Control | Git for Data & Models | ML Experiments Management.
- [Hub](https://github.com/activeloopai/Hub) ![](https://img.shields.io/github/stars/activeloopai/Hub.svg?style=social) - Hub is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size.
- [Quilt](https://github.com/quiltdata/quilt) ![](https://img.shields.io/github/stars/quiltdata/quilt.svg?style=social) - A self-organizing data hub for S3.

## Data Ingestion

## Data Storage

- [LakeFS](https://github.com/treeverse/lakeFS) ![](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=social) - Git-like capabilities for your object storage.

## Data Transformation

## Feature Engineering

- [FeatureTools](https://github.com/Featuretools/featuretools) ![](https://img.shields.io/github/stars/Featuretools/featuretools.svg?style=social) - An open source python framework for automated feature engineering

# Performance

## ML Compiler

## Profiling

**[⬆ back to top](#)**
