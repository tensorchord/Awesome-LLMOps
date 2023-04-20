**Awesome LLMOps**

<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge-flat2.svg"></a>

An awesome & curated list of the best LLMOps tools for developers.

**Contribute**

Contributions are most welcome, please adhere to the [contribution guidelines](contributing.md).

Table of Contents
=================

* [Table of Contents](#table-of-contents)
* [Training](#training)
   * [IDEs and Workspaces](#ides-and-workspaces)
   * [Foundation Model Fine Tuning](#foundation-model-fine-tuning)
   * [Frameworks for Training](#frameworks-for-training)
   * [Experiment Tracking](#experiment-tracking)
   * [Visualization](#visualization)
* [Model](#model)
   * [Model Management](#model-management)
   * [Large Language Model](#large-language-model)
   * [CV Foundation Model](#cv-foundation-model)
* [Serving](#serving)
   * [Frameworks/Servers for Serving](#frameworksservers-for-serving)
   * [Large Model Serving](#large-model-serving)
   * [Optimizations](#optimizations)
   * [Observability](#observability)
* [LLMOps](#llmops)
* [Search](#search)
   * [Vector search](#vector-search)
* [Data](#data)
   * [Data Management](#data-management)
   * [Data Storage](#data-storage)
   * [Data Tracking](#data-tracking)
   * [Feature Engineering](#feature-engineering)
   * [Data/Feature enrichment](#datafeature-enrichment)
* [Large Scale Deployment](#large-scale-deployment)
   * [ML Platforms](#ml-platforms)
   * [Workflow](#workflow)
   * [Scheduling](#scheduling)
* [Performance](#performance)
   * [ML Compiler](#ml-compiler)
   * [Profiling](#profiling)
* [AutoML](#automl)
* [Federated ML](#federated-ml)
* [Awesome Lists](#awesome-lists)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

# Training

## IDEs and Workspaces

- [code server](https://github.com/coder/code-server) ![](https://img.shields.io/github/stars/coder/code-server.svg?style=social) - Run VS Code on any machine anywhere and access it in the browser.
- [conda](https://github.com/conda/conda) ![](https://img.shields.io/github/stars/conda/conda.svg?style=social) -  OS-agnostic, system-level binary package manager and ecosystem.
- [Docker](https://github.com/moby/moby) ![](https://img.shields.io/github/stars/moby/moby.svg?style=social) - Moby is an open-source project created by Docker to enable and accelerate software containerization.
- [envd](https://github.com/tensorchord/envd) ![](https://img.shields.io/github/stars/tensorchord/envd.svg?style=social) - 🏕️ Reproducible development environment for AI/ML.
- [Jupyter Notebooks](https://github.com/jupyter/notebook) ![](https://img.shields.io/github/stars/jupyter/notebook.svg?style=social) - The Jupyter notebook is a web-based notebook environment for interactive computing.

**[⬆ back to ToC](#table-of-contents)**

## Foundation Model Fine Tuning

- [alpaca-lora](https://github.com/tloen/alpaca-lora) ![](https://img.shields.io/github/stars/tloen/alpaca-lora.svg?style=social) - Instruct-tune LLaMA on consumer hardware
- [LMFlow](https://github.com/OptimalScale/LMFlow) ![](https://img.shields.io/github/stars/OptimalScale/LMFlow.svg?style=social) - An Extensible Toolkit for Finetuning and Inference of Large Foundation Models
- [Lora](https://github.com/cloneofsimo/lora) ![](https://img.shields.io/github/stars/cloneofsimo/lora.svg?style=social) - Using Low-rank adaptation to quickly fine-tune diffusion models.
- [peft](https://github.com/huggingface/peft) ![](https://img.shields.io/github/stars/huggingface/peft.svg?style=social) - State-of-the-art Parameter-Efficient Fine-Tuning.

**[⬆ back to ToC](#table-of-contents)**

## Frameworks for Training

- [Apache MXNet](https://github.com/apache/mxnet) ![](https://img.shields.io/github/stars/apache/mxnet.svg?style=social) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler.
- [Caffe](https://github.com/BVLC/caffe) ![](https://img.shields.io/github/stars/BVLC/caffe.svg?style=social) - A fast open framework for deep learning. 
- [ColossalAI](https://github.com/hpcaitech/ColossalAI) ![](https://img.shields.io/github/stars/hpcaitech/ColossalAI.svg?style=social) - An integrated large-scale model training system with efficient parallelization techniques.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) - DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [Horovod](https://github.com/horovod/horovod) ![](https://img.shields.io/github/stars/horovod/horovod.svg?style=social) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
- [Jax](https://github.com/google/jax) ![](https://img.shields.io/github/stars/google/jax.svg?style=social) - Autograd and XLA for high-performance machine learning research.
- [Kedro](https://github.com/kedro-org/kedro) ![](https://img.shields.io/github/stars/kedro-org/kedro.svg?style=social) - Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code.
- [Keras](https://github.com/keras-team/keras) ![](https://img.shields.io/github/stars/keras-team/keras.svg?style=social) - Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow.
- [LightGBM](https://github.com/microsoft/LightGBM) ![](https://img.shields.io/github/stars/microsoft/LightGBM.svg?style=social) - A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
- [MegEngine](https://github.com/MegEngine/MegEngine) ![](https://img.shields.io/github/stars/MegEngine/MegEngine.svg?style=social) - MegEngine is a fast, scalable and easy-to-use deep learning framework, with auto-differentiation.
- [metric-learn](https://github.com/scikit-learn-contrib/metric-learn) ![](https://img.shields.io/github/stars/scikit-learn-contrib/metric-learn.svg?style=social) - Metric Learning Algorithms in Python.
- [MindSpore](https://github.com/mindspore-ai/mindspore) ![](https://img.shields.io/github/stars/mindspore-ai/mindspore.svg?style=social) - MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios.
- [Oneflow](https://github.com/Oneflow-Inc/oneflow) ![](https://img.shields.io/github/stars/Oneflow-Inc/oneflow.svg?style=social) - OneFlow is a performance-centered and open-source deep learning framework.
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ![](https://img.shields.io/github/stars/PaddlePaddle/Paddle.svg?style=social) - Machine Learning Framework from Industrial Practice.
- [PyTorch](https://github.com/pytorch/pytorch) ![](https://img.shields.io/github/stars/pytorch/pytorch.svg?style=social) -  Tensors and Dynamic neural networks in Python with strong GPU acceleration.
- [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) ![](https://img.shields.io/github/stars/PyTorchLightning/pytorch-lightning.svg?style=social) - The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.
- [XGBoost](https://github.com/dmlc/xgboost) ![](https://img.shields.io/github/stars/dmlc/xgboost.svg?style=social) - Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) ![](https://img.shields.io/github/stars/scikit-learn/scikit-learn.svg?style=social) - Machine Learning in Python.
- [TensorFlow](https://github.com/tensorflow/tensorflow) ![](https://img.shields.io/github/stars/tensorflow/tensorflow.svg?style=social) - An Open Source Machine Learning Framework for Everyone.
- [VectorFlow](https://github.com/Netflix/vectorflow) ![](https://img.shields.io/github/stars/Netflix/vectorflow.svg?style=social) - A minimalist neural network library optimized for sparse data and single machine environments.

**[⬆ back to ToC](#table-of-contents)**

## Experiment Tracking

- [Aim](https://github.com/aimhubio/aim) ![](https://img.shields.io/github/stars/aimhubio/aim.svg?style=social) - an easy-to-use and performant open-source experiment tracker.
- [ClearML](https://github.com/allegroai/clearml) ![](https://img.shields.io/github/stars/allegroai/clearml.svg?style=social) - Auto-Magical CI/CD to streamline your ML workflow. Experiment Manager, MLOps and Data-Management
- [Guild AI](https://github.com/guildai/guildai) ![](https://img.shields.io/github/stars/guildai/guildai.svg?style=social) - Experiment tracking, ML developer tools.
- [MLRun](https://github.com/mlrun/mlrun) ![](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=social) - Machine Learning automation and tracking.
- [Kedro-Viz](https://github.com/kedro-org/kedro-viz) ![](https://img.shields.io/github/stars/kedro-org/kedro-viz.svg?style=social) - Kedro-Viz is an interactive development tool for building data science pipelines with Kedro. Kedro-Viz also allows users to view and compare different runs in the Kedro project.
- [LabNotebook](https://github.com/henripal/labnotebook) ![](https://img.shields.io/github/stars/henripal/labnotebook.svg?style=social) - LabNotebook is a tool that allows you to flexibly monitor, record, save, and query all your machine learning experiments.
- [Sacred](https://github.com/IDSIA/sacred) ![](https://img.shields.io/github/stars/IDSIA/sacred.svg?style=social) - Sacred is a tool to help you configure, organize, log and reproduce experiments.

**[⬆ back to ToC](#table-of-contents)**

## Visualization

- [Maniford](https://github.com/uber/manifold) ![](https://img.shields.io/github/stars/uber/manifold.svg?style=social) - A model-agnostic visual debugging tool for machine learning.
- [netron](https://github.com/lutzroeder/netron) ![](https://img.shields.io/github/stars/lutzroeder/netron.svg?style=social) - Visualizer for neural network, deep learning, and machine learning models.
- [OpenOps](https://github.com/ThePlugJumbo/openops) ![](https://img.shields.io/github/stars/theplugjumbo/openops.svg?style=social) - Bring multiple data streams into one dashboard.
- [TensorBoard](https://github.com/tensorflow/tensorboard) ![](https://img.shields.io/github/stars/tensorflow/tensorboard.svg?style=social) - TensorFlow's Visualization Toolkit.
- [TensorSpace](https://github.com/tensorspace-team/tensorspace) ![](https://img.shields.io/github/stars/tensorspace-team/tensorspace.svg?style=social) - Neural network 3D visualization framework, build interactive and intuitive model in browsers, support pre-trained deep learning models from TensorFlow, Keras, TensorFlow.js.
- [dtreeviz](https://github.com/parrt/dtreeviz) ![](https://img.shields.io/github/stars/parrt/dtreeviz.svg?style=social) - A python library for decision tree visualization and model interpretation.
- [Zetane Viewer](https://github.com/zetane/viewer) ![](https://img.shields.io/github/stars/zetane/viewer.svg?style=social) - ML models and internal tensors 3D visualizer.
- [Zeno](https://github.com/zeno-ml/zeno) ![](https://img.shields.io/github/stars/zeno-ml/zeno.svg?style=social) - AI evaluation platform for interactively exploring data and model outputs.

**[⬆ back to ToC](#table-of-contents)**

# Model

## Model Management

- [dvc](https://github.com/iterative/dvc) ![](https://img.shields.io/github/stars/iterative/dvc.svg?style=social) - Data Version Control | Git for Data & Models | ML Experiments Management
- [ModelDB](https://github.com/VertaAI/modeldb) ![](https://img.shields.io/github/stars/VertaAI/modeldb.svg?style=social) - Open Source ML Model Versioning, Metadata, and Experiment Management
- [MLEM](https://github.com/iterative/mlem) ![](https://img.shields.io/github/stars/iterative/mlem.svg?style=social) - A tool to package, serve, and deploy any ML model on any platform.
- [ormb](https://github.com/kleveross/ormb) ![](https://img.shields.io/github/stars/kleveross/ormb.svg?style=social) - Docker for Your ML/DL Models Based on OCI Artifacts

**[⬆ back to ToC](#table-of-contents)**

## Large Language Model

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) ![](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca.svg?style=social) - Code and documentation to train Stanford's Alpaca models, and generate the data.
- [Bloom](https://github.com/bigscience-workshop/model_card) ![](https://img.shields.io/github/stars/bigscience-workshop/model_card.svg?style=social) - BigScience Large Open-science Open-access Multilingual Language Model
- [dolly](https://github.com/databrickslabs/dolly) ![](https://img.shields.io/github/stars/databrickslabs/dolly.svg?style=social) - Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform
- [GLM-130B (ChatGLM)](https://github.com/THUDM/GLM-130B) ![](https://img.shields.io/github/stars/THUDM/GLM-130B.svg?style=social) - An Open Bilingual Pre-Trained Model (ICLR 2023)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) ![](https://img.shields.io/github/stars/EleutherAI/gpt-neox.svg?style=social) - An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.

**[⬆ back to ToC](#table-of-contents)**

## CV Foundation Model

- [disco-diffusion](https://github.com/alembics/disco-diffusion) ![](https://img.shields.io/github/stars/alembics/disco-diffusion.svg?style=social) - A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations.
- [midjourney](https://www.midjourney.com/home/) - Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.
- [stable-diffusion](https://github.com/CompVis/stable-diffusion) ![](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?style=social) - A latent text-to-image diffusion model
- [stable-diffusion v2](https://github.com/Stability-AI/stablediffusion) ![](https://img.shields.io/github/stars/Stability-AI/stablediffusion.svg?style=social) - High-Resolution Image Synthesis with Latent Diffusion Models

**[⬆ back to ToC](#table-of-contents)**

# Serving

## Frameworks/Servers for Serving

- [BentoML](https://github.com/bentoml/BentoML) ![](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=social) - The Unified Model Serving Framework
- [ForestFlow](https://github.com/ForestFlow/ForestFlow) ![](https://img.shields.io/github/stars/ForestFlow/ForestFlow.svg?style=social) - Policy-driven Machine Learning Model Server.
- [Multi Model Server](https://github.com/awslabs/multi-model-server) ![](https://img.shields.io/github/stars/awslabs/multi-model-server.svg?style=social) - Multi Model Server is a tool for serving neural net models for inference.
- [Neuropod](https://github.com/uber/neuropod) ![](https://img.shields.io/github/stars/uber/neuropod.svg?style=social) - A uniform interface to run deep learning models from multiple frameworks
- [Pinferencia](https://github.com/underneathall/pinferencia) ![](https://img.shields.io/github/stars/underneathall/pinferencia.svg?style=social) - Python + Inference - Model Deployment library in Python. Simplest model inference server ever.
- [Service Streamer](https://github.com/ShannonAI/service-streamer) ![](https://img.shields.io/github/stars/ShannonAI/service-streamer.svg?style=social) - Boosting your Web Services of Deep Learning Applications.
- [TFServing](https://github.com/tensorflow/serving) ![](https://img.shields.io/github/stars/tensorflow/serving.svg?style=social) - A flexible, high-performance serving system for machine learning models.
- [Torchserve](https://github.com/pytorch/serve) ![](https://img.shields.io/github/stars/pytorch/serve.svg?style=social) - Serve, optimize and scale PyTorch models in production
- [Triton Server (TRTIS)](https://github.com/triton-inference-server/server) ![](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=social) - The Triton Inference Server provides an optimized cloud and edge inferencing solution.

**[⬆ back to ToC](#table-of-contents)**

## Large Model Serving

- [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve) ![](https://img.shields.io/github/stars/deep-diver/Alpaca-LoRA-Serve.svg?style=social) - Alpaca-LoRA as Chatbot service
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) ![](https://img.shields.io/github/stars/microsoft/DeepSpeed-MII.svg?style=social) - MII makes low-latency and high-throughput inference possible, powered by DeepSpeed.
- [FlexGen](https://github.com/FMInference/FlexGen) ![](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social) - Running large language models on a single GPU for throughput-oriented scenarios.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) ![](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=social) - Port of Facebook's LLaMA model in C/C++
- [MOSEC](https://github.com/mosecorg/mosec) ![](https://img.shields.io/github/stars/mosecorg/mosec?style=social) - A machine learning model serving framework with dynamic batching and pipelined stages, provides an easy-to-use Python interface.
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) ![](https://img.shields.io/github/stars/ggerganov/whisper.cpp.svg?style=social) - Port of OpenAI's Whisper model in C/C++
- [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion) ![](https://img.shields.io/github/stars/stochasticai/x-stable-diffusion.svg?style=social) - Real-time inference for Stable Diffusion - 0.88s latency. Covers AITemplate, nvFuser, TensorRT, FlashAttention.

**[⬆ back to ToC](#table-of-contents)**

## Optimizations

- [FeatherCNN](https://github.com/Tencent/FeatherCNN) ![](https://img.shields.io/github/stars/Tencent/FeatherCNN.svg?style=social) - FeatherCNN is a high performance inference engine for convolutional neural networks.
- [Forward](https://github.com/Tencent/Forward) ![](https://img.shields.io/github/stars/Tencent/Forward.svg?style=social) - A library for high performance deep learning inference on NVIDIA GPUs.
- [NCNN](https://github.com/Tencent/ncnn) ![](https://img.shields.io/github/stars/Tencent/ncnn.svg?style=social) - ncnn is a high-performance neural network inference framework optimized for the mobile platform.
- [PocketFlow](https://github.com/Tencent/PocketFlow) ![](https://img.shields.io/github/stars/Tencent/PocketFlow.svg?style=social) - use AutoML to do model compression.
- [TensorFlow Model Optimization](https://github.com/tensorflow/model-optimization) ![](https://img.shields.io/github/stars/tensorflow/model-optimization.svg?style=social) - A suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution.
- [TNN](https://github.com/Tencent/TNN) ![](https://img.shields.io/github/stars/Tencent/TNN.svg?style=social) - A uniform deep learning inference framework for mobile, desktop and server.

**[⬆ back to ToC](#table-of-contents)**

## Observability

- [Deepchecks](https://github.com/deepchecks/deepchecks) ![](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=social) - Tests for Continuous Validation of ML Models & Data. Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.
- [Evidently](https://github.com/evidentlyai/evidently) ![](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=social) - Evaluate and monitor ML models from validation to production.
- [Great Expectations](https://github.com/great-expectations/great_expectations) ![](https://img.shields.io/github/stars/great-expectations/great_expectations.svg?style=social) - Always know what to expect from your data.
- [whylogs](https://github.com/whylabs/whylogs) ![](https://img.shields.io/github/stars/whylabs/whylogs.svg?style=social) - The open standard for data logging

**[⬆ back to ToC](#table-of-contents)**

# LLMOps

- [deeplake](https://github.com/activeloopai/deeplake) ![](https://img.shields.io/github/stars/activeloopai/Hub.svg?style=social) - Stream large multimodal datasets to achieve near 100% GPU utilization. Query, visualize, & version control data. Access data w/o the need to recompute the embeddings for the model finetuning.
- [Haystack](https://github.com/deepset-ai/haystack) ![](https://img.shields.io/github/stars/deepset-ai/haystack.svg?style=social) - Quickly compose applications with LLM Agents, semantic search, question-answering and more. 
- [langchain](https://github.com/hwchase17/langchain) ![](https://img.shields.io/github/stars/hwchase17/langchain.svg?style=social) - Building applications with LLMs through composability
- [LlamaIndex](https://github.com/jerryjliu/llama_index) ![](https://img.shields.io/github/stars/jerryjliu/llama_index.svg?style=social) - Provides a central interface to connect your LLMs with external data.
- [xTuring](https://github.com/stochasticai/xturing) ![](https://img.shields.io/github/stars/stochasticai/xturing.svg?style=social) - Build and control your personal LLMs with fast and efficient fine-tuning.
- [ZenML](https://github.com/zenml-io/zenml) ![](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=social) - Open-source framework for orchestrating, experimenting and deploying production-grade ML solutions, with built-in `langchain` & `llama_index` integrations.

**[⬆ back to ToC](#table-of-contents)**

# Search

## Vector search

- [AquilaDB](https://github.com/Aquila-Network/AquilaDB) ![](https://img.shields.io/github/stars/Aquila-Network/AquilaDB.svg?style=social) - An easy to use Neural Search Engine. Index latent vectors along with JSON metadata and do efficient k-NN search.
- [Chroma](https://github.com/chroma-core/chroma) ![](https://img.shields.io/github/stars/chroma-core/chroma.svg?style=social) - the open source embedding database
- [Jina](https://github.com/jina-ai/jina) ![](https://img.shields.io/github/stars/jina-ai/jina.svg?style=social) - Build multimodal AI services via cloud native technologies · Neural Search · Generative AI · Cloud Native
- [Marqo](https://github.com/marqo-ai/marqo) ![](https://img.shields.io/github/stars/marqo-ai/marqo.svg?style=social) - Tensor search for humans.
- [Milvus](https://github.com/milvus-io/milvus) ![](https://img.shields.io/github/stars/milvus-io/milvus.svg?style=social) - Vector database for scalable similarity search and AI applications.
- [Pinecone](https://www.pinecone.io/) - The Pinecone vector database makes it easy to build high-performance vector search applications. Developer-friendly, fully managed, and easily scalable without infrastructure hassles
- [Qdrant](https://github.com/qdrant/qdrant) ![](https://img.shields.io/github/stars/qdrant/qdrant.svg?style=social) - Vector Search Engine and Database for the next generation of AI applications. Also available in the cloud
- [txtai](https://github.com/neuml/txtai) ![](https://img.shields.io/github/stars/neuml/txtai.svg?style=social) - Build AI-powered semantic search applications
- [Vald](https://github.com/vdaas/vald) ![](https://img.shields.io/github/stars/vdaas/vald.svg?style=social) - A Highly Scalable Distributed Vector Search Engine
- [Vearch](https://github.com/vearch/vearch) ![](https://img.shields.io/github/stars/vearch/vearch.svg?style=social) - A distributed system for embedding-based vector retrieval
- [Weaviate](https://github.com/semi-technologies/weaviate) ![](https://img.shields.io/github/stars/semi-technologies/weaviate.svg?style=social) - Weaviate is an open source vector search engine that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients.

**[⬆ back to ToC](#table-of-contents)**

# Data

## Data Management
- [ArtiVC](https://github.com/InfuseAI/ArtiVC) ![](https://img.shields.io/github/stars/InfuseAI/ArtiVC.svg?style=social) - A version control system to manage large files.
Lake is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size.
- [Dolt](https://github.com/dolthub/dolt) ![](https://img.shields.io/github/stars/dolthub/dolt.svg?style=social) - Git for Data.
- [DVC](https://github.com/iterative/dvc) ![](https://img.shields.io/github/stars/iterative/dvc.svg?style=social) - Data Version Control | Git for Data & Models | ML Experiments Management.
- [Delta-Lake](https://github.com/delta-io/delta) ![](https://img.shields.io/github/stars/delta-io/delta.svg?style=social) - Storage layer that brings scalable, ACID transactions to Apache Spark and other engines.
- [Pachyderm](https://github.com/pachyderm/pachyderm) ![](https://img.shields.io/github/stars/pachyderm/pachyderm.svg?style=social) - Pachyderm is a version control system for data.
- [Quilt](https://github.com/quiltdata/quilt) ![](https://img.shields.io/github/stars/quiltdata/quilt.svg?style=social) - A self-organizing data hub for S3.

**[⬆ back to ToC](#table-of-contents)**

## Data Storage

- [JuiceFS](https://github.com/juicedata/juicefs) ![](https://img.shields.io/github/stars/juicedata/juicefs.svg?style=social) -  A distributed POSIX file system built on top of Redis and S3.
- [LakeFS](https://github.com/treeverse/lakeFS) ![](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=social) - Git-like capabilities for your object storage.
- [Lance](https://github.com/eto-ai/lance) ![](https://img.shields.io/github/stars/eto-ai/lance.svg?style=social) - Modern columnar data format for ML implemented in Rust.

**[⬆ back to ToC](#table-of-contents)**

## Data Tracking

- [Piperider](https://github.com/InfuseAI/piperider) ![](https://img.shields.io/github/stars/InfuseAI/piperider.svg?style=social) - A CLI tool that allows you to build data profiles and write assertion tests for easily evaluating and tracking your data's reliability over time.
- [LUX](https://github.com/lux-org/lux) ![](https://img.shields.io/github/stars/lux-org/lux.svg?style=social) - A Python library that facilitates fast and easy data exploration by automating the visualization and data analysis process.

**[⬆ back to ToC](#table-of-contents)**

## Feature Engineering

- [Featureform](https://github.com/featureform/featureform) ![](https://img.shields.io/github/stars/featureform/featureform.svg?style=social) - The Virtual Feature Store. Turn your existing data infrastructure into a feature store.
- [FeatureTools](https://github.com/Featuretools/featuretools) ![](https://img.shields.io/github/stars/Featuretools/featuretools.svg?style=social) - An open source python framework for automated feature engineering

**[⬆ back to ToC](#table-of-contents)**

## Data/Feature enrichment

- [Upgini](https://github.com/upgini/upgini) ![](https://img.shields.io/github/stars/upgini/upgini.svg?style=social) - Free automated data & feature enrichment library for machine learning: automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features
- [Feast](https://github.com/feast-dev/feast) ![](https://img.shields.io/github/stars/feast-dev/feast.svg?style=social) - An open source feature store for machine learning.

**[⬆ back to ToC](#table-of-contents)**

# Large Scale Deployment

## ML Platforms

- [ClearML](https://github.com/allegroai/clearml) ![](https://img.shields.io/github/stars/allegroai/clearml.svg?style=social) - Auto-Magical CI/CD to streamline your ML workflow. Experiment Manager, MLOps and Data-Management.
- [MLflow](https://github.com/mlflow/mlflow) ![](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=social) - Open source platform for the machine learning lifecycle.
- [MLRun](https://github.com/mlrun/mlrun) ![](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=social) - An open MLOps platform for quickly building and managing continuous ML applications across their lifecycle.
- [ModelFox](https://github.com/modelfoxdotdev/modelfox) ![](https://img.shields.io/github/stars/modelfoxdotdev/modelfox.svg?style=social) - ModelFox is a platform for managing and deploying machine learning models.
- [Kserve](https://github.com/kserve/kserve) ![](https://img.shields.io/github/stars/kserve/kserve.svg?style=social) - Standardized Serverless ML Inference Platform on Kubernetes
- [Kubeflow](https://github.com/kubeflow/kubeflow) ![](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=social) - Machine Learning Toolkit for Kubernetes.
- [PAI](https://github.com/microsoft/pai) ![](https://img.shields.io/github/stars/microsoft/pai.svg?style=social) - Resource scheduling and cluster management for AI.
- [Polyaxon](https://github.com/polyaxon/polyaxon) ![](https://img.shields.io/github/stars/polyaxon/polyaxon.svg?style=social) - Machine Learning Management & Orchestration Platform.
- [Primehub](https://github.com/InfuseAI/primehub) ![](https://img.shields.io/github/stars/InfuseAI/primehub.svg?style=social) - An effortless infrastructure for machine learning built on the top of Kubernetes.
- [Seldon-core](https://github.com/SeldonIO/seldon-core) ![](https://img.shields.io/github/stars/SeldonIO/seldon-core.svg?style=social) - An MLOps framework to package, deploy, monitor and manage thousands of production machine learning models

**[⬆ back to ToC](#table-of-contents)**

## Workflow

- [Airflow](https://airflow.apache.org/) ![](https://img.shields.io/github/stars/apache/airflow?style=social) - A platform to programmatically author, schedule and monitor workflows.
- [aqueduct](https://github.com/aqueducthq/aqueduct) ![](https://img.shields.io/github/stars/aqueducthq/aqueduct.svg?style=social) - An Open-Source Platform for Production Data Science
- [Argo Workflows](https://github.com/argoproj/argo-workflows) ![](https://img.shields.io/github/stars/argoproj/argo-workflows.svg?style=social) - Workflow engine for Kubernetes.
- [Flyte](https://github.com/flyteorg/flyte) ![](https://img.shields.io/github/stars/flyteorg/flyte.svg?style=social) - Kubernetes-native workflow automation platform for complex, mission-critical data and ML processes at scale.
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines) ![](https://img.shields.io/github/stars/kubeflow/pipelines.svg?style=social) - Machine Learning Pipelines for Kubeflow.
- [Metaflow](https://github.com/Netflix/metaflow) ![](https://img.shields.io/github/stars/Netflix/metaflow.svg?style=social) - Build and manage real-life data science projects with ease!
- [Ploomber](https://github.com/ploomber/ploomber) ![](https://img.shields.io/github/stars/ploomber/ploomber.svg?style=social) - The fastest way to build data pipelines. Develop iteratively, deploy anywhere.
- [Prefect](https://github.com/PrefectHQ/prefect) ![](https://img.shields.io/github/stars/PrefectHQ/prefect.svg?style=social) - The easiest way to automate your data.
- [VDP](https://github.com/instill-ai/vdp) ![](https://img.shields.io/github/stars/instill-ai/vdp.svg?style=social) - An open-source unstructured data ETL tool to streamline the end-to-end unstructured data processing pipeline.
- [ZenML](https://github.com/zenml-io/zenml) ![](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=social) - MLOps framework to create reproducible pipelines.

**[⬆ back to ToC](#table-of-contents)**

## Scheduling

- [Kueue](https://github.com/kubernetes-sigs/kueue) ![](https://img.shields.io/github/stars/kubernetes-sigs/kueue.svg?style=social) - Kubernetes-native Job Queueing.
- [PAI](https://github.com/microsoft/pai) ![](https://img.shields.io/github/stars/microsoft/pai.svg?style=social) - Resource scheduling and cluster management for AI (Open-sourced by Microsoft).
- [Slurm](https://github.com/SchedMD/slurm) ![](https://img.shields.io/github/stars/SchedMD/slurm.svg?style=social) - A Highly Scalable Workload Manager.
- [Volcano](https://github.com/volcano-sh/volcano) ![](https://img.shields.io/github/stars/volcano-sh/volcano.svg?style=social) - A Cloud Native Batch System (Project under CNCF).
- [Yunikorn](https://github.com/apache/yunikorn-core) ![](https://img.shields.io/github/stars/apache/yunikorn-core.svg?style=social) - Light-weight, universal resource scheduler for container orchestrator systems.

**[⬆ back to ToC](#table-of-contents)**

# Performance

## ML Compiler

- [ONNX-MLIR](https://github.com/onnx/onnx-mlir) ![](https://img.shields.io/github/stars/onnx/onnx-mlir.svg?style=social) - Compiler technology to transform a valid Open Neural Network Exchange (ONNX) graph into code that implements the graph with minimum runtime support.
- [TVM](https://github.com/apache/tvm) ![](https://img.shields.io/github/stars/apache/tvm.svg?style=social) - Open deep learning compiler stack for cpu, gpu and specialized accelerators

**[⬆ back to ToC](#table-of-contents)**

## Profiling

- [octoml-profile](https://github.com/octoml/octoml-profile) ![](https://img.shields.io/github/stars/octoml/octoml-profile.svg?style=social) - octoml-profile is a python library and cloud service designed to provide the simplest experience for assessing and optimizing the performance of PyTorch models on cloud hardware with state-of-the-art ML acceleration technology.
- [scalene](https://github.com/plasma-umass/scalene) ![](https://img.shields.io/github/stars/plasma-umass/scalene.svg?style=social) - a high-performance, high-precision CPU, GPU, and memory profiler for Python

**[⬆ back to ToC](#table-of-contents)**

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
- [Pycaret](https://github.com/pycaret/pycaret) ![](https://img.shields.io/github/stars/pycaret/pycaret.svg?style=social) - An open-source, low-code machine learning library in Python that automates machine learning workflows.
- [Ray Tune](github.com/ray-project/ray) ![](https://img.shields.io/github/stars/ect/ray.svg?style=social) - Scalable Hyperparameter Tuning.
- [REMBO](https://github.com/ziyuw/rembo) ![](https://img.shields.io/github/stars/ziyuw/rembo.svg?style=social) - Bayesian optimization in high-dimensions via random embedding.
- [RoBO](https://github.com/automl/RoBO) ![](https://img.shields.io/github/stars/automl/RoBO.svg?style=social) - a Robust Bayesian Optimization framework.
- [scikit-optimize(skopt)](https://github.com/scikit-optimize/scikit-optimize) ![](https://img.shields.io/github/stars/scikit-optimize/scikit-optimize.svg?style=social) - Sequential model-based optimization with a `scipy.optimize` interface.
- [Spearmint](https://github.com/HIPS/Spearmint) ![](https://img.shields.io/github/stars/HIPS/Spearmint.svg?style=social) - a software package to perform Bayesian optimization.
- [TPOT](http://automl.info/tpot/) ![](https://img.shields.io/github/stars/tpot/.svg?style=social) - one of the very first AutoML methods and open-source software packages.
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta) ![](https://img.shields.io/github/stars/tristandeleu/pytorch-meta.svg?style=social) - A Meta-Learning library for PyTorch.
- [Vegas](https://github.com/huawei-noah/vega) ![](https://img.shields.io/github/stars/huawei-noah/vega.svg?style=social) - an AutoML algorithm tool chain by Huawei Noah's Arb Lab.

**[⬆ back to ToC](#table-of-contents)**

# Federated ML

- [EasyFL](https://github.com/EasyFL-AI/EasyFL) ![](https://img.shields.io/github/stars/EasyFL-AI/EasyFL.svg?style=social) - An Easy-to-use Federated Learning Platform
- [FATE](https://github.com/FederatedAI/FATE) ![](https://img.shields.io/github/stars/FederatedAI/FATE.svg?style=social) - An Industrial Grade Federated Learning Framework
- [FedML](https://github.com/FedML-AI/FedML) ![](https://img.shields.io/github/stars/FedML-AI/FedML.svg?style=social) - The federated learning and analytics library enabling secure and collaborative machine learning on decentralized data anywhere at any scale. Supporting large-scale cross-silo federated learning, cross-device federated learning on smartphones/IoTs, and research simulation.
- [Flower](https://github.com/adap/flower) ![](https://img.shields.io/github/stars/adap/flower.svg?style=social) - A Friendly Federated Learning Framework
- [Harmonia](https://github.com/ailabstw/harmonia) ![](https://img.shields.io/github/stars/ailabstw/harmonia.svg?style=social) - Harmonia is an open-source project aiming at developing systems/infrastructures and libraries to ease the adoption of federated learning (abbreviated to FL) for researches and production usage.
- [TensorFlow Federated](https://github.com/tensorflow/federated) ![](https://img.shields.io/github/stars/tensorflow/federated.svg?style=social) - A framework for implementing federated learning

**[⬆ back to ToC](#table-of-contents)**

# Awesome Lists

- [Awesome Argo](https://github.com/terrytangyuan/awesome-argo) ![](https://img.shields.io/github/stars/terrytangyuan/awesome-argo.svg?style=social) - A curated list of awesome projects and resources related to Argo
- [Awesome AutoDL](https://github.com/D-X-Y/Awesome-AutoDL) ![](https://img.shields.io/github/stars/D-X-Y/Awesome-AutoDL.svg?style=social) - Automated Deep Learning: Neural Architecture Search Is Not the End (a curated list of AutoDL resources and an in-depth analysis)
- [Awesome AutoML](https://github.com/windmaple/awesome-AutoML) ![](https://img.shields.io/github/stars/windmaple/awesome-AutoML.svg?style=social) - Curating a list of AutoML-related research, tools, projects and other resources
- [Awesome AutoML Papers](https://github.com/hibayesian/awesome-automl-papers) ![](https://img.shields.io/github/stars/hibayesian/awesome-automl-papers.svg?style=social) - A curated list of automated machine learning papers, articles, tutorials, slides and projects
- [Awesome Federated Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning) ![](https://img.shields.io/github/stars/chaoyanghe/Awesome-Federated-Learning.svg?style=social) - A curated list of federated learning publications, re-organized from Arxiv (mostly)
- [awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning) ![](https://img.shields.io/github/stars/weimingwill/awesome-federated-learning.svg?style=social) - All materials you need for Federated Learning: blogs, videos, papers, and softwares, etc.
- [Awesome Open MLOps](https://github.com/fuzzylabs/awesome-open-mlops) ![](https://img.shields.io/github/stars/fuzzylabs/awesome-open-mlops.svg?style=social) - This is the Fuzzy Labs guide to the universe of free and open source MLOps tools.
- [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning) ![](https://img.shields.io/github/stars/EthicalML/awesome-production-machine-learning.svg?style=social) - A curated list of awesome open source libraries to deploy, monitor, version and scale your machine learning
- [Awesome Tensor Compilers](https://github.com/merrymercy/awesome-tensor-compilers) ![](https://img.shields.io/github/stars/merrymercy/awesome-tensor-compilers.svg?style=social) - A list of awesome compiler projects and papers for tensor computation and deep learning.
- [kelvins/awesome-mlops](https://github.com/kelvins/awesome-mlops) ![](https://img.shields.io/github/stars/kelvins/awesome-mlops.svg?style=social) - A curated list of awesome MLOps tools.
- [visenger/awesome-mlops](https://github.com/visenger/awesome-mlops) ![](https://img.shields.io/github/stars/visenger/awesome-mlops.svg?style=social) - An awesome list of references for MLOps - Machine Learning Operations
- [currentslab/awesome-vector-search](https://github.com/currentslab/awesome-vector-search) ![](https://img.shields.io/github/stars/currentslab/awesome-vector-search.svg?style=social) - A curated list of awesome vector search framework/engine, library, cloud service and research papers to vector similarity search.

**[⬆ back to ToC](#table-of-contents)**
