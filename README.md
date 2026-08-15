# Awesome LLMOps

<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://img.shields.io/discord/974584200327991326?style=flat&logo=discord&cacheSeconds=60"></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge-flat2.svg"></a>

An awesome & curated list of the best LLMOps tools for developers.

> [!NOTE]
> Contributions are most welcome, please adhere to the [contribution guidelines](contributing.md).

## Table of Contents

- [Awesome LLMOps](#awesome-llmops)
  - [Table of Contents](#table-of-contents)
  - [Model](#model)
    - [Large Language Model](#large-language-model)
    - [CV Foundation Model](#cv-foundation-model)
    - [Audio Foundation Model](#audio-foundation-model)
    - [Robotics Foundation Model](#robotics-foundation-model)
  - [Serving](#serving)
    - [Large Model Serving](#large-model-serving)
    - [Frameworks/Servers for Serving](#frameworksservers-for-serving)
  - [Security](#security)
    - [Frameworks for LLM security](#frameworks-for-llm-security)
    - [Observability](#observability)
  - [LLMOps](#llmops)
  - [Search](#search)
    - [Vector search](#vector-search)
  - [Code AI](#code-ai)
  - [Training](#training)
    - [IDEs and Workspaces](#ides-and-workspaces)
    - [Foundation Model Fine Tuning](#foundation-model-fine-tuning)
    - [Frameworks for Training](#frameworks-for-training)
    - [Experiment Tracking](#experiment-tracking)
    - [Visualization](#visualization)
    - [Model Editing](#model-editing)
  - [Data](#data)
    - [Data Management](#data-management)
    - [Data Storage](#data-storage)
    - [Data Tracking](#data-tracking)
    - [Feature Engineering](#feature-engineering)
    - [Data/Feature enrichment](#datafeature-enrichment)
  - [Large Scale Deployment](#large-scale-deployment)
    - [ML Platforms](#ml-platforms)
    - [Workflow](#workflow)
    - [Scheduling](#scheduling)
    - [Model Management](#model-management)
  - [Performance](#performance)
    - [ML Compiler](#ml-compiler)
    - [Profiling](#profiling)
  - [AutoML](#automl)
  - [Optimizations](#optimizations)
  - [Federated ML](#federated-ml)
  - [Awesome Lists](#awesome-lists)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## Model

### Large Language Model

| Project                                                                 | Details                                                                                                                                                                                    | Repository                                                                                                |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                  | Code and documentation to train Stanford's Alpaca models, and generate the data.                                                                                                           | ![GitHub Badge](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca.svg?style=flat-square)      |
| [BELLE](https://github.com/LianjiaTech/BELLE)                           | A 7B Large Language Model fine-tune by 34B Chinese Character Corpus, based on LLaMA and Alpaca.                                                                                            | ![GitHub Badge](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg?style=flat-square)              |
| [Bloom](https://github.com/bigscience-workshop/model_card)              | BigScience Large Open-science Open-access Multilingual Language Model                                                                                                                      | ![GitHub Badge](https://img.shields.io/github/stars/bigscience-workshop/model_card.svg?style=flat-square) |
| [dolly](https://github.com/databrickslabs/dolly)                        | Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform                                                                                              | ![GitHub Badge](https://img.shields.io/github/stars/databrickslabs/dolly.svg?style=flat-square)           |
| [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b-instruct)         | Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license. |                                                                                                           |
| [FastChat (Vicuna)](https://github.com/lm-sys/FastChat)                 | An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5.                                                                     | ![GitHub Badge](https://img.shields.io/github/stars/lm-sys/FastChat.svg?style=flat-square)                |
| [Gemma](https://www.kaggle.com/models/google/gemma)                     | Gemma is a family of lightweight, open models built from the research and technology that Google used to create the Gemini models.                                                         |                                                                                                           |
| [GLM-6B (ChatGLM)](https://github.com/THUDM/ChatGLM-6B)                 | An Open Bilingual Pre-Trained Model, quantization of ChatGLM-130B, can run on consumer-level GPUs.                                                                                         | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg?style=flat-square)               |
| [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)                     | ChatGLM2-6B is the second-generation version of the open-source bilingual (Chinese-English) chat model [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B).                                  | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg?style=flat-square)              |
| [GLM-130B (ChatGLM)](https://github.com/THUDM/GLM-130B)                 | An Open Bilingual Pre-Trained Model (ICLR 2023)                                                                                                                                            | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/GLM-130B.svg?style=flat-square)                 |
| [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)                      | An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.                                                                                   | ![GitHub Badge](https://img.shields.io/github/stars/EleutherAI/gpt-neox.svg?style=flat-square)            |
| [Luotuo](https://github.com/LC1332/Luotuo-Chinese-LLM)                  | A Chinese LLM, Based on LLaMA and fine tune by Stanford Alpaca, Alpaca LoRA, Japanese-Alpaca-LoRA.                                                                                         | ![GitHub Badge](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM.svg?style=flat-square)      |
| [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.                                                                                          |                                                                                                           |
| [StableLM](https://github.com/Stability-AI/StableLM)                    | StableLM: Stability AI Language Models                                                                                                                                                     | ![GitHub Badge](https://img.shields.io/github/stars/Stability-AI/StableLM.svg?style=flat-square)          |

**[⬆ back to ToC](#table-of-contents)**

### CV Foundation Model

| Project                                                                        | Details                                                                                                                                          | Repository                                                                                                   |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| [disco-diffusion](https://github.com/alembics/disco-diffusion)                 | A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations.                                  | ![GitHub Badge](https://img.shields.io/github/stars/alembics/disco-diffusion.svg?style=flat-square)          |
| [midjourney](https://www.midjourney.com/home/)                                 | Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.            |                                                                                                              |
| [segment-anything (SAM)](https://github.com/facebookresearch/segment-anything) | produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. | ![GitHub Badge](https://img.shields.io/github/stars/facebookresearch/segment-anything.svg?style=flat-square) |
| [stable-diffusion](https://github.com/CompVis/stable-diffusion)                | A latent text-to-image diffusion model                                                                                                           | ![GitHub Badge](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?style=flat-square)          |

**[⬆ back to ToC](#table-of-contents)**

### Audio Foundation Model

| Project                                      | Details                                                                                                                                                                                                       | Repository                                                                                |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [bark](https://github.com/suno-ai/bark)      | Bark is a transformer-based text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. | ![GitHub Badge](https://img.shields.io/github/stars/suno-ai/bark.svg?style=flat-square)   |
| [whisper](https://github.com/openai/whisper) | Robust Speech Recognition via Large-Scale Weak Supervision                                                                                                                                                    | ![GitHub Badge](https://img.shields.io/github/stars/openai/whisper.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Robotics Foundation Model

> [!NOTE]
> **Emerging Architectures in VLA:**
> - **Continuous Diffusion Language Models:** Integrate diffusion heads or flow-matching to VLMs (e.g., DiVLA, OpenPI), enabling smooth, precise continuous action generation rather than discretized tokens.
> - **Recurrent Language Models:** Utilize State Space Models (SSMs) like Mamba or recurrent transformers (e.g., RoboMamba, RD-VLA) to reduce inference memory and handle temporal dependencies, allowing iterative reasoning for complex robotic decision-making.

| Project                                                   | Details                                                                                                                                                                                                    | Repository                                                                                             |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [DiVLA](https://github.com/hustvl/DiVLA)                  | A continuous diffusion-based Vision-Language-Action model that integrates diffusion policies into autoregressive VLMs for robust and precise continuous robotic control.                                   | ![GitHub Badge](https://img.shields.io/github/stars/hustvl/DiVLA.svg?style=flat-square)                |
| [LeRobot](https://github.com/huggingface/lerobot)         | A central community library by Hugging Face for AI in robotics — end-to-end learning tools, data pipelines, and support for training/deploying VLA models.                                                 | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/lerobot.svg?style=flat-square)         |
| [Octo](https://github.com/octo-models/octo)               | A transformer-based generalist robot policy pretrained on 800K+ robot trajectories from the Open X-Embodiment dataset. Supports language instructions, goal images, and fine-tuning to new embodiments.      | ![GitHub Badge](https://img.shields.io/github/stars/octo-models/octo.svg?style=flat-square)            |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | Open-source VLA models from Physical Intelligence, including π₀ and π₀.5 — flow-based vision-language-action models pretrained on large-scale robot data with fine-tuning support.                         | ![GitHub Badge](https://img.shields.io/github/stars/Physical-Intelligence/openpi.svg?style=flat-square) |
| [OpenVLA](https://github.com/openvla/openvla)             | A 7B-parameter open-source Vision-Language-Action model trained on 970K+ robot demonstrations from the Open X-Embodiment dataset for generalist robotic manipulation.                                      | ![GitHub Badge](https://img.shields.io/github/stars/openvla/openvla.svg?style=flat-square)             |
| [RoboMamba](https://github.com/hustvl/RoboMamba)          | An efficient VLA model leveraging State Space Models (Mamba) instead of standard self-attention, offering linear inference complexity for efficient, recurrent robotic reasoning.                          | ![GitHub Badge](https://img.shields.io/github/stars/hustvl/RoboMamba.svg?style=flat-square)            |
| [SmolVLA](https://huggingface.co/blog/smolvla)            | A compact ~450M parameter VLA by Hugging Face, designed to be computationally efficient and accessible, running on consumer GPUs or CPUs. Part of the LeRobot ecosystem.                                   |                                                                                                        |

## Serving

### Large Model Serving

| Project                                                                               | Details                                                                                                         | Repository                                                                                                       |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)                  | Alpaca-LoRA as Chatbot service                                                                                  | ![GitHub Badge](https://img.shields.io/github/stars/deep-diver/Alpaca-LoRA-Serve.svg?style=flat-square)          |
| [OneComp](https://github.com/FujitsuResearch/OneCompression)                               | Fujitsu Research's post-training quantization pipeline for LLMs (QEP, AutoBit, JointQ, rotation) with vLLM plugin (arXiv:2603.28845).                             | ![GitHub Badge](https://img.shields.io/github/stars/FujitsuResearch/OneCompression.svg?style=flat-square)        |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2)                                 | fast inference engine for Transformer models in C++                                                             | ![GitHub Badge](https://img.shields.io/github/stars/OpenNMT/CTranslate2.svg?style=flat-square)                   |
| [Clip-as-a-service](https://github.com/jina-ai/clip-as-service)                       | serving the OpenAI CLIP model                                                                                   | ![GitHub Badge](https://img.shields.io/github/stars/jina-ai/clip-as-service.svg?style=flat-square)               |
| [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)                           | MII makes low-latency and high-throughput inference possible, powered by DeepSpeed.                             | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/DeepSpeed-MII.svg?style=flat-square)               |
| [Faster Whisper](https://github.com/guillaumekln/faster-whisper)                      | fast inference engine for whisper in C++ using CTranslate2.                                                     | ![GitHub Badge](https://img.shields.io/github/stars/guillaumekln/faster-whisper.svg?style=flat-square)           |
| [FlexGen](https://github.com/FMInference/FlexGen)                                     | Running large language models on a single GPU for throughput-oriented scenarios. *(Archived)*                   | ![GitHub Badge](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=flat-square)                   |
| [Flowise](https://github.com/FlowiseAI/Flowise)                                       | Drag & drop UI to build your customized LLM flow using LangchainJS.                                             | ![GitHub Badge](https://img.shields.io/github/stars/FlowiseAI/Flowise.svg?style=flat-square)                     |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)                                   | Port of Facebook's LLaMA model in C/C++                                                                         | ![GitHub Badge](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=flat-square)                   |
| [LLMKube](https://github.com/defilantech/LLMKube)                                     | Kubernetes operator for LLM inference with pluggable runtimes (llama.cpp, PersonaPlex/Moshi, generic), multi-GPU sharding, NVIDIA CUDA and Apple Silicon Metal support, and GGUF/MLX/SafeTensors model formats. | ![GitHub Badge](https://img.shields.io/github/stars/defilantech/LLMKube.svg?style=flat-square)                   |
| [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy)                               | Python-free Rust inference server with OpenAI API compatibility and hot model swapping                        | ![GitHub …39392 tokens truncated…rn2learn](https://github.com/learnables/learn2learn)                     | PyTorch Meta-learning Framework for Researchers.                                                                                                                                | ![GitHub Badge](https://img.shields.io/github/stars/learnables/learn2learn.svg?style=flat-square)          |
| [Ludwig](https://github.com/uber/ludwig)                                     | a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code.                                                         | ![GitHub Badge](https://img.shields.io/github/stars/uber/ludwig.svg?style=flat-square)                     |
| [MOE](https://github.com/Yelp/MOE)                                           | a global, black box optimization engine for real world metric optimization by Yelp.                                                                                             | ![GitHub Badge](https://img.shields.io/github/stars/Yelp/MOE.svg?style=flat-square)                        |
| [Model Search](https://github.com/google/model_search)                       | a framework that implements AutoML algorithms for model architecture search at scale.                                                                                           | ![GitHub Badge](https://img.shields.io/github/stars/google/model_search.svg?style=flat-square)             |
| [NASGym](https://github.com/gomerudo/nas-env)                                | a proof-of-concept OpenAI Gym environment for Neural Architecture Search (NAS).                                                                                                 | ![GitHub Badge](https://img.shields.io/github/stars/gomerudo/nas-env.svg?style=flat-square)                |
| [NNI](https://github.com/Microsoft/nni)                                      | An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning. | ![GitHub Badge](https://img.shields.io/github/stars/Microsoft/nni.svg?style=flat-square)                   |
| [Optuna](https://github.com/optuna/optuna)                                   | A hyperparameter optimization framework.                                                                                                                                        | ![GitHub Badge](https://img.shields.io/github/stars/optuna/optuna.svg?style=flat-square)                   |
| [Pycaret](https://github.com/pycaret/pycaret)                                | An open-source, low-code machine learning library in Python that automates machine learning workflows.                                                                          | ![GitHub Badge](https://img.shields.io/github/stars/pycaret/pycaret.svg?style=flat-square)                 |
| [Ray Tune](github.com/ray-project/ray)                                       | Scalable Hyperparameter Tuning.                                                                                                                                                 | ![GitHub Badge](https://img.shields.io/github/stars/ray-project/ray.svg?style=flat-square)                 |
| [REMBO](https://github.com/ziyuw/rembo)                                      | Bayesian optimization in high-dimensions via random embedding.                                                                                                                  | ![GitHub Badge](https://img.shields.io/github/stars/ziyuw/rembo.svg?style=flat-square)                     |
| [RoBO](https://github.com/automl/RoBO)                                       | a Robust Bayesian Optimization framework.                                                                                                                                       | ![GitHub Badge](https://img.shields.io/github/stars/automl/RoBO.svg?style=flat-square)                     |
| [scikit-optimize(skopt)](https://github.com/scikit-optimize/scikit-optimize) | Sequential model-based optimization with a `scipy.optimize` interface.                                                                                                          | ![GitHub Badge](https://img.shields.io/github/stars/scikit-optimize/scikit-optimize.svg?style=flat-square) |
| [Spearmint](https://github.com/HIPS/Spearmint)                               | a software package to perform Bayesian optimization.                                                                                                                            | ![GitHub Badge](https://img.shields.io/github/stars/HIPS/Spearmint.svg?style=flat-square)                  |
| [TPOT](http://automl.info/tpot/)                                             | one of the very first AutoML methods and open-source software packages.                                                                                                         | ![GitHub Badge](https://img.shields.io/github/stars/EpistasisLab/tpot.svg?style=flat-square)               |
| [Torchmeta](https://github.com/tristandeleu/pytorch-meta)                    | A Meta-Learning library for PyTorch.                                                                                                                                            | ![GitHub Badge](https://img.shields.io/github/stars/tristandeleu/pytorch-meta.svg?style=flat-square)       |
| [Vegas](https://github.com/huawei-noah/vega)                                 | an AutoML algorithm tool chain by Huawei Noah's Arb Lab.                                                                                                                        | ![GitHub Badge](https://img.shields.io/github/stars/huawei-noah/vega.svg?style=flat-square)                |

**[⬆ back to ToC](#table-of-contents)**

## Optimizations

| Project                                                                           | Details                                                                                                                          | Repository                                                                                               |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Entroly](https://github.com/juyterman1000/entroly)                               | Information-theoretic context optimization proxy. Cuts LLM token costs by 70–95% with zero accuracy loss using greedy submodular knapsack maximization. | ![GitHub Badge](https://img.shields.io/github/stars/juyterman1000/entroly.svg?style=flat-square) |
| [FeatherCNN](https://github.com/Tencent/FeatherCNN)                               | FeatherCNN is a high performance inference engine for convolutional neural networks.                                             | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/FeatherCNN.svg?style=flat-square)            |
| [Forward](https://github.com/Tencent/Forward)                                     | A library for high performance deep learning inference on NVIDIA GPUs.                                                           | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/Forward.svg?style=flat-square)               |
| [LangWatch](https://github.com/langwatch/langwatch)                               | LangWatch Optimization Studio is your laboratory to create, evaluate, and optimize your LLM workflows using DSPy optimizers | ![GitHub Badge](https://img.shields.io/github/stars/langwatch/langwatch.svg?style=flat-square) |
| [lean-ctx](https://github.com/yvgude/lean-ctx)                                    | Context runtime and MCP server that reduces AI coding agent token costs via session caching, AST-aware compression, and shell output patterns. [Website](https://leanctx.com) | ![GitHub Badge](https://img.shields.io/github/stars/yvgude/lean-ctx.svg?style=flat-square) |
| [NCNN](https://github.com/Tencent/ncnn)                                           | ncnn is a high-performance neural network inference framework optimized for the mobile platform.                                 | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/ncnn.svg?style=flat-square)                  |
| [PocketFlow](https://github.com/Tencent/PocketFlow)                               | use AutoML to do model compression.                                                                                              | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/PocketFlow.svg?style=flat-square)            |
| [TensorFlow Model Optimization](https://github.com/tensorflow/model-optimization) | A suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution. | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/model-optimization.svg?style=flat-square) |
| [TNN](https://github.com/Tencent/TNN)                                             | A uniform deep learning inference framework for mobile, desktop and server.                                                      | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/TNN.svg?style=flat-square)                   |
| [optimum-tpu](https://github.com/huggingface/optimum-tpu)                         | Google TPU optimizations for transformers models                                                                                 | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/optimum-tpu.svg?style=flat-square)       |
| [agent-opt](https://github.com/future-agi/agent-opt) | Automated optimization engine for improving agent workflows using feedback-driven iterative refinements. | ![GitHub Badge](https://img.shields.io/github/stars/future-agi/agent-opt?style=flat-square) |


**[⬆ back to ToC](#table-of-contents)**

## Federated ML

| Project                                                         | Details                                                                                                                                                                                                                                                                          | Repository                                                                                      |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [EasyFL](https://github.com/EasyFL-AI/EasyFL)                   | An Easy-to-use Federated Learning Platform                                                                                                                                                                                                                                       | ![GitHub Badge](https://img.shields.io/github/stars/EasyFL-AI/EasyFL.svg?style=flat-square)     |
| [FATE](https://github.com/FederatedAI/FATE)                     | An Industrial Grade Federated Learning Framework                                                                                                                                                                                                                                 | ![GitHub Badge](https://img.shields.io/github/stars/FederatedAI/FATE.svg?style=flat-square)     |
| [FedML](https://github.com/FedML-AI/FedML)                      | The federated learning and analytics library enabling secure and collaborative machine learning on decentralized data anywhere at any scale. Supporting large-scale cross-silo federated learning, cross-device federated learning on smartphones/IoTs, and research simulation. | ![GitHub Badge](https://img.shields.io/github/stars/FedML-AI/FedML.svg?style=flat-square)       |
| [Flower](https://github.com/adap/flower)                        | A Friendly Federated Learning Framework                                                                                                                                                                                                                                          | ![GitHub Badge](https://img.shields.io/github/stars/adap/flower.svg?style=flat-square)          |
| [Harmonia](https://github.com/ailabstw/harmonia)                | Harmonia is an open-source project aiming at developing systems/infrastructures and libraries to ease the adoption of federated learning (abbreviated to FL) for researches and production usage.                                                                                | ![GitHub Badge](https://img.shields.io/github/stars/ailabstw/harmonia.svg?style=flat-square)    |
| [TensorFlow Federated](https://github.com/tensorflow/federated) | A framework for implementing federated learning                                                                                                                                                                                                                                  | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/federated.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Awesome Lists

| Project                                                                                                 | Details                                                                                                                           | Repository                                                                                                               |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Awesome Argo](https://github.com/terrytangyuan/awesome-argo)                                           | A curated list of awesome projects and resources related to Argo                                                                  | ![GitHub Badge](https://img.shields.io/github/stars/terrytangyuan/awesome-argo.svg?style=flat-square)                    |
| [Awesome AutoDL](https://github.com/D-X-Y/Awesome-AutoDL)                                               | Automated Deep Learning: Neural Architecture Search Is Not the End (a curated list of AutoDL resources and an in-depth analysis)  | ![GitHub Badge](https://img.shields.io/github/stars/D-X-Y/Awesome-AutoDL.svg?style=flat-square)                          |
| [Awesome AutoML](https://github.com/windmaple/awesome-AutoML)                                           | Curating a list of AutoML-related research, tools, projects and other resources                                                   | ![GitHub Badge](https://img.shields.io/github/stars/windmaple/awesome-AutoML.svg?style=flat-square)                      |
| [Awesome AutoML Papers](https://github.com/hibayesian/awesome-automl-papers)                            | A curated list of automated machine learning papers, articles, tutorials, slides and projects                                     | ![GitHub Badge](https://img.shields.io/github/stars/hibayesian/awesome-automl-papers.svg?style=flat-square)              |
| [Awesome-Code-LLM](https://github.com/huybery/Awesome-Code-LLM)                                         | 👨‍💻 An awesome and curated list of best code-LLM for research.                                                                     | ![GitHub Badge](https://img.shields.io/github/stars/huybery/Awesome-Code-LLM.svg?style=flat-square)                      |
| [Awesome Federated Learning Systems](https://github.com/AmberLJC/FLsystem-paper/blob/main/README.md)    | A curated list of Federated Learning Systems related academic papers, articles, tutorials, slides and projects.                   | ![GitHub Badge](https://img.shields.io/github/stars/AmberLJC/FLsystem-paper.svg?style=flat-square)                       |
| [Awesome Federated Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning)                  | A curated list of federated learning publications, re-organized from Arxiv (mostly)                                               | ![GitHub Badge](https://img.shields.io/github/stars/chaoyanghe/Awesome-Federated-Learning.svg?style=flat-square)         |
| [awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning)acc              | All materials you need for Federated Learning: blogs, videos, papers, and softwares, etc.                                         | ![GitHub Badge](https://img.shields.io/github/stars/weimingwill/awesome-federated-learning.svg?style=flat-square)        |
| [Awesome Open MLOps](https://github.com/fuzzylabs/awesome-open-mlops)                                   | This is the Fuzzy Labs guide to the universe of free and open source MLOps tools.                                                 | ![GitHub Badge](https://img.shields.io/github/stars/fuzzylabs/awesome-open-mlops.svg?style=flat-square)                  |
| [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning) | A curated list of awesome open source libraries to deploy, monitor, version and scale your machine learning                       | ![GitHub Badge](https://img.shields.io/github/stars/EthicalML/awesome-production-machine-learning.svg?style=flat-square) |
| [Awesome Tensor Compilers](https://github.com/merrymercy/awesome-tensor-compilers)                      | A list of awesome compiler projects and papers for tensor computation and deep learning.                                          | ![GitHub Badge](https://img.shields.io/github/stars/merrymercy/awesome-tensor-compilers.svg?style=flat-square)           |
| [kelvins/awesome-mlops](https://github.com/kelvins/awesome-mlops)                                       | A curated list of awesome MLOps tools.                                                                                            | ![GitHub Badge](https://img.shields.io/github/stars/kelvins/awesome-mlops.svg?style=flat-square)                         |
| [visenger/awesome-mlops](https://github.com/visenger/awesome-mlops)                                     | Machine Learning Operations - An awesome list of references for MLOps                                                             | ![GitHub Badge](https://img.shields.io/github/stars/visenger/awesome-mlops.svg?style=flat-square)                        |
| [currentslab/awesome-vector-search](https://github.com/currentslab/awesome-vector-search)               | A curated list of awesome vector search framework/engine, library, cloud service and research papers to vector similarity search. | ![GitHub Badge](https://img.shields.io/github/stars/currentslab/awesome-vector-search.svg?style=flat-square)             |
| [pleisto/flappy](https://github.com/pleisto/flappy)                                                     | Production-Ready LLM Agent SDK for Every Developer                                                                                | ![GitHub Badge](https://img.shields.io/github/stars/pleisto/flappy.svg?style=flat-square)                                |

**[⬆ back to ToC](#table-of-contents)**

