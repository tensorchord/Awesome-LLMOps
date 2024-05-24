# Awesome LLMOps

<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge-flat2.svg"></a>

An awesome & curated list of the best LLMOps tools for developers.

## Contribute

Contributions are most welcome, please adhere to the [contribution guidelines](contributing.md).

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Model](#model)
  - [Large Language Model](#large-language-model)
  - [CV Foundation Model](#cv-foundation-model)
  - [Audio Foundation Model](#audio-foundation-model)
- [Serving](#serving)
  - [Large Model Serving](#large-model-serving)
  - [Frameworks/Servers for Serving](#frameworksservers-for-serving)
  - [Observability](#observability)
- [Security](#security)
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

| Project | Details | Repository |
|---|---|---|
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | Code and documentation to train Stanford's Alpaca models, and generate the data. | ![GitHub Badge](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca.svg?style=flat-square) |
| [BELLE](https://github.com/LianjiaTech/BELLE) | A 7B Large Language Model fine-tune by 34B Chinese Character Corpus, based on LLaMA and Alpaca. | ![GitHub Badge](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg?style=flat-square) |
| [Bloom](https://github.com/bigscience-workshop/model_card) | BigScience Large Open-science Open-access Multilingual Language Model | ![GitHub Badge](https://img.shields.io/github/stars/bigscience-workshop/model_card.svg?style=flat-square) |
| [dolly](https://github.com/databrickslabs/dolly) | Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform | ![GitHub Badge](https://img.shields.io/github/stars/databrickslabs/dolly.svg?style=flat-square) |
| [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b-instruct) | Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license. | |
| [FastChat (Vicuna)](https://github.com/lm-sys/FastChat) | An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5. | ![GitHub Badge](https://img.shields.io/github/stars/lm-sys/FastChat.svg?style=flat-square) |
| [Gemma](https://www.kaggle.com/models/google/gemma) | Gemma is a family of lightweight, open models built from the research and technology that Google used to create the Gemini models.| |
| [GLM-6B (ChatGLM)](https://github.com/THUDM/ChatGLM-6B) | An Open Bilingual Pre-Trained Model, quantization of ChatGLM-130B, can run on consumer-level GPUs. | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg?style=flat-square) |
| [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) | ChatGLM2-6B is the second-generation version of the open-source bilingual (Chinese-English) chat model [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B). | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg?style=flat-square) |
| [GLM-130B (ChatGLM)](https://github.com/THUDM/GLM-130B) | An Open Bilingual Pre-Trained Model (ICLR 2023) | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/GLM-130B.svg?style=flat-square) |
| [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) | An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. | ![GitHub Badge](https://img.shields.io/github/stars/EleutherAI/gpt-neox.svg?style=flat-square) |
| [Luotuo](https://github.com/LC1332/Luotuo-Chinese-LLM) | A Chinese LLM, Based on LLaMA and fine tune by Stanford Alpaca, Alpaca LoRA, Japanese-Alpaca-LoRA. | ![GitHub Badge](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM.svg?style=flat-square) |
| [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. | |
| [StableLM](https://github.com/Stability-AI/StableLM) | StableLM: Stability AI Language Models | ![GitHub Badge](https://img.shields.io/github/stars/Stability-AI/StableLM.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### CV Foundation Model

| Project | Details | Repository |
|---|---|---|
| [disco-diffusion](https://github.com/alembics/disco-diffusion) | A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations. | ![GitHub Badge](https://img.shields.io/github/stars/alembics/disco-diffusion.svg?style=flat-square) |
| [midjourney](https://www.midjourney.com/home/) | Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species. | |
| [segment-anything (SAM)](https://github.com/facebookresearch/segment-anything) | produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. | ![GitHub Badge](https://img.shields.io/github/stars/facebookresearch/segment-anything.svg?style=flat-square) |
| [stable-diffusion](https://github.com/CompVis/stable-diffusion) | A latent text-to-image diffusion model | ![GitHub Badge](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?style=flat-square) |
| [stable-diffusion v2](https://github.com/Stability-AI/stablediffusion) | High-Resolution Image Synthesis with Latent Diffusion Models | ![GitHub Badge](https://img.shields.io/github/stars/Stability-AI/stablediffusion.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Audio Foundation Model

| Project | Details | Repository |
|---|---|---|
| [bark](https://github.com/suno-ai/bark) | Bark is a transformer-based text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. | ![GitHub Badge](https://img.shields.io/github/stars/suno-ai/bark.svg?style=flat-square) |
| [whisper](https://github.com/openai/whisper) | Robust Speech Recognition via Large-Scale Weak Supervision | ![GitHub Badge](https://img.shields.io/github/stars/openai/whisper.svg?style=flat-square) |

## Serving

### Large Model Serving

| Project | Details | Repository |
|---|---|---|
| [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve) | Alpaca-LoRA as Chatbot service | ![GitHub Badge](https://img.shields.io/github/stars/deep-diver/Alpaca-LoRA-Serve.svg?style=flat-square) |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | fast inference engine for Transformer models in C++ | ![GitHub Badge](https://img.shields.io/github/stars/OpenNMT/CTranslate2.svg?style=flat-square) |
| [Clip-as-a-service](https://github.com/jina-ai/clip-as-service) | serving the OpenAI CLIP model | ![GitHub Badge](https://img.shields.io/github/stars/jina-ai/clip-as-service.svg?style=flat-square) |
| [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) | MII makes low-latency and high-throughput inference possible, powered by DeepSpeed. | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/DeepSpeed-MII.svg?style=flat-square) |
| [Faster Whisper](https://github.com/guillaumekln/faster-whisper) | fast inference engine for whisper in C++ using CTranslate2. | ![GitHub Badge](https://img.shields.io/github/stars/guillaumekln/faster-whisper.svg?style=flat-square) |
| [FlexGen](https://github.com/FMInference/FlexGen) | Running large language models on a single GPU for throughput-oriented scenarios. | ![GitHub Badge](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=flat-square) |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag & drop UI to build your customized LLM flow using LangchainJS. | ![GitHub Badge](https://img.shields.io/github/stars/FlowiseAI/Flowise.svg?style=flat-square) |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Port of Facebook's LLaMA model in C/C++ | ![GitHub Badge](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=flat-square) |
| [Infinity](https://github.com/michaelfeil/infinity) | Rest API server for serving text-embeddings | ![GitHub Badge](https://img.shields.io/github/stars/michaelfeil/infinity.svg?style=flat-square) |
| [Modelz-LLM](https://github.com/tensorchord/modelz-llm) | OpenAI compatible API for LLMs and embeddings (LLaMA, Vicuna, ChatGLM and many others) | ![GitHub Badge](https://img.shields.io/github/stars/tensorchord/modelz-llm.svg?style=flat-square) |
| [Ollama](https://github.com/jmorganca/ollama) | Serve Llama 2 and other large language models locally from command line or through a browser interface. | ![GitHub Badge](https://img.shields.io/github/stars/jmorganca/ollama.svg?style=flat-square) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Inference engine for TensorRT on Nvidia GPUs | ![GitHub Badge](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=flat-square) |
| [text-generation-inference](https://github.com/huggingface/text-generation-inference) | Large Language Model Text Generation Inference | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/text-generation-inference.svg?style=flat-square) |
| [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) | Inference for text-embedding models | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/text-embeddings-inference.svg?style=flat-square) |
| [vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs. | ![GitHub stars](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=flat-square) |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Port of OpenAI's Whisper model in C/C++ | ![GitHub Badge](https://img.shields.io/github/stars/ggerganov/whisper.cpp.svg?style=flat-square) |
| [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion) | Real-time inference for Stable Diffusion - 0.88s latency. Covers AITemplate, nvFuser, TensorRT, FlashAttention. | ![GitHub Badge](https://img.shields.io/github/stars/stochasticai/x-stable-diffusion.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Frameworks/Servers for Serving

| Project | Details | Repository |
|---|---|---|
| [BentoML](https://github.com/bentoml/BentoML) | The Unified Model Serving Framework | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=flat-square) |
| [Jina](https://github.com/jina-ai/jina) | Build multimodal AI services via cloud native technologies · Model Serving · Generative AI · Neural Search · Cloud Native | ![GitHub Badge](https://img.shields.io/github/stars/jina-ai/jina.svg?style=flat-square) |
| [Mosec](https://github.com/mosecorg/mosec) | A machine learning model serving framework with dynamic batching and pipelined stages, provides an easy-to-use Python interface. | ![GitHub Badge](https://img.shields.io/github/stars/mosecorg/mosec?style=flat-square) |
| [TFServing](https://github.com/tensorflow/serving) | A flexible, high-performance serving system for machine learning models. | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/serving.svg?style=flat-square) |
| [Torchserve](https://github.com/pytorch/serve) | Serve, optimize and scale PyTorch models in production | ![GitHub Badge](https://img.shields.io/github/stars/pytorch/serve.svg?style=flat-square) |
| [Triton Server (TRTIS)](https://github.com/triton-inference-server/server) | The Triton Inference Server provides an optimized cloud and edge inferencing solution. | ![GitHub Badge](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=flat-square) |
| [langchain-serve](https://github.com/jina-ai/langchain-serve) | Serverless LLM apps on Production with Jina AI Cloud | ![GitHub Badge](https://img.shields.io/github/stars/jina-ai/langchain-serve.svg?style=flat-square) |
| [lanarky](https://github.com/ajndkr/lanarky) | FastAPI framework to build production-grade LLM applications | ![GitHub Badge](https://img.shields.io/github/stars/ajndkr/lanarky.svg?style=flat-square) |
| [ray-llm](https://github.com/ray-project/ray-llm) | LLMs on Ray - RayLLM | ![GitHub Badge](https://img.shields.io/github/stars/ray-project/ray-llm.svg?style=flat-square) |
| [Xinference](https://github.com/xorbitsai/inference) | Replace OpenAI GPT with another LLM in your app by changing a single line of code. Xinference gives you the freedom to use any LLM you need. With Xinference, you're empowered to run inference with any open-source language models, speech recognition models, and multimodal models, whether in the cloud, on-premises, or even on your laptop. | ![GitHub Badge](https://img.shields.io/github/stars/xorbitsai/inference.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Security

### Frameworks for LLM security

| Project | Details | Repository |
|---|---|---|
| [Plexiglass](https://github.com/kortex-labs/plexiglass) | A Python Machine Learning Pentesting Toolbox for Adversarial Attacks. Works with LLMs, DNNs, and other machine learning algorithms. | ![GitHub Badge](https://img.shields.io/github/stars/kortex-labs/plexiglass?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Observability

| Project | Details | Repository |
|---|---|---|
| [Azure OpenAI Logger](https://github.com/aavetis/azure-openai-logger) | "Batteries included" logging solution for your Azure OpenAI instance. | ![GitHub Badge](https://img.shields.io/github/stars/aavetis/azure-openai-logger?style=flat-square) |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | Tests for Continuous Validation of ML Models & Data. Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort. | ![GitHub Badge](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=flat-square) |
| [Evidently](https://github.com/evidentlyai/evidently) | Evaluate and monitor ML models from validation to production. | ![GitHub Badge](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=flat-square) |
| [Fiddler AI](https://github.com/fiddler-labs/fiddler-auditor) | Evaluate, monitor, analyze, and improve machine learning and generative models from pre-production to production. Ship more ML and LLMs into production, and monitor ML and LLM metrics like hallucination, PII, and toxicity. | ![GitHub Badge](https://img.shields.io/github/stars/fiddler-labs/fiddler-auditor.svg?style=flat-square) |
| [Giskard](https://github.com/Giskard-AI/giskard) | Testing framework dedicated to ML models, from tabular to LLMs. Detect risks of biases, performance issues and errors in 4 lines of code. | ![GitHub Badge](https://img.shields.io/github/stars/Giskard-AI/giskard.svg?style=flat-square) |
| [Great Expectations](https://github.com/great-expectations/great_expectations) | Always know what to expect from your data. | ![GitHub Badge](https://img.shields.io/github/stars/great-expectations/great_expectations.svg?style=flat-square) |
| [whylogs](https://github.com/whylabs/whylogs) | The open standard for data logging | ![GitHub Badge](https://img.shields.io/github/stars/whylabs/whylogs.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## LLMOps

| Project | Details | Repository |
|---|---|---|
| [agenta](https://github.com/Agenta-AI/agenta) | The LLMOps platform to build robust LLM apps. Easily experiment and evaluate different prompts, models, and workflows to build robust apps. | ![GitHub Badge](https://img.shields.io/github/stars/Agenta-AI/agenta.svg?style=flat-square) |
| [AI studio](https://github.com/missingstudio/ai) | A Reliable Open Source AI studio to build core infrastructure stack for your LLM Applications. It allows you to gain visibility, make your application reliable, and prepare it for production with features such as caching, rate limiting, exponential retry, model fallback, and more. | ![GitHub Badge](https://img.shields.io/github/stars/missingstudio/ai.svg?style=flat-square) |
| [Arize-Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLMs, vision, language, and tabular models. | ![GitHub Badge](https://img.shields.io/github/stars/Arize-ai/phoenix.svg?style=flat-square) |
| [BudgetML](https://github.com/ebhy/budgetml) | Deploy a ML inference service on a budget in less than 10 lines of code. | ![GitHub Badge](https://img.shields.io/github/stars/ebhy/budgetml.svg?style=flat-square) |
| [CometLLM](https://github.com/comet-ml/comet-llm) | The 100% opensource LLMOps platform to log, manage, and visualize your LLM prompts and chains. Track prompt templates, prompt variables, prompt duration, token usage, and other metadata. Score prompt outputs and visualize chat history all within a single UI. | ![GitHub Badge](https://img.shields.io/github/stars/comet-ml/comet-llm.svg?style=flat-square) |
| [deeplake](https://github.com/activeloopai/deeplake) | Stream large multimodal datasets to achieve near 100% GPU utilization. Query, visualize, & version control data. Access data w/o the need to recompute the embeddings for the model finetuning. | ![GitHub Badge](https://img.shields.io/github/stars/activeloopai/Hub.svg?style=flat-square) |
| [Dify](https://github.com/langgenius/dify) | Open-source framework aims to enable developers (and even non-developers) to quickly build useful applications based on large language models, ensuring they are visual, operable, and improvable. | ![GitHub Badge](https://img.shields.io/github/stars/langgenius/dify.svg?style=flat-square) |
| [Dstack](https://github.com/dstackai/dstack) | Cost-effective LLM development in any cloud (AWS, GCP, Azure, Lambda, etc). | ![GitHub Badge](https://img.shields.io/github/stars/dstackai/dstack.svg?style=flat-square) |
| [Embedchain](https://github.com/embedchain/embedchain) | Framework to create ChatGPT like bots over your dataset. | ![GitHub Badge](https://img.shields.io/github/stars/embedchain/embedchain.svg?style=flat-square) |
| [Fiddler AI](https://www.fiddler.ai/llmops) | Evaluate, monitor, analyze, and improve MLOps and LLMOps from pre-production to production. | |
| [Glide](https://github.com/EinStack/glide) | Cloud-Native LLM Routing Engine. Improve LLM app resilience and speed. | ![GitHub Badge](https://img.shields.io/github/stars/einstack/glide.svg?style=flat-square) |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Creating semantic cache to store responses from LLM queries. | ![GitHub Badge](https://img.shields.io/github/stars/zilliztech/GPTCache.svg?style=flat-square) |
| [Haystack](https://github.com/deepset-ai/haystack) | Quickly compose applications with LLM Agents, semantic search, question-answering and more. | ![GitHub Badge](https://img.shields.io/github/stars/deepset-ai/haystack.svg?style=flat-square) |
| [Izlo](https://getizlo.com/) | Prompt management tools for teams. Store, improve, test, and deploy your prompts in one unified workspace. | |
| [Keywords AI](https://keywordsai.co/) | A unified DevOps platform for AI software. Keywords AI makes it easy for developers to build LLM applications. | |
| [langchain](https://github.com/hwchase17/langchain) | Building applications with LLMs through composability | ![GitHub Badge](https://img.shields.io/github/stars/hwchase17/langchain.svg?style=flat-square) |
| [LangFlow](https://github.com/logspace-ai/langflow) | An effortless way to experiment and prototype LangChain flows with drag-and-drop components and a chat interface. | ![GitHub Badge](https://img.shields.io/github/stars/logspace-ai/langflow.svg?style=flat-square) |
| [Langfuse](https://github.com/langfuse/langfuse) | Open Source LLM Engineering Platform: Traces, evals, prompt management and metrics to debug and improve your LLM application. | ![GitHub Badge](https://img.shields.io/github/stars/langfuse/langfuse.svg?style=flat-square) |
| [LangKit](https://github.com/whylabs/langkit) | Out-of-the-box LLM telemetry collection library that extracts features and profiles prompts, responses and metadata about how your LLM is performing over time to find problems at scale. | ![GitHub Badge](https://img.shields.io/github/stars/whylabs/langkit.svg?style=flat-square) |
| [LiteLLM 🚅](https://github.com/BerriAI/litellm/) | A simple & light 100 line package to **standardize LLM API calls** across OpenAI, Azure, Cohere, Anthropic, Replicate API Endpoints | ![GitHub Badge](https://img.shields.io/github/stars/BerriAI/litellm.svg?style=flat-square) |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Provides a central interface to connect your LLMs with external data. | ![GitHub Badge](https://img.shields.io/github/stars/jerryjliu/llama_index.svg?style=flat-square) |
| [LLMApp](https://github.com/pathwaycom/llm-app) | LLM App is a Python library that helps you build real-time LLM-enabled data pipelines with few lines of code. | ![GitHub Badge](https://img.shields.io/github/stars/pathwaycom/llm-app.svg?style=flat-square) |
| [LLMFlows](https://github.com/stoyan-stoyanov/llmflows) | LLMFlows is a framework for building simple, explicit, and transparent LLM applications such as chatbots, question-answering systems, and agents. | ![GitHub Badge](https://img.shields.io/github/stars/stoyan-stoyanov/llmflows.svg?style=flat-square) |
| [LLMonitor](https://github.com/llmonitor/llmonitor) | Observability and monitoring for AI apps and agents. Debug agents with powerful tracing and logging. Usage analytics and dive deep into the history of your requests. Developer friendly modules with plug-and-play integration into LangChain. | ![GitHub Badge](https://img.shields.io/github/stars/llmonitor/llmonitor.svg?style=flat-square) |
| [magentic](https://github.com/jackmpcollins/magentic) | Seamlessly integrate LLMs as Python functions. Use type annotations to specify structured output. Mix LLM queries and function calling with regular Python code to create complex LLM-powered functionality. | ![GitHub Badge](https://img.shields.io/github/stars/jackmpcollins/magentic.svg?style=flat-square) |
| [Manag.ai](https://www.manag.ai) | Your all-in-one prompt management and observability platform. Craft, track, and perfect your LLM prompts with ease. | |
| [Mirascope](https://github.com/Mirascope/mirascope) | Intuitive convenience tooling for lightning-fast, efficient development and ensuring quality in LLM-based applications | ![GitHub Badge](https://img.shields.io/github/stars/Mirascope/mirascope.svg?style=flat-square) |
| [OpenLIT](https://github.com/openlit/openlit) | OpenLIT is an OpenTelemetry-native GenAI and LLM Application Observability tool and provides OpenTelmetry Auto-instrumentation for monitoring LLMs, VectorDBs and Frameworks. It provides valuable insights into token & cost usage, user interaction, and performance related metrics. | ![GitHub Badge](https://img.shields.io/github/stars/dokulabs/doku.svg?style=flat-square) |
| [Parea AI](https://www.parea.ai/) | Platform and SDK for AI Engineers providing tools for LLM evaluation, observability, and a version-controlled enhanced prompt playground. | ![GitHub Badge](https://img.shields.io/github/stars/parea-ai/parea-sdk-py?style=flat-square) |
| [Pezzo 🕹️](https://github.com/pezzolabs/pezzo) | Pezzo is the open-source LLMOps platform built for developers and teams. In just two lines of code, you can seamlessly troubleshoot your AI operations, collaborate and manage your prompts in one place, and instantly deploy changes to any environment. | ![GitHub Badge](https://img.shields.io/github/stars/pezzolabs/pezzo.svg?style=flat-square) |
| [PromptHub](https://www.prompthub.us) | Full stack prompt management tool designed to be usable by technical and non-technical team members. Test, version, collaborate, deploy, and monitor, all from one place. | |
| [promptfoo](https://github.com/typpo/promptfoo) | Open-source tool for testing & evaluating prompt quality. Create test cases, automatically check output quality and catch regressions, and reduce evaluation cost. | ![GitHub Badge](https://img.shields.io/github/stars/typpo/promptfoo.svg?style=flat-square) |
| [Prompteams](https://www.prompteams.com) | Prompt management system. Version, test, collaborate, and retrieve prompts through real-time APIs. Have GitHub style with repos, branches, and commits (and commit history). | |
| [prompttools](https://github.com/hegelai/prompttools) | Open-source tools for testing and experimenting with prompts. The core idea is to enable developers to evaluate prompts using familiar interfaces like code and notebooks. In just a few lines of codes, you can test your prompts and parameters across different models (whether you are using OpenAI, Anthropic, or LLaMA models). You can even evaluate the retrieval accuracy of vector databases. | ![GitHub Badge](https://img.shields.io/github/stars/hegelai/prompttools.svg?style=flat-square) |
| [TreeScale](https://treescale.com) | All In One Dev Platform For LLM Apps. Deploy LLM-enhanced APIs seamlessly using tools for prompt optimization, semantic querying, version management, statistical evaluation, and performance tracking. As a part of the developer friendly API implementation TreeScale offers Elastic LLM product, which makes a unified API Endpoint for all major LLM providers and open source models. | |
| [TrueFoundry](https://www.truefoundry.com/) | Deploy LLMOps tools like Vector DBs, Embedding server etc on your own Kubernetes (EKS,AKS,GKE,On-prem) Infra including deploying, Fine-tuning, tracking Prompts and serving Open Source LLM Models with full Data Security and Optimal GPU Management. Train and Launch your LLM Application at Production scale with best Software Engineering practices. | |
| [ReliableGPT 💪](https://github.com/BerriAI/reliableGPT/) | Handle OpenAI Errors (overloaded OpenAI servers, rotated keys, or context window errors) for your production LLM Applications. | ![GitHub Badge](https://img.shields.io/github/stars/BerriAI/reliableGPT.svg?style=flat-square) |
| [Portkey](https://portkey.ai/) | Control Panel with an observability suite & an AI gateway — to ship fast, reliable, and cost-efficient apps. | |
| [Vellum](https://www.vellum.ai/) | An AI product development platform to experiment with, evaluate, and deploy advanced LLM apps. | |
| [Weights & Biases (Prompts)](https://docs.wandb.ai/guides/prompts) | A suite of LLMOps tools within the developer-first W&B MLOps platform. Utilize W&B Prompts for visualizing and inspecting LLM execution flow, tracking inputs and outputs, viewing intermediate results, securely managing prompts and LLM chain configurations. | |
| [Wordware](https://www.wordware.ai) | A web-hosted IDE where non-technical domain experts work with AI Engineers to build task-specific AI agents. It approaches prompting as a new programming language rather than low/no-code blocks. | |
| [xTuring](https://github.com/stochasticai/xturing) | Build and control your personal LLMs with fast and efficient fine-tuning. | ![GitHub Badge](https://img.shields.io/github/stars/stochasticai/xturing.svg?style=flat-square) |
| [ZenML](https://github.com/zenml-io/zenml) | Open-source framework for orchestrating, experimenting and deploying production-grade ML solutions, with built-in `langchain` & `llama_index` integrations. | ![GitHub Badge](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Search

### Vector search

| Project | Details | Repository |
|---|---|---|
| [AquilaDB](https://github.com/Aquila-Network/AquilaDB) | An easy to use Neural Search Engine. Index latent vectors along with JSON metadata and do efficient k-NN search. | ![GitHub Badge](https://img.shields.io/github/stars/Aquila-Network/AquilaDB.svg?style=flat-square) |
| [Awadb](https://github.com/awa-ai/awadb) | AI Native database for embedding vectors | ![GitHub Badge](https://img.shields.io/github/stars/awa-ai/awadb.svg?style=flat-square) |
| [Chroma](https://github.com/chroma-core/chroma) | the open source embedding database | ![GitHub Badge](https://img.shields.io/github/stars/chroma-core/chroma.svg?style=flat-square) |
| [Infinity](https://github.com/infiniflow/infinity) | The AI-native database built for LLM applications, providing incredibly fast vector and full-text search | ![GitHub Badge](https://img.shields.io/github/stars/infiniflow/infinity.svg?style=flat-square) |
| [Lancedb](https://github.com/lancedb/lancedb) | Developer-friendly, serverless vector database for AI applications. Easily add long-term memory to your LLM apps! | ![GitHub Badge](https://img.shields.io/github/stars/lancedb/lancedb.svg?style=flat-square) |
| [Marqo](https://github.com/marqo-ai/marqo) | Tensor search for humans. | ![GitHub Badge](https://img.shields.io/github/stars/marqo-ai/marqo.svg?style=flat-square) |
| [Milvus](https://github.com/milvus-io/milvus) | Vector database for scalable similarity search and AI applications. | ![GitHub Badge](https://img.shields.io/github/stars/milvus-io/milvus.svg?style=flat-square) |
| [Pinecone](https://www.pinecone.io/) | The Pinecone vector database makes it easy to build high-performance vector search applications. Developer-friendly, fully managed, and easily scalable without infrastructure hassles. | |
| [pgvector](https://github.com/pgvector/pgvector) | Open-source vector similarity search for Postgres. | ![GitHub Badge](https://img.shields.io/github/stars/pgvector/pgvector.svg?style=flat-square) |
| [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) | Vector database plugin for Postgres, written in Rust, specifically designed for LLM. | ![GitHub Badge](https://img.shields.io/github/stars/tensorchord/pgvecto.rs.svg?style=flat-square) |
| [Qdrant](https://github.com/qdrant/qdrant) | Vector Search Engine and Database for the next generation of AI applications. Also available in the cloud | ![GitHub Badge](https://img.shields.io/github/stars/qdrant/qdrant.svg?style=flat-square) |
| [txtai](https://github.com/neuml/txtai) | Build AI-powered semantic search applications | ![GitHub Badge](https://img.shields.io/github/stars/neuml/txtai.svg?style=flat-square) |
| [Vald](https://github.com/vdaas/vald) | A Highly Scalable Distributed Vector Search Engine | ![GitHub Badge](https://img.shields.io/github/stars/vdaas/vald.svg?style=flat-square) |
| [Vearch](https://github.com/vearch/vearch) | A distributed system for embedding-based vector retrieval | ![GitHub Badge](https://img.shields.io/github/stars/vearch/vearch.svg?style=flat-square) |
| [VectorDB](https://github.com/jina-ai/vectordb) | A Python vector database you just need - no more, no less. | ![GitHub Badge](https://img.shields.io/github/stars/jina-ai/vectordb.svg?style=flat-square) |
| [Vellum](https://www.vellum.ai/products/retrieval) | A managed service for ingesting documents and performing hybrid semantic/keyword search across them. Comes with out-of-box support for OCR, text chunking, embedding model experimentation, metadata filtering, and production-grade APIs. | |
| [Weaviate](https://github.com/semi-technologies/weaviate) | Weaviate is an open source vector search engine that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients. | ![GitHub Badge](https://img.shields.io/github/stars/semi-technologies/weaviate.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Code AI

| Project | Details | Repository |
|---|---|---|
| [CodeGeeX](https://github.com/THUDM/CodeGeeX) | CodeGeeX: An Open Multilingual Code Generation Model (KDD 2023) | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/CodeGeeX.svg?style=flat-square) |
| [CodeGen](https://github.com/salesforce/CodeGen) | CodeGen is an open-source model for program synthesis. Trained on TPU-v4. Competitive with OpenAI Codex. | ![GitHub Badge](https://img.shields.io/github/stars/salesforce/CodeGen.svg?style=flat-square) |
| [CodeT5](https://github.com/salesforce/CodeT5) | Open Code LLMs for Code Understanding and Generation. | ![GitHub Badge](https://img.shields.io/github/stars/salesforce/CodeT5.svg?style=flat-square) |
| [Continue](https://github.com/continuedev/continue) | ⏩ the open-source autopilot for software development—bring the power of ChatGPT to VS Code | ![GitHub Badge](https://img.shields.io/github/stars/continuedev/continue.svg?style=flat-square) |
| [fauxpilot](https://github.com/fauxpilot/fauxpilot) | An open-source alternative to GitHub Copilot server | ![GitHub Badge](https://img.shields.io/github/stars/fauxpilot/fauxpilot.svg?style=flat-square) |
| [tabby](https://github.com/TabbyML/tabby) | Self-hosted AI coding assistant. An opensource / on-prem alternative to GitHub Copilot. | ![GitHub Badge](https://img.shields.io/github/stars/TabbyML/tabby.svg?style=flat-square) |

## Training

### IDEs and Workspaces

| Project | Details | Repository |
|---|---|---|
| [code server](https://github.com/coder/code-server) | Run VS Code on any machine anywhere and access it in the browser. | ![GitHub Badge](https://img.shields.io/github/stars/coder/code-server.svg?style=flat-square) |
| [conda](https://github.com/conda/conda) | OS-agnostic, system-level binary package manager and ecosystem. | ![GitHub Badge](https://img.shields.io/github/stars/conda/conda.svg?style=flat-square) |
| [Docker](https://github.com/moby/moby) | Moby is an open-source project created by Docker to enable and accelerate software containerization. | ![GitHub Badge](https://img.shields.io/github/stars/moby/moby.svg?style=flat-square) |
| [envd](https://github.com/tensorchord/envd) | 🏕️ Reproducible development environment for AI/ML. | ![GitHub Badge](https://img.shields.io/github/stars/tensorchord/envd.svg?style=flat-square) |
| [Jupyter Notebooks](https://github.com/jupyter/notebook) | The Jupyter notebook is a web-based notebook environment for interactive computing. | ![GitHub Badge](https://img.shields.io/github/stars/jupyter/notebook.svg?style=flat-square) |
| [Kurtosis](https://github.com/kurtosis-tech/kurtosis) | A build, packaging, and run system for ephemeral multi-container environments. | ![GitHub Badge](https://img.shields.io/github/stars/kurtosis-tech/kurtosis.svg?style=flat-square) |
| [Wordware](https://www.wordware.ai) | A web-hosted IDE where non-technical domain experts work with AI Engineers to build task-specific AI agents. It approaches prompting as a new programming language rather than low/no-code blocks. | |

**[⬆ back to ToC](#table-of-contents)**

### Foundation Model Fine Tuning

| Project                                                                    | Details                                                                                                                                                                                          | Repository |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| [alpaca-lora](https://github.com/tloen/alpaca-lora)                        | Instruct-tune LLaMA on consumer hardware                                                                                                                                                         | ![GitHub Badge](https://img.shields.io/github/stars/tloen/alpaca-lora.svg?style=flat-square) |
| [finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler) | A PyTorch Lightning extension that accelerates and enhances foundation model experimentation with flexible fine-tuning schedules.                                                                | ![GitHub Badge](https://img.shields.io/github/stars/speediedan/finetuning-scheduler.svg?style=flat-square) |
| [Flyflow](https://github.com/flyflow-devs)                                 | Open source, high performance fine tuning as a service for GPT4 quality models with 5x lower latency and 3x lower cost  | ![GitHub Badge](https://img.shields.io/github/stars/flyflow-devs/flyflow.svg?style=flat-square) |
| [LMFlow](https://github.com/OptimalScale/LMFlow)                           | An Extensible Toolkit for Finetuning and Inference of Large Foundation Models                                                                                                                    | ![GitHub Badge](https://img.shields.io/github/stars/OptimalScale/LMFlow.svg?style=flat-square) |
| [Lora](https://github.com/cloneofsimo/lora)                                | Using Low-rank adaptation to quickly fine-tune diffusion models.                                                                                                                                 | ![GitHub Badge](https://img.shields.io/github/stars/cloneofsimo/lora.svg?style=flat-square) |
| [peft](https://github.com/huggingface/peft)                                | State-of-the-art Parameter-Efficient Fine-Tuning.                                                                                                                                                | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/peft.svg?style=flat-square) |
| [p-tuning-v2](https://github.com/THUDM/P-tuning-v2)                        | An optimized prompt tuning strategy achieving comparable performance to fine-tuning on small/medium-sized models and sequence tagging challenges. [(ACL 2022)](https://arxiv.org/abs/2110.07602) | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/P-tuning-v2.svg?style=flat-square) |
| [QLoRA](https://github.com/artidoro/qlora)                                 | Efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.                  | ![GitHub Badge](https://img.shields.io/github/stars/artidoro/qlora.svg?style=flat-square) |
| [TRL](https://github.com/huggingface/trl)                                  | Train transformer language models with reinforcement learning.                                                                                                                                   | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/trl.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Frameworks for Training

| Project | Details | Repository |
|---|---|---|
| [Accelerate](https://github.com/huggingface/accelerate) | 🚀 A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision. | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/accelerate.svg?style=flat-square) |
| [Apache MXNet](https://github.com/apache/mxnet) | Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler. | ![GitHub Badge](https://img.shields.io/github/stars/apache/mxnet.svg?style=flat-square) |
| [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | A tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures. | ![GitHub Badge](https://img.shields.io/github/stars/OpenAccess-AI-Collective/axolotl.svg?style=flat-square) |
| [Caffe](https://github.com/BVLC/caffe) | A fast open framework for deep learning. | ![GitHub Badge](https://img.shields.io/github/stars/BVLC/caffe.svg?style=flat-square) |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | An integrated large-scale model training system with efficient parallelization techniques. | ![GitHub Badge](https://img.shields.io/github/stars/hpcaitech/ColossalAI.svg?style=flat-square) |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective. | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=flat-square) |
| [Horovod](https://github.com/horovod/horovod) | Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. | ![GitHub Badge](https://img.shields.io/github/stars/horovod/horovod.svg?style=flat-square) |
| [Jax](https://github.com/google/jax) | Autograd and XLA for high-performance machine learning research. | ![GitHub Badge](https://img.shields.io/github/stars/google/jax.svg?style=flat-square) |
| [Kedro](https://github.com/kedro-org/kedro) | Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code. | ![GitHub Badge](https://img.shields.io/github/stars/kedro-org/kedro.svg?style=flat-square) |
| [Keras](https://github.com/keras-team/keras) | Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. | ![GitHub Badge](https://img.shields.io/github/stars/keras-team/keras.svg?style=flat-square) |
| [LightGBM](https://github.com/microsoft/LightGBM) | A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks. | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/LightGBM.svg?style=flat-square) |
| [MegEngine](https://github.com/MegEngine/MegEngine) | MegEngine is a fast, scalable and easy-to-use deep learning framework, with auto-differentiation. | ![GitHub Badge](https://img.shields.io/github/stars/MegEngine/MegEngine.svg?style=flat-square) |
| [metric-learn](https://github.com/scikit-learn-contrib/metric-learn) | Metric Learning Algorithms in Python. | ![GitHub Badge](https://img.shields.io/github/stars/scikit-learn-contrib/metric-learn.svg?style=flat-square) |
| [MindSpore](https://github.com/mindspore-ai/mindspore) | MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. | ![GitHub Badge](https://img.shields.io/github/stars/mindspore-ai/mindspore.svg?style=flat-square) |
| [Oneflow](https://github.com/Oneflow-Inc/oneflow) | OneFlow is a performance-centered and open-source deep learning framework. | ![GitHub Badge](https://img.shields.io/github/stars/Oneflow-Inc/oneflow.svg?style=flat-square) |
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | Machine Learning Framework from Industrial Practice. | ![GitHub Badge](https://img.shields.io/github/stars/PaddlePaddle/Paddle.svg?style=flat-square) |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration. | ![GitHub Badge](https://img.shields.io/github/stars/pytorch/pytorch.svg?style=flat-square) |
| [PyTorch Lightning](https://github.com/lightning-AI/lightning) | Deep learning framework to train, deploy, and ship AI products Lightning fast. | ![GitHub Badge](https://img.shields.io/github/stars/lightning-AI/lightning.svg?style=flat-square) |
| [XGBoost](https://github.com/dmlc/xgboost) | Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library. | ![GitHub Badge](https://img.shields.io/github/stars/dmlc/xgboost.svg?style=flat-square) |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Machine Learning in Python. | ![GitHub Badge](https://img.shields.io/github/stars/scikit-learn/scikit-learn.svg?style=flat-square) |
| [TensorFlow](https://github.com/tensorflow/tensorflow) | An Open Source Machine Learning Framework for Everyone. | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/tensorflow.svg?style=flat-square) |
| [VectorFlow](https://github.com/Netflix/vectorflow) | A minimalist neural network library optimized for sparse data and single machine environments. | ![GitHub Badge](https://img.shields.io/github/stars/Netflix/vectorflow.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Experiment Tracking

| Project | Details | Repository |
|---|---|---|
| [Aim](https://github.com/aimhubio/aim) | an easy-to-use and performant open-source experiment tracker. | ![GitHub Badge](https://img.shields.io/github/stars/aimhubio/aim.svg?style=flat-square) |
| [ClearML](https://github.com/allegroai/clearml) | Auto-Magical CI/CD to streamline your ML workflow. Experiment Manager, MLOps and Data-Management | ![GitHub Badge](https://img.shields.io/github/stars/allegroai/clearml.svg?style=flat-square) |
| [Comet](https://github.com/comet-ml/comet-examples) | Comet is an MLOps platform that offers experiment tracking, model production management, a model registry, and full data lineage from training straight through to production. Comet plays nicely with all your favorite tools, so you don't have to change your existing workflow. Check out CometLLM for all your prompt engineering needs! | ![GitHub Badge](https://img.shields.io/github/stars/comet-ml/comet-examples.svg?style=flat-square) |
| [Guild AI](https://github.com/guildai/guildai) | Experiment tracking, ML developer tools. | ![GitHub Badge](https://img.shields.io/github/stars/guildai/guildai.svg?style=flat-square) |
| [MLRun](https://github.com/mlrun/mlrun) | Machine Learning automation and tracking. | ![GitHub Badge](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=flat-square) |
| [Kedro-Viz](https://github.com/kedro-org/kedro-viz) | Kedro-Viz is an interactive development tool for building data science pipelines with Kedro. Kedro-Viz also allows users to view and compare different runs in the Kedro project. | ![GitHub Badge](https://img.shields.io/github/stars/kedro-org/kedro-viz.svg?style=flat-square) |
| [LabNotebook](https://github.com/henripal/labnotebook) | LabNotebook is a tool that allows you to flexibly monitor, record, save, and query all your machine learning experiments. | ![GitHub Badge](https://img.shields.io/github/stars/henripal/labnotebook.svg?style=flat-square) |
| [Sacred](https://github.com/IDSIA/sacred) | Sacred is a tool to help you configure, organize, log and reproduce experiments. | ![GitHub Badge](https://img.shields.io/github/stars/IDSIA/sacred.svg?style=flat-square) |
| [Weights & Biases](https://github.com/wandb/wandb) | A developer first, lightweight, user-friendly experiment tracking and visualization tool for machine learning projects, streamlining collaboration and simplifying MLOps. W&B excels at tracking LLM-powered applications, featuring W&B Prompts for LLM execution flow visualization, input and output monitoring, and secure management of prompts and LLM chain configurations. | ![GitHub Badge](https://img.shields.io/github/stars/wandb/wandb.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Visualization

| Project | Details | Repository |
|---|---|---|
| [Fiddler AI](https://github.com/fiddler-labs) | Rich dashboards, reports, and UMAP to perform root cause analysis, pinpoint problem areas, like correctness, safety, and privacy issues, and improve LLM outcomes. | |
| [Maniford](https://github.com/uber/manifold) | A model-agnostic visual debugging tool for machine learning. | ![GitHub Badge](https://img.shields.io/github/stars/uber/manifold.svg?style=flat-square) |
| [netron](https://github.com/lutzroeder/netron) | Visualizer for neural network, deep learning, and machine learning models. | ![GitHub Badge](https://img.shields.io/github/stars/lutzroeder/netron.svg?style=flat-square) |
| [OpenOps](https://github.com/ThePlugJumbo/openops) | Bring multiple data streams into one dashboard. | ![GitHub Badge](https://img.shields.io/github/stars/theplugjumbo/openops.svg?style=flat-square) |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | TensorFlow's Visualization Toolkit. | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/tensorboard.svg?style=flat-square) |
| [TensorSpace](https://github.com/tensorspace-team/tensorspace) | Neural network 3D visualization framework, build interactive and intuitive model in browsers, support pre-trained deep learning models from TensorFlow, Keras, TensorFlow.js. | ![GitHub Badge](https://img.shields.io/github/stars/tensorspace-team/tensorspace.svg?style=flat-square) |
| [dtreeviz](https://github.com/parrt/dtreeviz) | A python library for decision tree visualization and model interpretation. | ![GitHub Badge](https://img.shields.io/github/stars/parrt/dtreeviz.svg?style=flat-square) |
| [Zetane Viewer](https://github.com/zetane/viewer) | ML models and internal tensors 3D visualizer. | ![GitHub Badge](https://img.shields.io/github/stars/zetane/viewer.svg?style=flat-square) |
| [Zeno](https://github.com/zeno-ml/zeno) | AI evaluation platform for interactively exploring data and model outputs. | ![GitHub Badge](https://img.shields.io/github/stars/zeno-ml/zeno.svg?style=flat-square) |

### Model Editing

| Project | Details | Repository |
|---|---|---|
| [FastEdit](https://github.com/hiyouga/FastEdit) | FastEdit aims to assist developers with injecting fresh and customized knowledge into large language models efficiently using one single command. | ![GitHub Badge](https://img.shields.io/github/stars/hiyouga/FastEdit.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Data

### Data Management

| Project | Details | Repository |
|---|---|---|
| [ArtiVC](https://github.com/InfuseAI/ArtiVC) | A version control system to manage large files. Lake is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size. | ![GitHub Badge](https://img.shields.io/github/stars/InfuseAI/ArtiVC.svg?style=flat-square) |
| [Dolt](https://github.com/dolthub/dolt) | Git for Data. | ![GitHub Badge](https://img.shields.io/github/stars/dolthub/dolt.svg?style=flat-square) |
| [DVC](https://github.com/iterative/dvc) | Data Version Control - Git for Data & Models - ML Experiments Management. | ![GitHub Badge](https://img.shields.io/github/stars/iterative/dvc.svg?style=flat-square) |
| [Delta-Lake](https://github.com/delta-io/delta) | Storage layer that brings scalable, ACID transactions to Apache Spark and other engines. | ![GitHub Badge](https://img.shields.io/github/stars/delta-io/delta.svg?style=flat-square) |
| [Pachyderm](https://github.com/pachyderm/pachyderm) | Pachyderm is a version control system for data. | ![GitHub Badge](https://img.shields.io/github/stars/pachyderm/pachyderm.svg?style=flat-square) |
| [Quilt](https://github.com/quiltdata/quilt) | A self-organizing data hub for S3. | ![GitHub Badge](https://img.shields.io/github/stars/quiltdata/quilt.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Data Storage

| Project | Details | Repository |
|---|---|---|
| [JuiceFS](https://github.com/juicedata/juicefs) | A distributed POSIX file system built on top of Redis and S3. | ![GitHub Badge](https://img.shields.io/github/stars/juicedata/juicefs.svg?style=flat-square) |
| [LakeFS](https://github.com/treeverse/lakeFS) | Git-like capabilities for your object storage. | ![GitHub Badge](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=flat-square) |
| [Lance](https://github.com/eto-ai/lance) | Modern columnar data format for ML implemented in Rust. | ![GitHub Badge](https://img.shields.io/github/stars/eto-ai/lance.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Data Tracking

| Project | Details | Repository |
|---|---|---|
| [Piperider](https://github.com/InfuseAI/piperider) | A CLI tool that allows you to build data profiles and write assertion tests for easily evaluating and tracking your data's reliability over time. | ![GitHub Badge](https://img.shields.io/github/stars/InfuseAI/piperider.svg?style=flat-square) |
| [LUX](https://github.com/lux-org/lux) | A Python library that facilitates fast and easy data exploration by automating the visualization and data analysis process. | ![GitHub Badge](https://img.shields.io/github/stars/lux-org/lux.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Feature Engineering

| Project | Details | Repository |
|---|---|---|
| [Featureform](https://github.com/featureform/featureform) | The Virtual Feature Store. Turn your existing data infrastructure into a feature store. | ![GitHub Badge](https://img.shields.io/github/stars/featureform/featureform.svg?style=flat-square) |
| [FeatureTools](https://github.com/Featuretools/featuretools) | An open source python framework for automated feature engineering | ![GitHub Badge](https://img.shields.io/github/stars/Featuretools/featuretools.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Data/Feature enrichment

| Project | Details | Repository |
|---|---|---|
| [Upgini](https://github.com/upgini/upgini) | Free automated data & feature enrichment library for machine learning: automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features | ![GitHub Badge](https://img.shields.io/github/stars/upgini/upgini.svg?style=flat-square) |
| [Feast](https://github.com/feast-dev/feast) | An open source feature store for machine learning. | ![GitHub Badge](https://img.shields.io/github/stars/feast-dev/feast.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Large Scale Deployment

### ML Platforms

| Project | Details | Repository |
|---|---|---|
| [Comet](https://github.com/comet-ml/comet-examples) | Comet is an MLOps platform that offers experiment tracking, model production management, a model registry, and full data lineage from training straight through to production. Comet plays nicely with all your favorite tools, so you don't have to change your existing workflow. Check out CometLLM for all your prompt engineering needs! | ![GitHub Badge](https://img.shields.io/github/stars/comet-ml/comet-examples.svg?style=flat-square) |
| [ClearML](https://github.com/allegroai/clearml) | Auto-Magical CI/CD to streamline your ML workflow. Experiment Manager, MLOps and Data-Management. | ![GitHub Badge](https://img.shields.io/github/stars/allegroai/clearml.svg?style=flat-square) |
| [Hopsworks](https://github.com/logicalclocks/hopsworks) | Hopsworks is a MLOps platform for training and operating large and small ML systems, including fine-tuning and serving LLMs. Hopsworks includes both a feature store and vector database for RAG. | ![GitHub Badge](https://img.shields.io/github/stars/logicalclocks/hopsworks.svg?style=flat-square) |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | An open platform for operating large language models (LLMs) in production. Fine-tune, serve, deploy, and monitor any LLMs with ease. | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/OpenLLM.svg?style=flat-square) |
| [MLflow](https://github.com/mlflow/mlflow) | Open source platform for the machine learning lifecycle. | ![GitHub Badge](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=flat-square) |
| [MLRun](https://github.com/mlrun/mlrun) | An open MLOps platform for quickly building and managing continuous ML applications across their lifecycle. | ![GitHub Badge](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=flat-square) |
| [ModelFox](https://github.com/modelfoxdotdev/modelfox) | ModelFox is a platform for managing and deploying machine learning models. | ![GitHub Badge](https://img.shields.io/github/stars/modelfoxdotdev/modelfox.svg?style=flat-square) |
| [Kserve](https://github.com/kserve/kserve) | Standardized Serverless ML Inference Platform on Kubernetes | ![GitHub Badge](https://img.shields.io/github/stars/kserve/kserve.svg?style=flat-square) |
| [Kubeflow](https://github.com/kubeflow/kubeflow) | Machine Learning Toolkit for Kubernetes. | ![GitHub Badge](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=flat-square) |
| [PAI](https://github.com/microsoft/pai) | Resource scheduling and cluster management for AI. | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/pai.svg?style=flat-square) |
| [Polyaxon](https://github.com/polyaxon/polyaxon) | Machine Learning Management & Orchestration Platform. | ![GitHub Badge](https://img.shields.io/github/stars/polyaxon/polyaxon.svg?style=flat-square) |
| [Primehub](https://github.com/InfuseAI/primehub) | An effortless infrastructure for machine learning built on the top of Kubernetes. | ![GitHub Badge](https://img.shields.io/github/stars/InfuseAI/primehub.svg?style=flat-square) |
| [OpenModelZ](https://github.com/tensorchord/openmodelz) | One-click machine learning deployment (LLM, text-to-image and so on) at scale on any cluster (GCP, AWS, Lambda labs, your home lab, or even a single machine). | ![GitHub Badge](https://img.shields.io/github/stars/tensorchord/openmodelz.svg?style=flat-square) |
| [Seldon-core](https://github.com/SeldonIO/seldon-core) | An MLOps framework to package, deploy, monitor and manage thousands of production machine learning models | ![GitHub Badge](https://img.shields.io/github/stars/SeldonIO/seldon-core.svg?style=flat-square) |
| [Starwhale](https://github.com/star-whale/starwhale) | An MLOps/LLMOps platform for model building, evaluation, and fine-tuning. | ![GitHub Badge](https://img.shields.io/github/stars/star-whale/starwhale.svg?style=flat-square) |
| [TrueFoundry](https://truefoundry.com/llmops) | A PaaS to deploy, Fine-tune and serve LLM Models on a company’s own Infrastructure with Data Security and Optimal GPU and Cost Management. Launch your LLM Application at Production scale with best DevSecOps practices. | |
| [Weights & Biases](https://github.com/wandb/wandb) | A lightweight and flexible platform for machine learning experiment tracking, dataset versioning, and model management, enhancing collaboration and streamlining MLOps workflows. W&B excels at tracking LLM-powered applications, featuring W&B Prompts for LLM execution flow visualization, input and output monitoring, and secure management of prompts and LLM chain configurations. | ![GitHub Badge](https://img.shields.io/github/stars/wandb/wandb.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Workflow

| Project | Details | Repository |
|---|---|---|
| [Airflow](https://airflow.apache.org/) | A platform to programmatically author, schedule and monitor workflows. | ![GitHub Badge](https://img.shields.io/github/stars/apache/airflow?style=flat-square) |
| [aqueduct](https://github.com/aqueducthq/aqueduct) | An Open-Source Platform for Production Data Science | ![GitHub Badge](https://img.shields.io/github/stars/aqueducthq/aqueduct.svg?style=flat-square) |
| [Argo Workflows](https://github.com/argoproj/argo-workflows) | Workflow engine for Kubernetes. | ![GitHub Badge](https://img.shields.io/github/stars/argoproj/argo-workflows.svg?style=flat-square) |
| [Flyte](https://github.com/flyteorg/flyte) | Kubernetes-native workflow automation platform for complex, mission-critical data and ML processes at scale. | ![GitHub Badge](https://img.shields.io/github/stars/flyteorg/flyte.svg?style=flat-square) |
| [Hamilton](https://github.com/dagworks-inc/hamilton) | A lightweight framework to represent ML/language model pipelines as a series of python functions. | ![GitHub Badge](https://img.shields.io/github/stars/dagworks-inc/hamilton.svg?style=flat-square) |
| [Kubeflow Pipelines](https://github.com/kubeflow/pipelines) | Machine Learning Pipelines for Kubeflow. | ![GitHub Badge](https://img.shields.io/github/stars/kubeflow/pipelines.svg?style=flat-square) |
| [LangFlow](https://github.com/logspace-ai/langflow) | An effortless way to experiment and prototype LangChain flows with drag-and-drop components and a chat interface. | ![GitHub Badge](https://img.shields.io/github/stars/logspace-ai/langflow.svg?style=flat-square) |
| [Metaflow](https://github.com/Netflix/metaflow) | Build and manage real-life data science projects with ease! | ![GitHub Badge](https://img.shields.io/github/stars/Netflix/metaflow.svg?style=flat-square) |
| [Ploomber](https://github.com/ploomber/ploomber) | The fastest way to build data pipelines. Develop iteratively, deploy anywhere. | ![GitHub Badge](https://img.shields.io/github/stars/ploomber/ploomber.svg?style=flat-square) |
| [Prefect](https://github.com/PrefectHQ/prefect) | The easiest way to automate your data. | ![GitHub Badge](https://img.shields.io/github/stars/PrefectHQ/prefect.svg?style=flat-square) |
| [VDP](https://github.com/instill-ai/vdp) | An open-source unstructured data ETL tool to streamline the end-to-end unstructured data processing pipeline. | ![GitHub Badge](https://img.shields.io/github/stars/instill-ai/vdp.svg?style=flat-square) |
| [ZenML](https://github.com/zenml-io/zenml) | MLOps framework to create reproducible pipelines. | ![GitHub Badge](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Scheduling

| Project | Details | Repository |
|---|---|---|
| [Kueue](https://github.com/kubernetes-sigs/kueue) | Kubernetes-native Job Queueing. | ![GitHub Badge](https://img.shields.io/github/stars/kubernetes-sigs/kueue.svg?style=flat-square) |
| [PAI](https://github.com/microsoft/pai) | Resource scheduling and cluster management for AI (Open-sourced by Microsoft). | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/pai.svg?style=flat-square) |
| [Slurm](https://github.com/SchedMD/slurm) | A Highly Scalable Workload Manager. | ![GitHub Badge](https://img.shields.io/github/stars/SchedMD/slurm.svg?style=flat-square) |
| [Volcano](https://github.com/volcano-sh/volcano) | A Cloud Native Batch System (Project under CNCF). | ![GitHub Badge](https://img.shields.io/github/stars/volcano-sh/volcano.svg?style=flat-square) |
| [Yunikorn](https://github.com/apache/yunikorn-core) | Light-weight, universal resource scheduler for container orchestrator systems. | ![GitHub Badge](https://img.shields.io/github/stars/apache/yunikorn-core.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Model Management

| Project | Details | Repository |
|---|---|---|
| [Comet](https://github.com/comet-ml/comet-examples) | Comet is an MLOps platform that offers Model Production Management, a Model Registry, and full model lineage from training straight through to production. Use Comet for model reproducibility, model debugging, model versioning, model visibility, model auditing, model governance, and model monitoring. | ![GitHub Badge](https://img.shields.io/github/stars/comet-ml/comet-examples.svg?style=flat-square) |
| [dvc](https://github.com/iterative/dvc) | ML Experiments Management - Data Version Control - Git for Data & Models | ![GitHub Badge](https://img.shields.io/github/stars/iterative/dvc.svg?style=flat-square) |
| [ModelDB](https://github.com/VertaAI/modeldb) | Open Source ML Model Versioning, Metadata, and Experiment Management | ![GitHub Badge](https://img.shields.io/github/stars/VertaAI/modeldb.svg?style=flat-square) |
| [MLEM](https://github.com/iterative/mlem) | A tool to package, serve, and deploy any ML model on any platform. | ![GitHub Badge](https://img.shields.io/github/stars/iterative/mlem.svg?style=flat-square) |
| [ormb](https://github.com/kleveross/ormb) | Docker for Your ML/DL Models Based on OCI Artifacts | ![GitHub Badge](https://img.shields.io/github/stars/kleveross/ormb.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Performance

### ML Compiler

| Project | Details | Repository |
|---|---|---|
| [ONNX-MLIR](https://github.com/onnx/onnx-mlir) | Compiler technology to transform a valid Open Neural Network Exchange (ONNX) graph into code that implements the graph with minimum runtime support. | ![GitHub Badge](https://img.shields.io/github/stars/onnx/onnx-mlir.svg?style=flat-square) |
| [TVM](https://github.com/apache/tvm) | Open deep learning compiler stack for cpu, gpu and specialized accelerators | ![GitHub Badge](https://img.shields.io/github/stars/apache/tvm.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

### Profiling

| Project | Details | Repository |
|---|---|---|
| [octoml-profile](https://github.com/octoml/octoml-profile) | octoml-profile is a python library and cloud service designed to provide the simplest experience for assessing and optimizing the performance of PyTorch models on cloud hardware with state-of-the-art ML acceleration technology. | ![GitHub Badge](https://img.shields.io/github/stars/octoml/octoml-profile.svg?style=flat-square) |
| [scalene](https://github.com/plasma-umass/scalene) | a high-performance, high-precision CPU, GPU, and memory profiler for Python | ![GitHub Badge](https://img.shields.io/github/stars/plasma-umass/scalene.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## AutoML

| Project | Details | Repository |
|---|---|---|
| [Archai](https://github.com/microsoft/archai) | a platform for Neural Network Search (NAS) that allows you to generate efficient deep networks for your applications. | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/archai.svg?style=flat-square) |
| [autoai](https://github.com/blobcity/autoai) | A framework to find the best performing AI/ML model for any AI problem. | ![GitHub Badge](https://img.shields.io/github/stars/blobcity/autoai.svg?style=flat-square) |
| [AutoGL](https://github.com/THUMNLab/AutoGL) | An autoML framework & toolkit for machine learning on graphs | ![GitHub Badge](https://img.shields.io/github/stars/THUMNLab/AutoGL.svg?style=flat-square) |
| [AutoGluon](https://github.com/awslabs/autogluon) | AutoML for Image, Text, and Tabular Data. | ![GitHub Badge](https://img.shields.io/github/stars/awslabs/autogluon.svg?style=flat-square) |
| [automl-gs](https://github.com/minimaxir/automl-gs) | Provide an input CSV and a target field to predict, generate a model + code to run it. | ![GitHub Badge](https://img.shields.io/github/stars/minimaxir/automl-gs.svg?style=flat-square) |
| [autokeras](https://github.com/keras-team/autokeras) | AutoML library for deep learning. | ![GitHub Badge](https://img.shields.io/github/stars/keras-team/autokeras.svg?style=flat-square) |
| [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) | Automatic architecture search and hyperparameter optimization for PyTorch. | ![GitHub Badge](https://img.shields.io/github/stars/automl/Auto-PyTorch.svg?style=flat-square) |
| [auto-sklearn](https://github.com/automl/auto-sklearn) | an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator. | ![GitHub Badge](https://img.shields.io/github/stars/automl/auto-sklearn.svg?style=flat-square) |
| [Dragonfly](https://github.com/dragonfly/dragonfly) | An open source python library for scalable Bayesian optimisation. | ![GitHub Badge](https://img.shields.io/github/stars/dragonfly/dragonfly.svg?style=flat-square) |
| [Determined](https://github.com/determined-ai/determined) | scalable deep learning training platform with integrated hyperparameter tuning support; includes Hyperband, PBT, and other search methods. | ![GitHub Badge](https://img.shields.io/github/stars/determined-ai/determined.svg?style=flat-square) |
| [DEvol (DeepEvolution)](https://github.com/joeddav/devol) | a basic proof of concept for genetic architecture search in Keras. | ![GitHub Badge](https://img.shields.io/github/stars/joeddav/devol.svg?style=flat-square) |
| [EvalML](https://github.com/alteryx/evalml) | An open source python library for AutoML. | ![GitHub Badge](https://img.shields.io/github/stars/alteryx/evalml.svg?style=flat-square) |
| [FEDOT](https://github.com/nccr-itmo/FEDOT) | AutoML framework for the design of composite pipelines. | ![GitHub Badge](https://img.shields.io/github/stars/nccr-itmo/FEDOT.svg?style=flat-square) |
| [FLAML](https://github.com/microsoft/FLAML) | Fast and lightweight AutoML ([paper](https://www.microsoft.com/en-us/research/publication/flaml-a-fast-and-lightweight-automl-library/)). | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/FLAML.svg?style=flat-square) |
| [Goptuna](https://github.com/c-bata/goptuna) | A hyperparameter optimization framework, inspired by Optuna. | ![GitHub Badge](https://img.shields.io/github/stars/c-bata/goptuna.svg?style=flat-square) |
| [HpBandSter](https://github.com/automl/HpBandSter) | a framework for distributed hyperparameter optimization. | ![GitHub Badge](https://img.shields.io/github/stars/automl/HpBandSter.svg?style=flat-square) |
| [HPOlib2](https://github.com/automl/HPOlib2) | a library for hyperparameter optimization and black box optimization benchmarks. | ![GitHub Badge](https://img.shields.io/github/stars/automl/HPOlib2.svg?style=flat-square) |
| [Hyperband](https://github.com/zygmuntz/hyperband) | open source code for tuning hyperparams with Hyperband. | ![GitHub Badge](https://img.shields.io/github/stars/zygmuntz/hyperband.svg?style=flat-square) |
| [Hypernets](https://github.com/DataCanvasIO/Hypernets) | A General Automated Machine Learning Framework. | ![GitHub Badge](https://img.shields.io/github/stars/DataCanvasIO/Hypernets.svg?style=flat-square) |
| [Hyperopt](https://github.com/hyperopt/hyperopt) | Distributed Asynchronous Hyperparameter Optimization in Python. | ![GitHub Badge](https://img.shields.io/github/stars/hyperopt/hyperopt.svg?style=flat-square) |
| [hyperunity](https://github.com/gdikov/hypertunity) | A toolset for black-box hyperparameter optimisation. | ![GitHub Badge](https://img.shields.io/github/stars/gdikov/hypertunity.svg?style=flat-square) |
| [Intelli](https://github.com/intelligentnode/Intelli) | A framework to connect a flow of ML models by applying graph theory. | ![GitHub Badge](https://img.shields.io/github/stars/intelligentnode/Intelli?style=flat-square) |
| [Katib](https://github.com/kubeflow/katib) | Katib is a Kubernetes-native project for automated machine learning (AutoML). | ![GitHub Badge](https://img.shields.io/github/stars/kubeflow/katib.svg?style=flat-square) |
| [Keras Tuner](https://github.com/keras-team/keras-tuner) | Hyperparameter tuning for humans. | ![GitHub Badge](https://img.shields.io/github/stars/keras-team/keras-tuner.svg?style=flat-square) |
| [learn2learn](https://github.com/learnables/learn2learn) | PyTorch Meta-learning Framework for Researchers. | ![GitHub Badge](https://img.shields.io/github/stars/learnables/learn2learn.svg?style=flat-square) |
| [Ludwig](https://github.com/uber/ludwig) | a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code. | ![GitHub Badge](https://img.shields.io/github/stars/uber/ludwig.svg?style=flat-square) |
| [MOE](https://github.com/Yelp/MOE) | a global, black box optimization engine for real world metric optimization by Yelp. | ![GitHub Badge](https://img.shields.io/github/stars/Yelp/MOE.svg?style=flat-square) |
| [Model Search](https://github.com/google/model_search) | a framework that implements AutoML algorithms for model architecture search at scale. | ![GitHub Badge](https://img.shields.io/github/stars/google/model_search.svg?style=flat-square) |
| [NASGym](https://github.com/gomerudo/nas-env) | a proof-of-concept OpenAI Gym environment for Neural Architecture Search (NAS). | ![GitHub Badge](https://img.shields.io/github/stars/gomerudo/nas-env.svg?style=flat-square) |
| [NNI](https://github.com/Microsoft/nni) | An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning. | ![GitHub Badge](https://img.shields.io/github/stars/Microsoft/nni.svg?style=flat-square) |
| [Optuna](https://github.com/optuna/optuna) | A hyperparameter optimization framework. | ![GitHub Badge](https://img.shields.io/github/stars/optuna/optuna.svg?style=flat-square) |
| [Pycaret](https://github.com/pycaret/pycaret) | An open-source, low-code machine learning library in Python that automates machine learning workflows. | ![GitHub Badge](https://img.shields.io/github/stars/pycaret/pycaret.svg?style=flat-square) |
| [Ray Tune](github.com/ray-project/ray) | Scalable Hyperparameter Tuning. | ![GitHub Badge](https://img.shields.io/github/stars/ect/ray.svg?style=flat-square) |
| [REMBO](https://github.com/ziyuw/rembo) | Bayesian optimization in high-dimensions via random embedding. | ![GitHub Badge](https://img.shields.io/github/stars/ziyuw/rembo.svg?style=flat-square) |
| [RoBO](https://github.com/automl/RoBO) | a Robust Bayesian Optimization framework. | ![GitHub Badge](https://img.shields.io/github/stars/automl/RoBO.svg?style=flat-square) |
| [scikit-optimize(skopt)](https://github.com/scikit-optimize/scikit-optimize) | Sequential model-based optimization with a `scipy.optimize` interface. | ![GitHub Badge](https://img.shields.io/github/stars/scikit-optimize/scikit-optimize.svg?style=flat-square) |
| [Spearmint](https://github.com/HIPS/Spearmint) | a software package to perform Bayesian optimization. | ![GitHub Badge](https://img.shields.io/github/stars/HIPS/Spearmint.svg?style=flat-square) |
| [TPOT](http://automl.info/tpot/) | one of the very first AutoML methods and open-source software packages. | ![GitHub Badge](https://img.shields.io/github/stars/tpot/.svg?style=flat-square) |
| [Torchmeta](https://github.com/tristandeleu/pytorch-meta) | A Meta-Learning library for PyTorch. | ![GitHub Badge](https://img.shields.io/github/stars/tristandeleu/pytorch-meta.svg?style=flat-square) |
| [Vegas](https://github.com/huawei-noah/vega) | an AutoML algorithm tool chain by Huawei Noah's Arb Lab. | ![GitHub Badge](https://img.shields.io/github/stars/huawei-noah/vega.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Optimizations

| Project | Details | Repository |
|---|---|---|
| [FeatherCNN](https://github.com/Tencent/FeatherCNN) | FeatherCNN is a high performance inference engine for convolutional neural networks. | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/FeatherCNN.svg?style=flat-square) |
| [Forward](https://github.com/Tencent/Forward) | A library for high performance deep learning inference on NVIDIA GPUs. | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/Forward.svg?style=flat-square) |
| [NCNN](https://github.com/Tencent/ncnn) | ncnn is a high-performance neural network inference framework optimized for the mobile platform. | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/ncnn.svg?style=flat-square) |
| [PocketFlow](https://github.com/Tencent/PocketFlow) | use AutoML to do model compression. | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/PocketFlow.svg?style=flat-square) |
| [TensorFlow Model Optimization](https://github.com/tensorflow/model-optimization) | A suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution. | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/model-optimization.svg?style=flat-square) |
| [TNN](https://github.com/Tencent/TNN) | A uniform deep learning inference framework for mobile, desktop and server. | ![GitHub Badge](https://img.shields.io/github/stars/Tencent/TNN.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Federated ML

| Project | Details | Repository |
|---|---|---|
| [EasyFL](https://github.com/EasyFL-AI/EasyFL) | An Easy-to-use Federated Learning Platform | ![GitHub Badge](https://img.shields.io/github/stars/EasyFL-AI/EasyFL.svg?style=flat-square) |
| [FATE](https://github.com/FederatedAI/FATE) | An Industrial Grade Federated Learning Framework | ![GitHub Badge](https://img.shields.io/github/stars/FederatedAI/FATE.svg?style=flat-square) |
| [FedML](https://github.com/FedML-AI/FedML) | The federated learning and analytics library enabling secure and collaborative machine learning on decentralized data anywhere at any scale. Supporting large-scale cross-silo federated learning, cross-device federated learning on smartphones/IoTs, and research simulation. | ![GitHub Badge](https://img.shields.io/github/stars/FedML-AI/FedML.svg?style=flat-square) |
| [Flower](https://github.com/adap/flower) | A Friendly Federated Learning Framework | ![GitHub Badge](https://img.shields.io/github/stars/adap/flower.svg?style=flat-square) |
| [Harmonia](https://github.com/ailabstw/harmonia) | Harmonia is an open-source project aiming at developing systems/infrastructures and libraries to ease the adoption of federated learning (abbreviated to FL) for researches and production usage. | ![GitHub Badge](https://img.shields.io/github/stars/ailabstw/harmonia.svg?style=flat-square) |
| [TensorFlow Federated](https://github.com/tensorflow/federated) | A framework for implementing federated learning | ![GitHub Badge](https://img.shields.io/github/stars/tensorflow/federated.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**

## Awesome Lists

| Project | Details | Repository |
|---|---|---|
| [Awesome Argo](https://github.com/terrytangyuan/awesome-argo) | A curated list of awesome projects and resources related to Argo | ![GitHub Badge](https://img.shields.io/github/stars/terrytangyuan/awesome-argo.svg?style=flat-square) |
| [Awesome AutoDL](https://github.com/D-X-Y/Awesome-AutoDL) | Automated Deep Learning: Neural Architecture Search Is Not the End (a curated list of AutoDL resources and an in-depth analysis) | ![GitHub Badge](https://img.shields.io/github/stars/D-X-Y/Awesome-AutoDL.svg?style=flat-square) |
| [Awesome AutoML](https://github.com/windmaple/awesome-AutoML) | Curating a list of AutoML-related research, tools, projects and other resources | ![GitHub Badge](https://img.shields.io/github/stars/windmaple/awesome-AutoML.svg?style=flat-square) |
| [Awesome AutoML Papers](https://github.com/hibayesian/awesome-automl-papers) | A curated list of automated machine learning papers, articles, tutorials, slides and projects | ![GitHub Badge](https://img.shields.io/github/stars/hibayesian/awesome-automl-papers.svg?style=flat-square) |
| [Awesome-Code-LLM](https://github.com/huybery/Awesome-Code-LLM) | 👨‍💻 An awesome and curated list of best code-LLM for research. | ![GitHub Badge](https://img.shields.io/github/stars/huybery/Awesome-Code-LLM.svg?style=flat-square) |
| [Awesome Federated Learning Systems](https://github.com/AmberLJC/FLsystem-paper/blob/main/README.md) | A curated list of Federated Learning Systems related academic papers, articles, tutorials, slides and projects. | ![GitHub Badge](https://img.shields.io/github/stars/AmberLJC/FLsystem-paper.svg?style=flat-square) |
| [Awesome Federated Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning) | A curated list of federated learning publications, re-organized from Arxiv (mostly) | ![GitHub Badge](https://img.shields.io/github/stars/chaoyanghe/Awesome-Federated-Learning.svg?style=flat-square) |
| [awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning)acc | All materials you need for Federated Learning: blogs, videos, papers, and softwares, etc. | ![GitHub Badge](https://img.shields.io/github/stars/weimingwill/awesome-federated-learning.svg?style=flat-square) |
| [Awesome Open MLOps](https://github.com/fuzzylabs/awesome-open-mlops) | This is the Fuzzy Labs guide to the universe of free and open source MLOps tools. | ![GitHub Badge](https://img.shields.io/github/stars/fuzzylabs/awesome-open-mlops.svg?style=flat-square) |
| [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning) | A curated list of awesome open source libraries to deploy, monitor, version and scale your machine learning | ![GitHub Badge](https://img.shields.io/github/stars/EthicalML/awesome-production-machine-learning.svg?style=flat-square) |
| [Awesome Tensor Compilers](https://github.com/merrymercy/awesome-tensor-compilers) | A list of awesome compiler projects and papers for tensor computation and deep learning. | ![GitHub Badge](https://img.shields.io/github/stars/merrymercy/awesome-tensor-compilers.svg?style=flat-square) |
| [kelvins/awesome-mlops](https://github.com/kelvins/awesome-mlops) | A curated list of awesome MLOps tools. | ![GitHub Badge](https://img.shields.io/github/stars/kelvins/awesome-mlops.svg?style=flat-square) |
| [visenger/awesome-mlops](https://github.com/visenger/awesome-mlops) | Machine Learning Operations - An awesome list of references for MLOps | ![GitHub Badge](https://img.shields.io/github/stars/visenger/awesome-mlops.svg?style=flat-square) |
| [currentslab/awesome-vector-search](https://github.com/currentslab/awesome-vector-search) | A curated list of awesome vector search framework/engine, library, cloud service and research papers to vector similarity search. | ![GitHub Badge](https://img.shields.io/github/stars/currentslab/awesome-vector-search.svg?style=flat-square) |
| [pleisto/flappy](https://github.com/pleisto/flappy) | Production-Ready LLM Agent SDK for Every Developer | ![GitHub Badge](https://img.shields.io/github/stars/pleisto/flappy.svg?style=flat-square) |

**[⬆ back to ToC](#table-of-contents)**
