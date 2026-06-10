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
    - [Sampling](#sampling)
    - [Synthesis](#synthesis)
  - [Retrieval Augmented Generation](#retrieval-augmented-generation)
    - [Frameworks](#frameworks)
    - [Document Loaders](#document-loaders)
    - [Structured Data](#structured-data)
  - [Workflow Orchestration & Agent Frameworks](#workflow-orchestration--agent-frameworks)
  - [Sandbox](#sandbox)
  - [Eval](#eval)
  - [ML Platform](#ml-platform)
  - [Misc.](#misc)

Note: This awesome list focuses only on open-source tools. We welcome tools that are free and open source.

## Model

### Large Language Model

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [Aquila](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila) | Aquila Language Model is the first open source language model that supports both Chinese and English knowledge, Chinese and English instructions fine-tuning, with a context length of 2048. | ![GitHub Badge](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg?style=flat-square) |
| [Baichuan](https://github.com/baichuan-inc/Baichuan-13B) | A large-scale 13B pretraining language model developed by Baichuan Intelligent Technology | ![GitHub Badge](https://img.shields.io/github/stars/baichuan-inc/Baichuan-13B.svg?style=flat-square) |
| [Bloom](https://huggingface.co/bigscience/bloom) | BigScience Large Open-science Open-access Multilingual Language Model | - |
| [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) | ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. | ![GitHub Badge](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg?style=flat-square) |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | Chinese LLaMA & Alpaca LLMs | ![GitHub Badge](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg?style=flat-square) |
| [Falcon](https://huggingface.co/tiiuae/falcon-40b) | Falcon-40B is a 40B parameters causal decoder-only model built by TII and trained on 1,000B tokens of RefinedWeb enhanced with curated corpora. | - |
| [Gemma](https://github.com/google-deepmind/gemma) | Open weights LLM from Google DeepMind. | ![GitHub Badge](https://img.shields.io/github/stars/google-deepmind/gemma.svg?style=flat-square) |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | gpt4all: an ecosystem of open-source chatbots trained on a massive collections of clean assistant data including code, stories and dialogue | ![GitHub Badge](https://img.shields.io/github/stars/nomic-ai/gpt4all.svg?style=flat-square) |
| [Grok-1](https://github.com/xai-org/grok-1) | Grok open release | ![GitHub Badge](https://img.shields.io/github/stars/xai-org/grok-1.svg?style=flat-square) |
| [LLaMA](https://github.com/facebookresearch/llama) | Inference code for LLaMA models | ![GitHub Badge](https://img.shields.io/github/stars/facebookresearch/llama.svg?style=flat-square) |
| [Mistral](https://github.com/mistralai/mistral-src) | Reference implementation of Mistral AI 7B v0.1 model | ![GitHub Badge](https://img.shields.io/github/stars/mistralai/mistral-src.svg?style=flat-square) |
| [MPT-7B](https://github.com/mosaicml/llm-foundry) | MPT-7B is a GPT-style model with 7 billion parameters. | ![GitHub Badge](https://img.shields.io/github/stars/mosaicml/llm-foundry.svg?style=flat-square) |
| [OpenLLaMA](https://github.com/openlm-research/open_llama) | OpenLLaMA, a permissively licensed open source reproduction of Meta AI's LLaMA 7B trained on the RedPajama dataset | ![GitHub Badge](https://img.shields.io/github/stars/openlm-research/open_llama.svg?style=flat-square) |
| [Phi-3](https://github.com/microsoft/Phi-3CookBook) | Phi-3 is a family of open AI models developed by Microsoft | ![GitHub Badge](https://img.shields.io/github/stars/microsoft/Phi-3CookBook.svg?style=flat-square) |
| [Qwen](https://github.com/QwenLM/Qwen) | The official repo of Qwen (通义千问) chat & pretrained large language model proposed by Alibaba Cloud. | ![GitHub Badge](https://img.shields.io/github/stars/QwenLM/Qwen.svg?style=flat-square) |
| [Vicuna](https://github.com/lm-sys/FastChat) | An open platform for training, serving, and evaluating large language chatbots. | ![GitHub Badge](https://img.shields.io/github/stars/lm-sys/FastChat.svg?style=flat-square) |
| [Yi](https://github.com/01-ai/Yi) | A series of large language models trained from scratch by developers @01-ai | ![GitHub Badge](https://img.shields.io/github/stars/01-ai/Yi.svg?style=flat-square) |

### CV Foundation Model

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [CLIP](https://github.com/openai/CLIP) | Contrastive Language-Image Pretraining | ![GitHub Badge](https://img.shields.io/github/stars/openai/CLIP.svg?style=flat-square) |
| [SAM](https://github.com/facebookresearch/segment-anything) | The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. | ![GitHub Badge](https://img.shields.io/github/stars/facebookresearch/segment-anything.svg?style=flat-square) |
| [Stable Diffusion](https://github.com/CompVis/stable-diffusion) | A latent text-to-image diffusion model | ![GitHub Badge](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?style=flat-square) |

### Audio Foundation Model

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [AudioCraft](https://github.com/facebookresearch/audiocraft) | AudioCraft is a PyTorch library for deep learning research on audio generation. | ![GitHub Badge](https://img.shields.io/github/stars/facebookresearch/audiocraft.svg?style=flat-square) |
| [Whisper](https://github.com/openai/whisper) | Robust Speech Recognition via Large-Scale Weak Supervision | ![GitHub Badge](https://img.shields.io/github/stars/openai/whisper.svg?style=flat-square) |

### Robotics Foundation Model

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [OpenVLA](https://github.com/openvla/openvla) | OpenVLA: An Open-Source Vision-Language-Action Model | ![GitHub Badge](https://img.shields.io/github/stars/openvla/openvla.svg?style=flat-square) |
| [RT-2](https://github.com/kyegomez/RT-2) | Implementation of RT-2: Vision-Language-Action Models | ![GitHub Badge](https://img.shields.io/github/stars/kyegomez/RT-2.svg?style=flat-square) |

## Serving

### Large Model Serving

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [BentoML](https://github.com/bentoml/BentoML) | Build Production-Grade AI Applications | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=flat-square) |
| [Haystack](https://github.com/deepset-ai/haystack) | LLM orchestration framework to build customizable, production-ready LLM applications. Connect components (models, vector DBs, file converters) to pipelines or agents that can interact with your data. With advanced retrieval methods, it's best suited for building RAG, question answering, semantic search or conversational agent chatbots. | ![GitHub Badge](https://img.shields.io/github/stars/deepset-ai/haystack.svg?style=flat-square) |
| [HuggingFace TGI](https://github.com/huggingface/text-generation-inference) | Large Language Model Text Generation Inference | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/text-generation-inference.svg?style=flat-square) |
| [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) | Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text outputs. | - |
| [LocalAI](https://github.com/mudler/LocalAI) | Free, Open Source OpenAI alternative. Self-hosted, community-driven and local-first. Drop-in replacement for OpenAI running on consumer-grade hardware. | ![GitHub Badge](https://img.shields.io/github/stars/mudler/LocalAI.svg?style=flat-square) |
| [Mosec](https://github.com/mosecorg/mosec) | A high-performance and flexible model serving framework for building ML model-enabled backend and microservice | ![GitHub Badge](https://img.shields.io/github/stars/mosecorg/mosec.svg?style=flat-square) |
| [Ollama](https://github.com/jmorganca/ollama) | Get up and running with Llama 2 and other large language models locally | ![GitHub Badge](https://img.shields.io/github/stars/jmorganca/ollama.svg?style=flat-square) |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | Operating LLMs in production | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/OpenLLM.svg?style=flat-square) |
| [RayLLM](https://github.com/ray-project/ray-llm) | RayLLM - LLMs on Ray | ![GitHub Badge](https://img.shields.io/github/stars/ray-project/ray-llm.svg?style=flat-square) |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Run LLMs, AI, and Batch jobs on any cloud. Get maximum savings, highest GPU availability, and managed execution—all with a simple interface. | ![GitHub Badge](https://img.shields.io/github/stars/skypilot-org/skypilot.svg?style=flat-square) |
| [vLLM](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs | ![GitHub Badge](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=flat-square) |

### Frameworks/Servers for Serving

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [FastAPI](https://github.com/tiangolo/fastapi) | FastAPI framework, high performance, easy to learn, fast to code, ready for production | ![GitHub Badge](https://img.shields.io/github/stars/tiangolo/fastapi.svg?style=flat-square) |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | The Triton Inference Server provides an optimized cloud and edge inferencing solution. | ![GitHub Badge](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=flat-square) |

## Security

### Frameworks for LLM security

| Project | Details | Repository |
| ------- | ------- | ---------- |
| [Garak](https://github.com/leondz/garak) | LLM vulnerability scanner | ![GitHub Badge](https://img.shields.io/github/stars/leondz/garak.svg?style=flat-square) |
| [LLM Guard](https://github.com/laiyer-ai/llm-guard) | The Security Toolkit for LLM Interactions | ![GitHub Badge](https://img.shields.io/github/stars/laiyer-ai/llm-guard.svg?style=flat-square) |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational applications. | ![GitHub Badge](https://img.shields.io/github/stars/NVIDIA/NeMo-Guardrails.svg?style=flat-square) |
| [Pebblo](https://github.com/daxa-ai/pebblo) | Pebblo enables developers to safely load data and promote their Gen AI app to deployment without worrying about the organization's compliance and security requirements. | ![GitHub Badge](https://img.shields.io/github/stars/daxa-ai/pebblo.svg?style=flat-square) |
| [Purple Llama](https://github.com/meta-llama/PurpleLlama) | Set of tools to assess and improve LLM security. | ![GitHub Badge](https://img.shields.io/github/stars/meta-llama/PurpleLlama.svg?style=flat-square) |

### Observability

| Project                                                                        | Details                                                                                                                                                                                                                        | Repository                                                                                                       |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| [Azure OpenAI Logger](https://github.com/aavetis/azure-openai-logger)          | "Batteries included" logging solution for your Azure OpenAI instance.                                                                                                                                                          | ![GitHub Badge](https://img.shields.io/github/stars/aavetis/azure-openai-logger?style=flat-square)               |
| [ClevAgent](https://clevagent.io)                                              | Runtime monitoring for AI agents — heartbeat watchdog, loop detection, cost tracking, auto-restart. Python SDK or HTTP API.                                                                                                    |                                                                                                                  |
| [Deepchecks](https://github.com/deepchecks/deepchecks)                         | Tests for Continuous Validation of ML Models & Data. Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.                                                  | ![GitHub Badge](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=flat-square)                 |
| [Evidently](https://github.com/evidentlyai/evidently)                          | An open-source framework to evaluate, test and monitor ML and LLM-powered systems.                                                                                                                                             | ![GitHub Badge](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=flat-square)                 |
| [EvalView](https://github.com/hidai25/eval-view)                              | Regression testing for AI agents. Snapshot behavior, detect tool-call and output regressions, with golden-baseline diffing and LLM-as-judge scoring. Supports LangGraph, CrewAI, OpenAI, Claude, and any HTTP API.             | ![GitHub Badge](https://img.shields.io/github/stars/hidai25/eval-view.svg?style=flat-square)                     |
| [Fiddler AI](https://github.com/fiddler-labs/fiddler-auditor)                  | Evaluate, monitor, analyze, and improve machine learning and generative models from pre-production to production. Ship more ML and LLMs into production, and monitor ML and LLM metrics like hallucination, PII, and toxicity. | ![GitHub Badge](https://img.shields.io/github/stars/fiddler-labs/fiddler-auditor.svg?style=flat-square)          |
| [Giskard](https://github.com/Giskard-AI/giskard)                               | Testing framework dedicated to ML models, from tabular to LLMs. Detect risks of biases, performance issues and errors in 4 lines of code.                                                                                      | ![GitHub Badge](https://img.shields.io/github/stars/Giskard-AI/giskard.svg?style=flat-square)
| [NeBeso](https://github.com/badBoyDevop/nebeso-platform)                       | Open-source AI cost tracking — monitor tokens, cost, and latency per feature, user, and project. Works with OpenAI, Anthropic, Google, Mistral, and 40+ models. LangChain and LlamaIndex integrations included. | ![GitHub Badge](https://img.shields.io/github/stars/badBoyDevop/nebeso-platform.svg?style=flat-square)           |
| [QWED](https://github.com/QWED-AI/qwed-verification) | Deterministic verification protocol for LLM outputs using 8 formal verification engines (SymPy, Z3, AST, SQLGlot). Prevents hallucinations through mathematical proofs rather than statistical methods. | ![GitHub Badge](https://img.shields.io/github/stars/QWED-AI/qwed-verification.svg?style=flat-square) |
| [Great Expectations](https://github.com/great-expectations/great_expectations) | Always know what to expect from your data.                                                                                                                                                                                     | ![GitHub Badge](https://img.shields.io/github/stars/great-expectations/great_expectations.svg?style=flat-square) |
| [Helicone](https://github.com/Helicone/helicone)                              | Open source LLM observability platform. One line of code to monitor, evaluate, and experiment with features like prompt management, agent tracing, and evaluations.                                                            | ![GitHub Badge](https://img.shields.io/github/stars/Helicone/helicone.svg?style=flat-square)                     |
| [Traceloop OpenLLMetry](https://github.com/traceloop/openllmetry)                              | OpenTelemetry-based observability and monitoring for LLM and agents workflows.                                                           | ![GitHub Badge](https://img.shields.io/github/stars/traceloop/openllmetry.svg?style=flat-square)    |
| [Langfuse 🪢](https://langfuse.com) | Open-source LLM observability platform that helps teams collaboratively debug, analyze, and iterate on their LLM applications. | ![GitHub Badge](https://img.shields.io/github/stars/langfuse/langfuse.svg?style=flat-square)              |
| [whylogs](https://github.com/whylabs/whylogs)                                  | The open standard for data logging                                                                                                                                                                                             | ![GitHub Badge](https://img.shields.io/github/stars/whylabs/whylogs.svg?style=flat-square)                       |
| [onWatch](https://github.com/onllm-dev/onwatch) | Lightweight Go CLI that tracks AI API quota usage across 7 providers (Anthropic, OpenAI, GitHub Copilot, MiniMax, and more). Background daemon, <50MB RAM, zero telemetry, SQLite storage. | ![GitHub Badge](https://img.shields.io/github/stars/onllm-dev/onwatch.svg?style=flat-square) |
