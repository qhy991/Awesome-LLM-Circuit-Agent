# Awesome LLM Circuit Agent

![Awesome LLM Circuit Agent](assets/cover.png)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A curated list of papers, datasets, and resources related to **Large Language Models (LLMs) for Circuit Design**, covering both Digital (RTL) and Analog domains. This repository aims to track the rapid advancements in using AI agents for hardware design automation.

## 📖 Table of Contents

- [Digital Circuit Design (RTL)](#-digital-circuit-design-rtl)
  - [Code Generation & Synthesis](#-code-generation--synthesis)
  - [Verification & Testing](#-verification--testing)
  - [Optimization (PPA-aware)](#-optimization-ppa-aware)
  - [Reinforcement Learning Approaches](#-reinforcement-learning-approaches)
  - [Multi-Agent Systems & Workflows](#-multi-agent-systems--workflows)
  - [Reasoning & Graph-Based](#-reasoning--graph-based)
- [Analog Circuit Design](#-analog-circuit-design)
  - [Topology & Schematic Generation](#-topology--schematic-generation)
  - [Sizing & Optimization](#-sizing--optimization)
  - [Workflows & Multi-Agent](#-workflows--multi-agent)
  - [Specialized Applications](#-specialized-applications)
- [Analog Mind Series (Behzad Razavi)](#-analog-mind-series-behzad-razavi)
- [Datasets & Benchmarks](#-datasets--benchmarks)
- [Resources & Learning](#-resources--learning)
- [Contributing](#-contributing)

---

## 💻 Digital Circuit Design (RTL)

### 📝 Code Generation & Synthesis

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer**](https://arxiv.org/abs/2511.19472) | AAAI 2026 | 2025.11 | [Github](https://github.com/Mightlaus/PrefixGPT-AAAI26) | Prefix Adder, Transformer |
| [**QiMeng-CRUX: Narrowing the Gap between Natural Language and Verilog via Core Refined Understanding eXpression**](https://arxiv.org/abs/2511.20099) | arXiv | 2025.11 | - | NL2Verilog, CRUX |
| [**LocalV: Exploiting Information Locality for IP-level Verilog Generation**](https://openreview.net/forum?id=jiFcyj5VLe) | ICLR 2026 | 2025.09 | - | Verilog, IP-level |
| [**SPARC-RTL: Stochastic Prompt-Assisted RTL Code Synthesis**](https://openreview.net/forum?id=VdoEQJufI8) | ICLR 2026 | 2025.09 | - | Prompt Engineering |
| [**VeriGRAG: Enhancing LLM-Based Verilog Code Generation with Structure-Aware Soft Prompts**](https://arxiv.org/abs/2510.15914) | arXiv | 2025.10 | - | Structure-Aware |
| [**DeepV: A Model-Agnostic Retrieval-Augmented Framework for Verilog Code Generation**](https://arxiv.org/abs/2510.05327) | arXiv | 2025.10 | [Space](https://huggingface.co/spaces/FICS-LLM/DeepV) | RAG |
| [**CodeV: Empowering LLMs with HDL Generation through Multi-Level Summarization**](https://arxiv.org/abs/2407.10424) | arXiv | 2024.07 | [Model](https://huggingface.co/yang-z/CodeV-DS-6.7B) | Summarization |
| [**Data is all you need: Finetuning LLMs for Chip Design via an Automated design-data augmentation framework**](https://arxiv.org/abs/2403.11202) | DAC 2024 | 2024.03 | - | Finetuning |
| [**VeriGen: A Large Language Model for Verilog Code Generation**](https://arxiv.org/abs/2308.00708) | arXiv | 2023.07 | [Model](https://huggingface.co/shailja/fine-tuned-codegen-2B-Verilog) | Finetuning |
| [**RTL-LLM: Large Language Models for Hardware Design**](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-104.pdf) | UC Berkeley | 2025 | - | Multi-Language |

### ✅ Verification & Testing

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**R3A: Reliable RTL Repair Framework with Multi-Agent Fault Localization and Stochastic Tree-of-Thoughts Patch Generation**](https://arxiv.org/abs/2511.20090) | arXiv | 2025.11 | - | RTL Repair, Multi-Agent |
| [**TB or Not TB: Coverage-Driven Direct Preference Optimization for Verilog Stimulus Generation**](https://arxiv.org/abs/2511.15767) | arXiv | 2025.11 | - | Stimulus Gen, DPO |
| [**Automating Hardware Design and Verification from Architectural Papers via a Neural-Symbolic Graph Framework**](https://arxiv.org/abs/2511.06067) | arXiv | 2025.11 | - | Neural-Symbolic |
| [**Think with Self-Decoupling and Self-Verification: Automated RTL Design with Backtrack-ToT**](https://arxiv.org/abs/2511.13139) | arXiv | 2025.11 | - | Self-Verification |
| ![Star](https://img.shields.io/github/stars/AgenticHDL/CorrectHDL.svg?style=social&label=Star) <br> [**CorrectHDL: Agentic HDL Design with LLMs Leveraging High-Level Synthesis as Reference**](https://arxiv.org/abs/2511.16395) | arXiv | 2025.11 | [Github](https://github.com/AgenticHDL/CorrectHDL) | HLS, RAG |
| [**BugGen: A Self-Correcting Multi-Agent LLM Pipeline for Realistic RTL Bug Synthesis**](https://arxiv.org/abs/2506.10501) | arXiv | 2025.06 | - | Bug Synthesis, Multi-Agent |
| [**VeriSynth: Learning-Based Framework for Formal Verification of Hardware Designs**](https://arxiv.org/pdf/2505.09172) | arXiv | 2025.05 | [Github](https://github.com/eelab-dev/VeriSynth) | Formal Verification |
| [**RTL-Repair: Fast Symbolic Repair of Hardware Design Code**](https://kevinlaeufer.com/pdfs/rtl_repair_kevin_laeufer_asplos2024.pdf) | ASPLOS 2024 | 2024.04 | [Github](https://github.com/ekiwi/rtl-repair) | RTL Repair, Symbolic |

### 🚀 Optimization (PPA-aware)

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**LLM-VeriPPA: Power, Performance, and Area Optimization aware Verilog Code Generation**](https://arxiv.org/abs/2510.15899) | arXiv | 2025.10 | - | PPA Optimization |
| [**ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven RL**](https://arxiv.org/abs/2507.04736) | arXiv | 2025.07 | - | RL, PPA |
| ![Star](https://img.shields.io/github/stars/ABKGroup/ORFS-Agent.svg?style=social&label=Star) <br> [**ORFS-agent: Tool-Using Agents for Chip Design Optimization**](https://arxiv.org/abs/2506.08332) | arXiv | 2025.06 | [Github](https://github.com/ABKGroup/ORFS-Agent) | Physical Design |
| [**SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning**](https://arxiv.org/abs/2504.10369) | arXiv | 2025.04 | - | Symbolic Reasoning |
| [**Improving Large Language Model Hardware Generating Quality through Post-LLM Search**](https://mlforsystems.org/assets/papers/neurips2023/paper12.pdf) | NeurIPS 2023 | 2023.12 | - | Post-LLM Search |

### 🤖 Reinforcement Learning Approaches

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**RTLSeek: Boosting the LLM-Based RTL Generation with Diversity-Oriented RL**](https://openreview.net/forum?id=qO7g1dToiO) | ICLR 2026 | 2025.09 | - | Diversity-Oriented |
| [**EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation**](https://arxiv.org/abs/2511.12033) | arXiv | 2025.11 | - | Entropy-Aware |
| ![Star](https://img.shields.io/github/stars/kmcho2019/REvolution.svg?style=social&label=Star) <br> [**REvolution: An Evolutionary Framework for RTL Generation driven by LLMs**](https://arxiv.org/abs/2510.21407) | ASP-DAC 2026 | 2025.10 | [Github](https://github.com/kmcho2019/REvolution) | Evolutionary Algo |
| ![Star](https://img.shields.io/github/stars/omniAI-Lab/VeriRL.svg?style=social&label=Star) <br> [**VERIRL: Boosting the LLM-based Verilog Code Generation via Reinforcement Learning**](https://arxiv.org/abs/2508.18462) | arXiv | 2025.08 | [Github](https://github.com/omniAI-Lab/VeriRL) | RL |
| ![Star](https://img.shields.io/github/stars/NellyW8/VeriReason.svg?style=social&label=Star) <br> [**VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog**](https://openreview.net/forum?id=bkU1bQUSQD) | ICLR 2026 | 2025.09 | [Github](https://github.com/NellyW8/VeriReason) | RL, Reasoning |
| [**Improving LLM-Based Verilog Code Generation with Data Augmentation and RL**](https://ieeexplore.ieee.org/document/10992897) | DATE 2025 | 2025.03 | - | Data Augmentation |
| [**Large Language Model for Verilog Generation with Code-Structure-Guided RL**](https://arxiv.org/html/2407.18271v4) | arXiv | 2024.07 | [Code](https://anonymous.4open.science/r/veriseek-6467) | Structure-Guided |

### 🤝 Multi-Agent Systems & Workflows

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**CRADLE: Conversational RTL Design Space Exploration with LLM-based Multi-Agent Systems**](https://arxiv.org/abs/2508.08709) | arXiv | 2025.08 | - | DSE, Multi-Agent |
| [**VFlow: Discovering Optimal Agentic Workflows for Verilog Generation**](https://arxiv.org/abs/2504.03723) | arXiv | 2025.04 | - | Agentic Workflow |

### 🧠 Reasoning & Graph-Based

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**RTL++: Graph-enhanced LLM for RTL Code Generation**](https://arxiv.org/abs/2505.13479) | LAD '25 | 2025.05 | - | Graph-enhanced |
| [**Abstractions-of-Thought: Intermediate Representations for LLM Reasoning in Hardware Design**](https://arxiv.org/abs/2505.15873) | arXiv | 2025.05 | - | IR, Reasoning |
| [**CIRCUIT: A Benchmark for Circuit Interpretation and Reasoning Capabilities of LLMs**](https://arxiv.org/pdf/2502.07980) | arXiv | 2025.02 | - | Reasoning |
| ![Star](https://img.shields.io/github/stars/BUAA-Clab/ReasoningV.svg?style=social&label=Star) <br> [**ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model**](https://arxiv.org/abs/2504.14560) | arXiv | 2025.04 | [Github](https://github.com/BUAA-Clab/ReasoningV) | Hybrid Reasoning |

---

## ⚡ Analog Circuit Design

### 📐 Topology & Schematic Generation

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit**](https://arxiv.org/abs/2510.17002) | arXiv | 2025.10 | [Github](https://github.com/eelab-dev/EEschematic) | MLLM, Schematic |
| [**DiffCkt: A Diffusion Model-Based Hybrid Neural Network Framework for Automatic Transistor-Level Generation**](https://arxiv.org/pdf/2507.00444) | arXiv | 2025.07 | - | Diffusion Model |
| [**SpiceMixer: Netlist-Level Circuit Evolution**](https://arxiv.org/pdf/2506.01497) | arXiv | 2025.06 | - | Netlist Evolution |
| [**Schemato -- An LLM for Netlist-to-Schematic Conversion**](https://arxiv.org/pdf/2411.13899) | arXiv | 2024.11 | - | Netlist-to-Schematic |
| [**LaMAGIC: Language-Model-based Topology Generation for Analog Integrated Circuits**](https://arxiv.org/pdf/2407.18269) | arXiv | 2024.07 | - | Topology Generation |

### 📏 Sizing & Optimization

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**HeaRT: A Hierarchical Circuit Reasoning Tree-Based Agentic Framework for AMS Design Optimization**](https://arxiv.org/abs/2511.19669) | arXiv | 2025.11 | - | Reasoning, Optimization |
| ![Star](https://img.shields.io/github/stars/eelab-dev/EEsizer.svg?style=social&label=Star) <br> [**EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit**](https://arxiv.org/pdf/2509.25510) | arXiv | 2025.09 | [Github](https://github.com/eelab-dev/EEsizer) | Transistor Sizing |
| [**TopoSizing: An LLM-aided Framework of Topology-based Understanding and Sizing for AMS Circuits**](https://arxiv.org/pdf/2509.14169) | arXiv | 2025.09 | - | Topology-based |
| [**White-Box Reasoning: Synergizing LLM Strategy and gm/Id Data for Automated Analog Circuit Design**](https://arxiv.org/abs/2508.13172) | arXiv | 2025.08 | - | gm/Id, White-Box |
| [**RoSE-Opt: Robust and Efficient Analog Circuit Parameter Optimization with Knowledge-infused RL**](https://arxiv.org/pdf/2407.19150) | arXiv | 2024.07 | - | RL, Optimization |
| [**LLM-Enhanced Bayesian Optimization for Efficient Analog Layout Constraint Generation**](https://arxiv.org/pdf/2406.05250) | arXiv | 2024.06 | - | Bayesian Opt |
| [**Learning-driven Physically-aware Large-scale Circuit Gate Sizing**](https://arxiv.org/pdf/2403.08193) | arXiv | 2024.03 | - | Gate Sizing |

### 🔄 Workflows & Multi-Agent

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing**](https://arxiv.org/pdf/2511.03697) | arXiv | 2025.11 | - | Workflow |
| ![Star](https://img.shields.io/github/stars/laiyao1/AnalogCoderPro.svg?style=social&label=Star) <br> [**AnalogCoder-Pro: Unifying Analog Circuit Generation and Optimization via Multi-modal LLMs**](https://arxiv.org/abs/2508.02518) | arXiv | 2025.08 | [Github](https://github.com/laiyao1/AnalogCoderPro) | MLLM, Unifying |
| [**A Large Language Model-based Multi-Agent Framework for Analog Circuits' Sizing Relationships Extraction**](https://arxiv.org/pdf/2506.18424) | arXiv | 2025.06 | - | Sizing Relationships |
| [**Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence**](https://arxiv.org/abs/2504.14625) | arXiv | 2025.04 | - | Multi-Agent |

### 🔬 Specialized Applications

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**AnalogTester: A Large Language Model-Based Framework for Automatic Testbench Generation**](https://arxiv.org/abs/2507.09965) | arXiv | 2025.07 | - | Testbench Generation |
| [**LIMCA: LLM for Automating Analog In-Memory Computing Architecture Design Exploration**](https://arxiv.org/abs/2503.13301) | arXiv | 2025.03 | - | In-Memory Computing |
| [**FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design**](https://arxiv.org/pdf/2505.21923) | arXiv | 2025.05 | - | Layout-Constrained |
| [**DocEDA: Automated Extraction and Design of Analog Circuits from Documents with Large Language Model**](https://arxiv.org/pdf/2412.05301) | arXiv | 2024.12 | - | Document Extraction |
| [**AICircuit: A Multi-Level Dataset and Benchmark for AI-Driven Analog Integrated Circuit Design**](https://arxiv.org/pdf/2407.18272) | arXiv | 2024.07 | - | Dataset, Benchmark |
| [**DE-HNN: An effective neural model for Circuit Netlist representation**](https://arxiv.org/pdf/2404.00477) | arXiv | 2024.04 | - | Netlist Representation |
| [**Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis**](https://arxiv.org/pdf/2502.11812) | arXiv | 2025.02 | - | Circuit Analysis |

---

## 📊 Datasets & Benchmarks

| Title | Venue | Date | Code | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**VERIBENCH: End-to-End Formal Verification Benchmark for AI Code Generation in Lean 4**](https://openreview.net/pdf/f24bd52a5b9139e4311109bdeee80b27c311d838.pdf) | ICLR 2026 | 2025 | - | Formal Verification |
| [**Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code**](https://openreview.net/forum?id=2LmXLuCDsY) | ICLR 2026 | 2025.09 | - | Efficiency Benchmark |
| [**Refining Specs For LLM-Based RTL Agile Design**](https://openreview.net/forum?id=1FADg2UNPn) | ICLR 2026 | 2025.09 | - | Spec Refining |
| ![Star](https://img.shields.io/github/stars/scale-lab/MetRex.svg?style=social&label=Star) <br> [**MetRex: A Benchmark for Verilog Code Metric Reasoning Using LLMs**](https://arxiv.org/abs/2411.03471) | ASP-DAC 2025 | 2025.01 | [Github](https://github.com/scale-lab/MetRex) | Metric Reasoning |
| ![Star](https://img.shields.io/github/stars/hkust-zhiyao/RTLLM.svg?style=social&label=Star) <br> [**RTLLM: An Open-Source Benchmark for Design RTL Generation with Large Language Model**](https://arxiv.org/abs/2402.03375) | ASP-DAC 2024 | 2024.01 | [Github](https://github.com/hkust-zhiyao/RTLLM) | RTL Benchmark |
| ![Star](https://img.shields.io/github/stars/NVlabs/verilog-eval.svg?style=social&label=Star) <br> [**VerilogEval: Evaluating Large Language Models for Verilog Code Generation**](https://arxiv.org/abs/2308.05345) | ICCAD 2023 | 2023.10 | [Github](https://github.com/NVlabs/verilog-eval) | Verilog Benchmark |
| [**ReasoningV-5K Dataset**](https://huggingface.co/datasets/GipAI/ReaoningV) | HuggingFace | 2025.04 | [Dataset](https://huggingface.co/datasets/GipAI/ReaoningV) | Reasoning Dataset |
| [**PyraNet-Verilog Dataset**](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) | HuggingFace | 2024.07 | [Dataset](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) | Verilog Dataset |
| [**Verilog_GitHub Dataset**](https://huggingface.co/datasets/shailja/Verilog_GitHub) | HuggingFace | 2023.07 | [Dataset](https://huggingface.co/datasets/shailja/Verilog_GitHub) | Verilog Dataset |
| [**VHDL GitHub Deduplicated**](https://huggingface.co/datasets/rtl-llm/vhdl_github_deduplicated) | HuggingFace | 2025 | [Dataset](https://huggingface.co/datasets/rtl-llm/vhdl_github_deduplicated) | VHDL Dataset |
| [**Chisel-Verilog Pairs**](https://huggingface.co/datasets/rtl-llm/chisel-verilog-pairs) | HuggingFace | 2025 | [Dataset](https://huggingface.co/datasets/rtl-llm/chisel-verilog-pairs) | Chisel Dataset |
| [**PyMTL-Verilog Pairs**](https://huggingface.co/datasets/rtl-llm/PyMTL_Verilog_pairs) | HuggingFace | 2025.05 | [Dataset](https://huggingface.co/datasets/rtl-llm/PyMTL_Verilog_pairs) | PyMTL Dataset |

---

## 🧠 Analog Mind Series (Behzad Razavi)

A series of articles by Prof. Behzad Razavi published in IEEE Solid-State Circuits Magazine (SSCM), exploring fundamental concepts and advanced topics in analog circuit design.

| Title | Venue | Date | Link | Topic |
|:------|:-----:|:----:|:----:|:------|
| [**Analog Mind (Part 1)**](https://ieeexplore.ieee.org/document/10410055) | IEEE SSCM | 2024.Q1 | [IEEE](https://ieeexplore.ieee.org/document/10410055) | Analog Design Fundamentals |
| [**Analog Mind (Part 2)**](https://ieeexplore.ieee.org/document/10645490) | IEEE SSCM | 2024.Q2 | [IEEE](https://ieeexplore.ieee.org/document/10645490) | Analog Design Concepts |
| [**Analog Mind (Part 3)**](https://www.seas.ucla.edu/brweb/papers/Journals/BR_SSCM_3_2024.pdf) | IEEE SSCM | 2024.Q3 | [PDF](https://www.seas.ucla.edu/brweb/papers/Journals/BR_SSCM_3_2024.pdf) | Advanced Analog Topics |
| [**Analog Mind (Part 4)**](https://www.seas.ucla.edu/brweb/papers/Journals/BR_SSCM_4_2025.pdf) | IEEE SSCM | 2025.Q1 | [PDF](https://www.seas.ucla.edu/brweb/papers/Journals/BR_SSCM_4_2025.pdf) | Advanced Analog Topics |
| [**Analog Mind (Part 5)**](https://ieeexplore.ieee.org/document/10752795) | IEEE SSCM | 2024.Q4 | [IEEE](https://ieeexplore.ieee.org/document/10752795) | Analog Design Insights |
| [**Analog Mind (Part 6)**](https://ieeexplore.ieee.org/document/10857808) | IEEE SSCM | 2025.Q1 | [IEEE](https://ieeexplore.ieee.org/document/10857808) | Analog Design Insights |
| [**Analog Mind (Part 7)**](https://ieeexplore.ieee.org/document/11044975) | IEEE SSCM | 2025.Q2 | [IEEE](https://ieeexplore.ieee.org/document/11044975) | Analog Design Insights |
| [**Analog Mind (Part 8)**](https://ieeexplore.ieee.org/document/11262742) | IEEE SSCM | 2025.Q3 | [IEEE](https://ieeexplore.ieee.org/document/11262742) | Analog Design Insights |

*For complete list of Analog Mind articles, see [Behzad Razavi's IEEE Author Page](https://ieeexplore.ieee.org/author/37275476000)*

---

## 📚 Resources & Learning

| Title | Type | Topic |
|:------|:----:|:------|
| [**BrainWave NPU Microarchitecture Analysis**](https://github.com/dzwduan/fpga-npu/tree/main/doc) | Docs | NPU Architecture |
| [**EEschematic Presentation**](https://docs.google.com/presentation/d/e/2PACX-1vROdrVB1vpGM1tqHSvA2HpPmH6B2HpILzLM8kaqnePEtZ8UP_To8q5GsWh90YOtBjYZCUov2rnOzis7/pub?start=false&loop=false&delayms=3000&slide=id.p1) | Slides | AMS Circuit |
| [**ASIC Technology Lecture**](https://schaumont.dyn.wpi.edu/ece574f24/01asictechnology.html) | Course | ASIC |
| [**Digital System Design PDF**](https://d1.amobbs.com/bbs_upload782111/files_19/ourdev_489875.pdf) | PDF | Digital Design |
| [**Springer Book: Digital System Design**](https://link.springer.com/book/10.1007/978-3-031-41085-7?utm_medium=referral) | Book | Digital Design |

---

## 🤝 Contributing

We welcome contributions! If you know of a paper, tool, or resource that should be included, please:

1. **Fork** this repository
2. **Add** your entry following the existing format
3. **Submit** a pull request with a brief description

### Contribution Guidelines

- Ensure the paper/resource is relevant to LLM-based circuit design (RTL/Analog) or hardware automation
- Include proper citation with title, venue, date, and links
- Add appropriate topic tags
- Maintain chronological order (newest first)
- Check for duplicates before submitting

---

## 📄 Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@misc{awesome-llm-circuit-agent,
  author = {Haiyan Qin},
  title = {Awesome LLM Circuit Agent: A Curated Collection of LLM-Driven Circuit Design Research},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/qhy991/Awesome-LLM-Circuit-Agent}
}
```

---

## 📜 License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under a [Creative Commons Zero v1.0 Universal](LICENSE) license.

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

Maintained with ❤️ by the community

*Last Updated: November 2025*

</div>
