# Visual-Instruction Tuning for Chart Comprehension and Reasoning

Overview

This repository contains implementations of two systems designed for instruction-tuned chart comprehension and reasoning. Both systems leverage large language models (LLMs) together with specialized vision components to support a broad range of chart-related tasks such as summarization, question answering, fact-checking, and reasoning.

![instruction-examples](https://github.com/MehradShm/CharTex/blob/main/instruction-examples.png)

Instruction Generation

~191K instructions created over ~71K charts.

Tasks include summarization, open-ended QA, fact-checking, chain-of-thought reasoning, code generation, and novel tasks proposed by LLMs.

Gemini was used to generate the underlying datatable for instruction-label generation.
GPT-3.5 used for moderate tasks; GPT-4 used for complex reasoning and novel task creation labels.


Each instruction built from chart data tables + metadata, combined with task-specific prompt templates.

  
![Screenshot 2024-06-21 215938](https://github.com/vis-nlp/ChartInstruct/assets/47740795/a08ceaa3-39a4-48e2-8064-1a76abc7b2e1)

Modeling Details
End-to-End System

Architecture Base: Adapted from LLaVA with vision encoder + adapter + LLM.

Vision Encoder: Chart-specific encoder pretrained for chart images.

Language Models:

Llama2 (7B, decoder-only)

Gemma2 (3B, decoder-only)

Training:

Alignment Stage: Adapter fine-tuned while encoder/LLM frozen.

Instruction Tuning: Encoder frozen; adapter + LLM fine-tuned on instruction data.

Used 4 A100-80GB GPUs for parallel training
