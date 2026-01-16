# GEMINI - Project Context

## Overview
This repository is a NestJS/TensorFlow.js implementation of the Transformer architecture described in the "Attention Is All You Need" paper.

## Current Status
- **Educational Purposes Only**: This codebase is intended for learning and experimentation.
- **Work In Progress (WIP)**: The project is actively being developed.
- **Not Fully Tested**: While unit tests exist for core components, the system as a whole has not undergone rigorous production-level testing.

## Tech Stack
- **Framework**: NestJS
- **ML Backend**: TensorFlow.js (with GPU support detection)
- **Language**: TypeScript
- **Tokenization**: Hugging Face Tokenizers (with fallback regex support)

## Core Capabilities
- Transformer Encoder/Decoder stacks
- Training pipeline with Noam scheduler
- Inference API (including Greedy and Beam search)
- Real-time streaming visualization of decoding process
- Prometheus observability metrics
