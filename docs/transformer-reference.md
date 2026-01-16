# Transformer Reference (NestJS Implementation)

This document distills the architectural details from *Attention Is All You Need* (Vaswani et al., 2017) and bridges them to our NestJS/TypeScript stack. Keep it nearby when extending the codebase.

> This reference is part of an educational, still-evolving project—use it to understand the original design before adapting the code for real deployments.

## Core Hyperparameters

| Parameter | Paper Default | Notes |
| --- | --- | --- |
| `numLayers` | 6 encoder / 6 decoder | Configurable per deployment. |
| `dModel` | 512 | Embedding width, must be divisible by `numHeads`. |
| `numHeads` | 8 | Self-attention heads. |
| `dff` | 2048 | Feed-forward inner dimension. |
| `dropoutRate` | 0.1 | Applied after attention + FFN blocks. |
| `maxPositionEncoding` | 5000 | Sinusoidal positional encoding length. |
| `layerNormEpsilon` | 1e-6 | Stabilizes LayerNorm. |
| `vocabSize` | dataset dependent | 37k for WMT'14 En–De in the paper. |

## Data Flow Overview

1. **Token + Positional Embedding** – Input tokens are embedded to `dModel`, scaled by `sqrt(dModel)`, offset by sinusoidal encodings (Section 3.5, Fig. 2 of the paper).
2. **Encoder Stack** (Fig. 1 left)
   - Multi-head self-attention with padding mask.
   - Add & LayerNorm.
   - Position-wise FFN (`ReLU(W1x + b1)W2 + b2`).
   - Dropout after attention + FFN residual adds.
3. **Decoder Stack** (Fig. 1 right)
   - Masked multi-head self-attention (look-ahead + padding mask).
   - Encoder–decoder multi-head attention.
   - FFN block mirroring the encoder.
4. **Final Linear + Softmax** – Project decoder output to `vocabSize`, apply softmax for token probabilities (Section 3.4).
5. **Training Schedule** – Noam LR schedule: `lr = dModel^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})` with Adam (β1=0.9, β2=0.98, ε=1e-9).

## NestJS Mapping

- `TransformerCoreModule` provides the math blocks: embeddings, positional encodings, attention, FFN, encoder/decoder layers, and the assembled transformer model service.
- `TensorModule` wraps `@tensorflow/tfjs-node` operations with a DI-friendly service so we can swap runtimes (e.g., ONNX Runtime) without touching higher layers.
- `TrainingModule` orchestrates dataset ingestion, batching/masking, Noam LR scheduler, and checkpointing via injectable services.
- `ApiModule` exposes HTTP/WebSocket inference endpoints, loads checkpoints, and surfaces attention maps for visualization.
- `ObservabilityModule` provides health checks and Prometheus counters for inference/training throughput.

## Dataset + Tokenizer Pipeline

- **DatasetService** streams aligned sentence pairs from configurable files (`DATASET_SOURCE_PATH`, `DATASET_TARGET_PATH`) and optionally caps them via `DATASET_MAX_SAMPLES`. When no files are provided it falls back to the baked-in toy corpus.
- **TokenizerService** loads Hugging Face `tokenizer.json`/`tokenizer_config.json` files via `@huggingface/tokenizers`, respecting special tokens and max-length constraints. If no external tokenizer is configured, it builds a whitespace/BPE-lite vocab on the fly.
- Special token IDs can be overridden through env vars (`TOKENIZER_START_TOKEN`, `TOKENIZER_START_ID`, etc.) so checkpoints remain compatible with upstream research artifacts.
- Positional encoding length (`TRANSFORMER_MAX_POSITION`) controls both the sine/cosine lookup table and the padding length for dataset batches; keep it in sync with your tokenizer's max input length.

## Reference Material

- Original paper: `docs/attention-is-all-you-need.pdf`
- Pytorch reference implementations:  
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch  
  - https://github.com/bkhanal-11/transformers  
  - https://github.com/brandokoch/attention-is-all-you-need-paper
- Illustrated Transformer walkthrough: https://jalammar.github.io/illustrated-transformer/

## Glossary

- **Padding mask** – Masks out padded token positions during attention.
- **Look-ahead mask** – Prevents positions from attending to subsequent tokens in the decoder.
- **Beam search** – Keeps top `k` hypotheses during auto-regressive decoding.

## Next Steps Checklist

- [x] Hook dataset/tokenizer implementations into `TrainingModule`.
- [x] Implement checkpoint serialization (Transformer weights + optimizer state saved in JSON payloads).
- [ ] Add evaluation suite (BLEU, perplexity).
- [ ] Provide sample scripts (`npm run train:iwslt`, `npm run infer`).
