# Attention Is All You Need — NestJS Edition

A NestJS/TensorFlow.js implementation of the Transformer architecture described in *Attention Is All You Need*. The project aims to be a serious starting point for Node.js researchers building custom translation or sequence models with modern tooling (checkpoints, observability, CLI, Docker, CI, etc.).

> **IMPORTANT**: This repository is for **educational purposes only**. It is a **work in progress (WIP)** and has **not been fully tested**. Use it to learn about the *Attention Is All You Need* design and adapt it cautiously for production workloads.

## Project Structure

- `src/transformer-core` – Positional encoding, multi-head attention, feed-forward blocks, encoder/decoder stacks, checkpoint-aware transformer service.
- `src/training` – Dataset ingestion, learning rate scheduler, training service with BLEU/perplexity evaluation, optimizer state serialization.
- `src/api` – REST layer with translation + training endpoints and DTO validation.
- `src/common/tokenizer` – Hugging Face tokenizer loader (falls back to regex tokenizer) with configurable special tokens.
- `src/tensor` – TensorFlow backend abstraction that auto-detects GPU/CPU and exposes helper ops.
- `src/observability` – Health and Prometheus metrics endpoints.
- `src/cli` – `train` and `infer` commands for scripting.
- `scripts` – Benchmark harness, backend detection utilities, etc.
- `docs/transformer-reference.md` – Architecture cheat sheet derived from the paper.

## Architecture Walkthrough

1. **Request enters API layer** – `ApiController`/`ApiService` validate DTOs and call the `TransformerModelService` inside `TransformerCoreModule`.
2. **Tokenizer** – `TokenizerService` encodes text using either a Hugging Face `tokenizer.json` (for BPE/SentencePiece) or a fallback regex tokenizer. Special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`) are configurable via env vars.
3. **Tensor backend** – `TensorService` lazily loads `@tensorflow/tfjs-node-gpu` when CUDA is present (falls back to CPU). All tensor ops funnel through this service so you can instrument or swap backends later.
4. **Transformer core** – The Nest service wires positional encodings, multi-head attention, residual connections, FFNs, layer norms, and exposes utility methods:
   - `forward` for full encoder-decoder passes
   - `decodeSequences` for greedy/beam decoding with optional attention capture
   - checkpoint serialization/deserialization for warm starts
5. **Training pipeline** – `TrainingService` pulls aligned batches via `DatasetService`, applies the Noam scheduler (`NoamSchedulerService`), logs BLEU/perplexity samples, increments Prometheus counters, and persists checkpoints after each epoch.
6. **Observability** – `MetricsService` hosts Prometheus counters; `HealthController` exposes `/health` and `/health/metrics`. Logs are handled by Nest’s logger, but you can integrate Winston/Pino easily.

Keep this reference handy when diving into unfamiliar modules.

## Additional Resources

- `notebooks/attention-visualization.ipynb` – Jupyter notebook that fetches attention maps from the API and renders encoder/decoder head heatmaps.
- `docs/dataset-preparation.md` – step-by-step guide for preparing real translation datasets and tokenizers (SentencePiece/HF JSON) for this project.
- `docs/streaming-visualizer.md` – walkthrough of the built-in UI that consumes the streaming endpoint for greedy and beam decoding.

## Getting Started

```bash
npm install
npm run build
npm run start:dev
```

### Translation API


**cURL**

```
curl -X POST http://localhost:3000/transformer/translate \\
  -H 'Content-Type: application/json' \\
  -d '{
    "text": "attention is all you need",
    "maxLength": 32,
    "strategy": "beam",
    "beamWidth": 4,
    "includeAttention": true
  }'
```

- `strategy` supports `greedy` (default) and `beam`.
- Attention maps are returned per input sentence (`attention: [AttentionMaps|null,...]`). When attention is omitted, the property is absent. The HTTP response still contains only the first sentence’s translation since the endpoint accepts a single string.
- Attention maps are returned per input sentence (`attention: [AttentionMaps|null,...]`). When attention is omitted, the property is absent. The HTTP response still contains only the first sentence’s translation since the endpoint accepts a single string.

### Batch Translation API

**cURL**

```
curl -X POST http://localhost:3000/transformer/translate/batch \\
  -H 'Content-Type: application/json' \\
  -d '{
    "items": [
      { "text": "hello world", "strategy": "greedy", "includeAttention": true },
      { "text": "attention is all you need", "strategy": "beam", "beamWidth": 4 }
    ]
  }'
```

- Each sentence can specify its own `maxLength`, `strategy`, `beamWidth`, and `includeAttention` flag.
- The response looks like `{ "results": [ { translation, tokens, attention }, ... ] }` with `attention` present only for items that requested it.
- Sentences sharing `maxLength` values are decoded together for efficiency, while mixed `maxLength` requests fall into separate groups so each search obeys its own cap (no late-stage trimming surprises).

### Training API

```
POST http://localhost:3000/transformer/train/toy
{
  "epochs": 1,
  "batchSize": 2
}
```

For real datasets use the configuration options below.

## Datasets & Tokenizers

1. Create aligned source/target files (one sentence per line). Example: `data/train.en`, `data/train.de`.
2. Configure paths in `.env`:
   ```env
   DATASET_SOURCE_PATH=./data/train.en
   DATASET_TARGET_PATH=./data/train.de
   DATASET_MAX_SAMPLES=50000 # optional cap
   ```
3. Provide a tokenizer (recommended):
   ```env
   TOKENIZER_JSON_PATH=./tokenizer/tokenizer.json
   TOKENIZER_CONFIG_PATH=./tokenizer/tokenizer_config.json
   TOKENIZER_START_TOKEN=<s>
   TOKENIZER_END_TOKEN=</s>
   TOKENIZER_PAD_TOKEN=<pad>
   TOKENIZER_UNK_TOKEN=<unk>
   TOKENIZER_MAX_LENGTH=128
   ```
   Special token IDs (`TOKENIZER_*_ID`) can override defaults if your tokenizer uses different IDs.

If no tokenizer files are provided the project uses a basic regex tokenizer and builds a vocab from the dataset. Attention and decoding still work but translations won’t match modern corpora.

## Checkpoints

- `CHECKPOINT_SAVE_PATH` (default `checkpoints/latest.json`): transformer weights + optimizer state saved after each epoch.
- `CHECKPOINT_LOAD_PATH`: optional path to warm-start both the HTTP server and CLI tooling.
- REST training and `npm run cli:train` automatically restore from `CHECKPOINT_LOAD_PATH` when present.

- `GET /transformer/translate/stream` (Server-Sent Events) supports greedy and beam decoding:

```
curl -N "http://localhost:3000/transformer/translate/stream?text=attention%20is%20all%20you%20need&maxLength=32&strategy=beam&beamWidth=4"
```

- Greedy mode emits `token` events (token id/text + partial translation) followed by a `completed` event with the final string (and optional attention map).
- Beam mode emits `beam` updates (rank, score, partial tokens/translations) for the top `k` hypotheses, then a `completed` event once the best beam finalizes (now including attention maps when `includeAttention=true`).
- Add `beamThrottleMs` (0-5000) to the query string when you want the server to coalesce intermediate beam steps and emit updates no more frequently than every _n_ milliseconds.

### Streaming Visualizer UI

- Start the Nest server (`npm run start:dev`) and visit `http://localhost:3000/visualizer`.
- Enter a sentence, choose greedy or beam, and click **Start stream** to watch tokens or beam hypotheses arrive live. The UI uses the SSE endpoint above under the hood.
- The beam panel now highlights score evolution through a color-coded sparkline + legend, a live table of the top `k` candidates, and the historical log so you can correlate rank swaps with emitted tokens.
- The greedy log flushes every emitted token with text + vocab id, making it easier to debug stuck decoders or length issues.
- Attention dumps (when requested) are printed alongside the final translation so you can copy them directly into the included Jupyter notebook—even for beam streaming now that attention capture is wired up end-to-end.
- Tweak **Beam throttle (ms)** to match the new `beamThrottleMs` query parameter—set it to `0` for full fidelity or something like `250` to significantly lighten SSE traffic.
- Use the “API base URL” field when the visualizer is proxied through another domain/port; otherwise it automatically targets the server that hosts the page.

#### Visualizer Deployment & Security

- Serve `docs/streaming-visualizer` from any static host (`npx http-server docs/streaming-visualizer -p 4173` or drop it into your CDN) and point the **API base URL** field at the Nest server.
- When hosting the UI off-box, set `CORS_ALLOWED_ORIGINS=https://your-visualizer.example.com` before starting Nest so the SSE endpoint accepts the cross-origin connection.
- Protect `/transformer/translate/stream` the same way you guard other APIs—e.g., wrap it in an auth guard or front it with a reverse proxy that injects API keys—then let the visualizer send the appropriate headers.
- For production demos, consider routing the visualizer through a separate domain/subpath and locking both the static host and SSE endpoint behind SSO to keep checkpoints/tokenizers private.

## CLI Utilities

- `npm run cli:train [epochs] [batchSize]`
- `npm run cli:infer "your sentence" [maxLength] [strategy] [beamWidth]`

The CLI is useful for quick experiments or automation (no HTTP server required).

*Training*

```
npm run cli:train 2 4   # epochs=2, batchSize=4
```

This command spins up the Nest application context, trains on the configured dataset (loading checkpoints automatically), and shuts down when training completes.

*Inference*

```
npm run cli:infer "attention is all you need" 32 beam 4
```

- Positional arguments: `<text>` `[maxLength]` `[strategy]` `[beamWidth]`
- Outputs JSON with translation, tokens, and attention maps (always enabled for CLI inference so you can visualize heads quickly).
- Honors the same environment variables as the HTTP server, including tokenizer/dataset/checkpoint settings.

## Observability

- `GET /health` — liveness payload
- `GET /health/metrics` — Prometheus metrics including `inference_requests_total` and `training_steps_total`

`/health/metrics` exposes Prometheus counters:

| Metric | Description |
| --- | --- |
| `inference_requests_total` | Incremented per sentence processed (REST + CLI). |
| `training_steps_total` | Incremented per optimizer update during training. |
| `translation_stream_duration_ms` | Streaming latency histogram labeled by strategy (`greedy`/`beam`) and throttle state. |

You can scrape the endpoint directly:

```
curl http://localhost:3000/health
curl http://localhost:3000/health/metrics
```

Use these metrics to wire Grafana dashboards or alert on throughput/regressions.

## Docker

```
docker build -t nest-transformer .
docker run -p 3000:3000 nest-transformer
```

## Continuous Integration

`.github/workflows/ci.yml` runs lint + unit tests on pushes/PRs to `main`.

## Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `TRANSFORMER_LAYERS` | 4 | Encoder/decoder layer count. |
| `TRANSFORMER_D_MODEL` | 256 | Model dimension (`d_model`). |
| `TRANSFORMER_HEADS` | 8 | Attention heads. |
| `TRANSFORMER_DFF` | 1024 | Feed-forward inner dimension. |
| `TRANSFORMER_MAX_POSITION` | 256 | Positional encoding length & max tokens. |
| `TRANSFORMER_INPUT_VOCAB` | 4096 | Input vocab fallback when no external tokenizer. |
| `TRANSFORMER_TARGET_VOCAB` | 4096 | Target vocab fallback. |
| `CORS_ALLOWED_ORIGINS` | _unset_ | Optional comma-separated list of origins allowed to hit the API/streaming endpoints (useful when hosting the visualizer elsewhere). |
| `TRANSFORMER_WARMUP_STEPS` | 4000 | Noam warmup steps. |
| `DATASET_SOURCE_PATH` | — | Path to source-language file. |
| `DATASET_TARGET_PATH` | — | Path to target-language file. |
| `DATASET_MAX_SAMPLES` | — | Optional sample cap. |
| `TOKENIZER_JSON_PATH` | — | Hugging Face tokenizer JSON. |
| `TOKENIZER_CONFIG_PATH` | sibling `tokenizer_config.json` | Optional config file. |
| `TOKENIZER_MAX_LENGTH` | 64 | Default tokenizer padding/truncation length. |
| `TOKENIZER_*_TOKEN` | `<s>`, `</s>`, `<pad>`, `<unk>` | Special token overrides. |
| `TOKENIZER_*_ID` | — | Special token ID overrides. |
| `CHECKPOINT_SAVE_PATH` | `checkpoints/latest.json` | Checkpoint save path. |
| `CHECKPOINT_LOAD_PATH` | save path | Load path at startup. |
| `TFJS_FORCE_CPU` | `0` | Force CPU backend even if CUDA detected. |

## GPU Backend Detection

`src/tensor/backend-loader.ts` checks for `nvidia-smi` at startup. If CUDA is available, it loads `@tensorflow/tfjs-node-gpu`; otherwise it falls back to `@tensorflow/tfjs-node`. Set `TFJS_FORCE_CPU=1` to override detection.

## Benchmarking

```
npm run benchmark
```

Measures average forward-pass latency (configurable via env vars such as `BENCH_RUNS`).

```
npm run benchmark:stream
```

Benchmarks greedy vs. beam streaming (and beam with throttling). Tweak `BENCH_RUNS`, `BENCH_STREAM_MAXLEN`, `BENCH_STREAM_BEAM`, and `BENCH_STREAM_THROTTLE` to match your deployment profile.

## Testing

```
npm test
```

Covers tokenizer round-trips, scheduler math, transformer tensor shapes, and beam/greedy decode invariants.

## Notes

- Attention capture for greedy decoding runs per sentence to ensure accurate maps.
- Beam decoding now returns per-sentence attention arrays. Expect `null` entries when attention is unavailable.
- GPU builds require CUDA libraries installed (the loader logs a warning and falls back to CPU otherwise).
