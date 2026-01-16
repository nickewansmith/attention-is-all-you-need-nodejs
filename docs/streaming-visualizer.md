# Streaming Visualizer

A lightweight UI for observing the greedy and beam decoding loops exposed by `/transformer/translate/stream`. Use it whenever you need to debug convergence, demonstrate beam search behavior to stakeholders, or capture attention tensors for notebooks.

## Launching the tool

1. Run the Nest server (`npm run start:dev`).
2. Open `http://localhost:3000/visualizer` in a modern browser.
3. Fill out the form:
   - **Strategy** – `greedy` streams one token at a time; `beam` emits the top-`k` partial sequences every decoding step.
   - **Beam width** – only relevant for beam search; defaults to 4.
   - **Max length** – caps decoding to prevent runaway generations.
   - **Beam throttle (ms)** – optional cadence limiter that passes `beamThrottleMs` to the API so beam snapshots are sent at most once per interval; set `0` for full fidelity.
   - **Include attention** – greedy streaming can attach encoder/decoder snapshots to the `completed` event so you can paste them into `notebooks/attention-visualization.ipynb`.
   - **API base URL** – leave blank when the UI is hosted by the same Nest server. Populate this when the API sits behind a remote host or an SSH tunnel.
4. Press **Start stream**.

## Event anatomy

The visualizer consumes the same JSON payloads exposed by the SSE endpoint:

- `token` – `{ step, tokenId, token, partialTranslation }`. Rendered in the *Greedy token stream* log.
- `beam` – `{ rank, score, tokens, partialTranslation }`. Each event updates the table and historical log so you can watch how hypotheses reshuffle as scores change.
- `completed` – `{ translation, tokens, attention? }`. Terminates the stream, populates the final translation block, and dumps the attention map when available (now supported for beam streaming as well when `includeAttention=true`).

## Beam visualization & throttling

- The **Beam candidates** card now draws a sparkline for each rank (color-coded legend included) so you can see how log-probability scores evolve over time. The line chart consumes the very same SSE payloads, so any throttle applied server-side is reflected immediately.
- Combine the per-rank table, the sparkline, and the textual history to reason about why a hypothesis changed rank—if you see a dip in the chart and a new token in the log, you know the divergence step instantly.
- When `beamThrottleMs` is non-zero, the backend coalesces intermediate decoding steps and only emits after the interval elapses; the UI mirrors that cadence so you can tune the knob to your latency vs. fidelity needs.

## Standalone hosting & security

1. Build/deploy the static assets:
   - Local demo: `npx http-server docs/streaming-visualizer -p 4173` and browse to `http://localhost:4173`.
   - Production: copy the folder to your CDN/static bucket and serve it behind your existing TLS edge.
2. Configure Nest to accept cross-origin SSE connections by setting `CORS_ALLOWED_ORIGINS` to the origin that will host the UI (comma-separated list supported) before running `npm run start:dev` or Docker.
3. Add authentication the same way you protect the REST API:
   - Use a Nest `AuthGuard`/interceptor that requires API keys, JWTs, or SSO headers on `/transformer/translate/stream`.
   - Alternatively, terminate TLS + auth at a reverse proxy (NGINX, Traefik, API Gateway) and only expose the SSE endpoint internally.
4. In the visualizer, set **API base URL** to the public endpoint (e.g., `https://api.example.com`) so all EventSource calls include your auth headers/cookies.

These steps let you run polished demos without exposing checkpoints/tokenizers to the open internet.

## Tips

- Combine the visualizer with the Prometheus metrics to correlate throughput spikes with beam width changes (the UI increments `/health/metrics` counters just like REST requests).
- The log panes append raw strings intentionally—copy/paste them into bug reports or notebooks without extra formatting.
- Use the **Stop** button to abort long generations; it simply closes the underlying `EventSource`.
- When experimenting with alternate datasets/tokenizers, toggle *Include attention* and compare head activations between runs to ensure your preprocessing pipeline still steers the decoder toward relevant encoder positions.

## Extending the UI

Because the page lives under `docs/streaming-visualizer`, you can fork it without touching the API:

- Extend the built-in sparkline (e.g., overlap target BLEU curves or annotate divergence points) by tapping into the helper functions inside `app.js`.
- Swap the vanilla CSS for your design system and ship a polished demo to product teams.
- Hook up `fetch` calls to persist interesting runs (translation text + beam scores) into your datastore for offline analysis.
