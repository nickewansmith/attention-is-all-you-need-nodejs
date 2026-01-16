import { performance } from 'node:perf_hooks';
import { Test } from '@nestjs/testing';
import { TransformerCoreModule } from '../src/transformer-core/transformer-core.module';
import { TransformerModelService } from '../src/transformer-core/transformer.service';

async function runBenchmark() {
  process.env.TRANSFORMER_LAYERS = process.env.TRANSFORMER_LAYERS ?? '2';
  process.env.TRANSFORMER_D_MODEL = process.env.TRANSFORMER_D_MODEL ?? '128';
  process.env.TRANSFORMER_HEADS = process.env.TRANSFORMER_HEADS ?? '4';
  process.env.TRANSFORMER_DFF = process.env.TRANSFORMER_DFF ?? '256';
  process.env.TRANSFORMER_MAX_POSITION = process.env.TRANSFORMER_MAX_POSITION ?? '32';

  const moduleRef = await Test.createTestingModule({
    imports: [TransformerCoreModule],
  }).compile();

  const transformer = moduleRef.get(TransformerModelService);
  const input = [Array(16).fill(1)];
  const decoder = [Array(16).fill(1)];

  const warmup = transformer.forward(input, decoder, false);
  warmup.dispose();

  const runs = Number(process.env.BENCH_RUNS ?? 5);
  const timings: number[] = [];
  for (let i = 0; i < runs; i += 1) {
    const start = performance.now();
    const logits = transformer.forward(input, decoder, false);
    logits.dispose();
    timings.push(performance.now() - start);
  }

  const avg = timings.reduce((sum, value) => sum + value, 0) / timings.length;
  // eslint-disable-next-line no-console
  console.log(`Forward pass average over ${runs} runs: ${avg.toFixed(2)}ms`);

  await moduleRef.close();
}

runBenchmark().catch((error) => {
  // eslint-disable-next-line no-console
  console.error('Benchmark failed', error);
  process.exit(1);
});
