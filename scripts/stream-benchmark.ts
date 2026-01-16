import { performance } from 'node:perf_hooks';
import { Test } from '@nestjs/testing';
import { TransformerCoreModule } from '../src/transformer-core/transformer-core.module';
import { TransformerModelService } from '../src/transformer-core/transformer.service';

async function measure(label: string, runs: number, fn: () => Promise<void>) {
  const timings: number[] = [];
  for (let i = 0; i < runs; i += 1) {
    const start = performance.now();
    await fn();
    timings.push(performance.now() - start);
  }
  const average = timings.reduce((sum, value) => sum + value, 0) / timings.length;
  // eslint-disable-next-line no-console
  console.log(`${label} avg (${runs} runs): ${average.toFixed(2)}ms`);
}

async function runStreamBenchmark() {
  const runs = Number(process.env.BENCH_RUNS ?? 5);
  const maxLength = Number(process.env.BENCH_STREAM_MAXLEN ?? 32);
  const beamWidth = Number(process.env.BENCH_STREAM_BEAM ?? 4);
  const throttleMs = Number(process.env.BENCH_STREAM_THROTTLE ?? 250);

  const moduleRef = await Test.createTestingModule({
    imports: [TransformerCoreModule],
  }).compile();

  const transformer = moduleRef.get(TransformerModelService);
  const inputIds = Array(16).fill(1);

  await measure('Greedy stream', runs, () =>
    transformer.streamGreedy(inputIds, maxLength, false, async () => {}),
  );
  await measure('Beam stream (unthrottled)', runs, () =>
    transformer.streamBeam(inputIds, maxLength, beamWidth, 0, false, async () => {}),
  );
  await measure(`Beam stream (${throttleMs}ms throttle)`, runs, () =>
    transformer.streamBeam(inputIds, maxLength, beamWidth, throttleMs, false, async () => {}),
  );

  await moduleRef.close();
}

runStreamBenchmark().catch((error) => {
  // eslint-disable-next-line no-console
  console.error('Stream benchmark failed', error);
  process.exit(1);
});
