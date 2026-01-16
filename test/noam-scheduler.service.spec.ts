import { ConfigService } from '@nestjs/config';
import { NoamSchedulerService } from '../src/training/noam-scheduler.service';
import type { TransformerConfig } from '../src/transformer-core/types/transformer-config';
import type { TransformerConfigService } from '../src/transformer-core/transformer-config.service';

describe('NoamSchedulerService', () => {
  it('decays learning rate after warmup', () => {
    const config: TransformerConfig = {
      numLayers: 2,
      dModel: 128,
      numHeads: 4,
      dff: 256,
      inputVocabSize: 100,
      targetVocabSize: 100,
      maximumPositionEncoding: 32,
      dropoutRate: 0.1,
      layerNormEpsilon: 1e-6,
    };
    const transformerConfig = { getConfig: () => config } as TransformerConfigService;
    const scheduler = new NoamSchedulerService(new ConfigService(), transformerConfig);

    const lrWarmup = scheduler.getLearningRate(4000);
    const lrPostWarmup = scheduler.getLearningRate(8000);
    expect(lrWarmup).toBeGreaterThan(lrPostWarmup);
    expect(lrWarmup).toBeGreaterThan(0);
    expect(lrPostWarmup).toBeGreaterThan(0);
  });
});
