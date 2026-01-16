import { ConfigModule } from '@nestjs/config';
import { Test } from '@nestjs/testing';
import { TransformerConfigService } from '../src/transformer-core/transformer-config.service';
import { TransformerCoreModule } from '../src/transformer-core/transformer-core.module';
import { TransformerModelService } from '../src/transformer-core/transformer.service';
import { TensorService } from '../src/tensor/tensor.service';

jest.setTimeout(30000);

describe('TransformerModelService', () => {
  beforeAll(() => {
    process.env.TRANSFORMER_LAYERS = '2';
    process.env.TRANSFORMER_D_MODEL = '64';
    process.env.TRANSFORMER_HEADS = '4';
    process.env.TRANSFORMER_DFF = '128';
    process.env.TRANSFORMER_MAX_POSITION = '32';
    process.env.TRANSFORMER_INPUT_VOCAB = '128';
    process.env.TRANSFORMER_TARGET_VOCAB = '64';
  });

  it('returns logits with expected dimensions', async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [ConfigModule.forRoot({ isGlobal: true, ignoreEnvFile: true }), TransformerCoreModule],
    }).compile();

    const service = moduleRef.get(TransformerModelService);
    const config = moduleRef.get(TransformerConfigService).getConfig();

    const encoderInput = [[1, 2, 3, 0, 0]];
    const decoderInput = [[1, 4, 5, 0, 0]];
    const logits = service.forward(encoderInput, decoderInput, false);

    expect(logits.shape[0]).toBe(1);
    expect(logits.shape[1]).toBe(decoderInput[0].length);
    expect(logits.shape[2]).toBe(config.targetVocabSize);

    logits.dispose();
    await moduleRef.close();
  });

  it('exposes attention array for beam decoding', async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [ConfigModule.forRoot({ isGlobal: true, ignoreEnvFile: true }), TransformerCoreModule],
    }).compile();

    const service = moduleRef.get(TransformerModelService);
    const result = service.decodeSequences([[1, 2, 3, 0]], {
      maxLength: 5,
      strategy: 'beam',
      beamWidth: 2,
      includeAttention: true,
    });

    expect(result.sequences.length).toBe(1);
    expect(result.attention).toBeDefined();
    expect(result.attention?.length).toBe(1);

    await moduleRef.close();
  });

  it('builds combined masks with correct shape', async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [ConfigModule.forRoot({ isGlobal: true, ignoreEnvFile: true }), TransformerCoreModule],
    }).compile();

    const service = moduleRef.get(TransformerModelService) as any;
    const tensorService = moduleRef.get(TensorService);
    const seq = tensorService.tensor2d([[1, 2, 0, 0]], undefined, 'int32');
    const mask = service.createCombinedMask(seq);
    expect(mask.shape).toEqual([1, 1, 4, 4]);
    mask.dispose();
    seq.dispose();
    await moduleRef.close();
  });
});
