import { ConfigModule } from '@nestjs/config';
import { Test, TestingModule } from '@nestjs/testing';
import { TransformerCoreModule } from '../src/transformer-core/transformer-core.module';
import { TransformerModelService, StreamBeamUpdate } from '../src/transformer-core/transformer.service';
import { TensorService } from '../src/tensor/tensor.service';

const createTensorMock = () => ({ dispose: jest.fn(), shape: [1, 1] } as any);

describe('TransformerModelService.streamBeam', () => {
  let moduleRef: TestingModule;
  let service: TransformerModelService;
  let tensorService: TensorService;

  beforeAll(() => {
    process.env.TRANSFORMER_LAYERS = '2';
    process.env.TRANSFORMER_D_MODEL = '32';
    process.env.TRANSFORMER_HEADS = '4';
    process.env.TRANSFORMER_DFF = '64';
    process.env.TRANSFORMER_MAX_POSITION = '16';
    process.env.TRANSFORMER_INPUT_VOCAB = '64';
    process.env.TRANSFORMER_TARGET_VOCAB = '64';
  });

  beforeEach(async () => {
    moduleRef = await Test.createTestingModule({
      imports: [ConfigModule.forRoot({ isGlobal: true, ignoreEnvFile: true }), TransformerCoreModule],
    }).compile();

    service = moduleRef.get(TransformerModelService);
    tensorService = moduleRef.get(TensorService);

    // Overwrite tokenizer with deterministic stub so tests do not depend on HF assets.
    (service as any).tokenizer = {
      startTokenId: 0,
      endTokenId: 9,
      decode: (tokens: number[]) => tokens.join('-'),
      tokenFromId: () => '<tok>',
      vocabSize: 10,
    };
  });

  afterEach(async () => {
    jest.restoreAllMocks();
    await moduleRef.close();
  });

  it('captures attention and throttles updates until forced flush', async () => {
    // Stub tensor creations so we do not invoke tfjs kernels.
    jest.spyOn(tensorService, 'tensor2d').mockImplementation(() => createTensorMock());
    jest.spyOn(service as any, 'createPaddingMask').mockReturnValue(createTensorMock());
    jest.spyOn(service as any, 'runEncoder').mockReturnValue({ dispose: jest.fn() } as any);
    jest.spyOn(service as any, 'createCombinedMask').mockReturnValue(createTensorMock());

    const fakeLastLogits = { dispose: jest.fn() } as any;
    jest.spyOn(tensorService, 'logSoftmax').mockReturnValue({
      dataSync: () => new Float32Array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9]),
      dispose: jest.fn(),
    } as any);

    let decodeCall = 0;
    jest.spyOn(service as any, 'runDecoder').mockImplementation(
      (_decoderInput, _encOutput, _combinedMask, _encPaddingMask, _training, selfRecorder?: any, crossRecorder?: any) => {
        const weights = [[[[decodeCall + 1]]]];
        selfRecorder?.(0, weights as any);
        crossRecorder?.(0, weights as any);
        decodeCall += 1;
        return {
          shape: [1, decodeCall, 10],
          slice: () => ({
            reshape: () => fakeLastLogits,
          }),
          dispose: jest.fn(),
        } as any;
      },
    );

    const beamPlan = [5, 6, 9];
    let planIndex = 0;
    jest.spyOn(service as any, 'topK').mockImplementation(() => {
      const idx = beamPlan[Math.min(planIndex, beamPlan.length - 1)];
      planIndex += 1;
      return [{ index: idx, value: -planIndex }];
    });

    const nowSpy = jest.spyOn(Date, 'now');
    nowSpy.mockReturnValueOnce(200);
    nowSpy.mockReturnValueOnce(250);
    nowSpy.mockReturnValue(400);

    const updates: StreamBeamUpdate[] = [];
    const result = await service.streamBeam([1, 2, 3], 6, 1, 100, true, (event) => {
      updates.push(event);
    });

    expect(updates.length).toBe(2);
    expect(updates[0].tokens).toBeDefined();
    expect(result.attention).toBeDefined();
    expect(result.attention?.decoderSelf?.length).toBeGreaterThan(0);
    nowSpy.mockRestore();
  });
});
