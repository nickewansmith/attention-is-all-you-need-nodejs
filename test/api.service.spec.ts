import { Observable } from 'rxjs';
import type { MessageEvent } from '@nestjs/common';
import { ApiService } from '../src/api/api.service';
import { TransformerModelService } from '../src/transformer-core/transformer.service';
import { TokenizerService } from '../src/common/tokenizer/tokenizer.service';
import { TransformerConfigService } from '../src/transformer-core/transformer-config.service';
import { TrainingService } from '../src/training/training.service';
import { MetricsService } from '../src/observability/metrics.service';
import type { TranslateStreamRequestDto } from '../src/api/dto/translate.dto';

describe('ApiService translateStream', () => {
  let service: ApiService;
  let transformer: jest.Mocked<TransformerModelService>;
  let tokenizer: jest.Mocked<TokenizerService>;
  let configService: jest.Mocked<TransformerConfigService>;
  let trainingService: jest.Mocked<TrainingService>;
  let metricsService: jest.Mocked<MetricsService>;

  beforeEach(() => {
    transformer = {
      decodeSequences: jest.fn(),
      streamGreedy: jest.fn(),
      streamBeam: jest.fn(),
    } as unknown as jest.Mocked<TransformerModelService>;

    tokenizer = {
      encode: jest.fn().mockImplementation((text: string) => [text.length, 2, 3]),
      decode: jest.fn().mockImplementation((tokens: number[]) => tokens.join('-')),
      tokenFromId: jest.fn(),
    } as unknown as jest.Mocked<TokenizerService>;

    configService = {
      getConfig: jest.fn().mockReturnValue({ maximumPositionEncoding: 64 }),
    } as unknown as jest.Mocked<TransformerConfigService>;

    trainingService = {
      trainToyEpoch: jest.fn(),
    } as unknown as jest.Mocked<TrainingService>;

    metricsService = {
      markInferenceRequest: jest.fn(),
      markTrainingStep: jest.fn(),
      startStreamTimer: jest.fn().mockReturnValue(jest.fn()),
    } as unknown as jest.Mocked<MetricsService>;

    service = new ApiService(transformer, tokenizer, configService, trainingService, metricsService);
  });

  async function collectEvents(observable: Observable<MessageEvent>) {
    return new Promise<MessageEvent[]>((resolve, reject) => {
      const events: MessageEvent[] = [];
      observable.subscribe({
        next: (event) => events.push(event),
        error: reject,
        complete: () => resolve(events),
      });
    });
  }

  it('streams greedy tokens and final attention payload', async () => {
    transformer.streamGreedy.mockImplementation(async (_encoderInput, _maxLen, _includeAttention, onToken) => {
      await onToken({ step: 0, tokenId: 7, token: 'att', partialTranslation: 'att' });
      return {
        sequence: [1, 2, 3],
        attention: { encoder: [], decoderSelf: [], decoderCross: [] },
      };
    });

    const dto: TranslateStreamRequestDto = { text: 'hello', strategy: 'greedy', includeAttention: true, maxLength: 32 };
    const events = await collectEvents(service.translateStream(dto));

    expect(transformer.streamGreedy).toHaveBeenCalledWith(expect.anything(), 32, true, expect.any(Function));
    expect(events[0].data).toMatchObject({ type: 'token', tokenId: 7 });
    expect(events[1].data).toMatchObject({ type: 'completed', attention: { encoder: [] } });
    expect(metricsService.markInferenceRequest).toHaveBeenCalledTimes(1);
  });

  it('streams beam updates with throttle and returns final attention', async () => {
    transformer.streamBeam.mockImplementation(async (_encoderInput, _maxLen, _beamWidth, _throttle, _includeAtt, onUpdate) => {
      await onUpdate({ rank: 1, score: -0.1, tokens: [1, 2], partialTranslation: 'partial' });
      return {
        sequence: [1, 2, 3, 4],
        attention: { encoder: [], decoderSelf: [], decoderCross: [] },
      };
    });

    const dto: TranslateStreamRequestDto = {
      text: 'beam me',
      strategy: 'beam',
      beamWidth: 5,
      beamThrottleMs: 250,
      includeAttention: true,
    };

    const events = await collectEvents(service.translateStream(dto));

    expect(transformer.streamBeam).toHaveBeenCalledWith(
      expect.anything(),
      64,
      5,
      250,
      true,
      expect.any(Function),
    );
    expect(events[0].data).toMatchObject({ type: 'beam', rank: 1, partialTranslation: 'partial' });
    const completed = events.find((event) => (event.data as any).type === 'completed');
    expect(completed?.data).toMatchObject({ attention: { encoder: [] } });
    expect(metricsService.markInferenceRequest).toHaveBeenCalledTimes(1);
  });

  it('splits beam batches when maxLength differs yet still trims tokens/attention', () => {
    tokenizer.encode.mockClear();
    const attentionSnapshot = {
      layer: 0,
      heads: [
        [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
      ],
    };
    transformer.decodeSequences
      .mockReturnValueOnce({
        sequences: [[1, 2]],
        attention: [
          {
            encoder: [{ layer: 0, heads: [[[0.1]]] }],
            decoderSelf: [attentionSnapshot],
            decoderCross: [attentionSnapshot],
          },
        ],
      })
      .mockReturnValueOnce({
        sequences: [[5, 6, 7, 8]],
        attention: [null],
      });

    const response = service.translateBatch({
      items: [
        { text: 'foo', strategy: 'beam', beamWidth: 2, includeAttention: true, maxLength: 2 },
        { text: 'bar', strategy: 'beam', beamWidth: 2, maxLength: 4 },
      ],
    });

    expect(transformer.decodeSequences).toHaveBeenCalledTimes(2);
    expect(response.results[0].tokens).toEqual([1, 2]);
    const trimmedSelf = response.results[0].attention?.decoderSelf[0].heads[0];
    expect(trimmedSelf?.length).toBe(2);
    expect(response.results[1].tokens).toEqual([5, 6, 7, 8]);
    const encodeCalls = tokenizer.encode.mock.calls;
    expect(encodeCalls[0][1]).toBe(2);
    expect(encodeCalls[1][1]).toBe(4);
  });

  it('reuses beam batches when maxLength matches even if attention flags differ', () => {
    transformer.decodeSequences.mockReturnValue({
      sequences: [
        [1, 2, 9],
        [3, 4, 9],
      ],
      attention: [
        { encoder: [], decoderSelf: [], decoderCross: [] },
        { encoder: [], decoderSelf: [], decoderCross: [] },
      ],
    });

    const response = service.translateBatch({
      items: [
        { text: 'foo', strategy: 'beam', beamWidth: 2, includeAttention: true, maxLength: 4 },
        { text: 'bar', strategy: 'beam', beamWidth: 2, maxLength: 4 },
      ],
    });

    expect(transformer.decodeSequences).toHaveBeenCalledTimes(1);
    expect(response.results[0].attention).toBeDefined();
    expect(response.results[1].attention).toBeUndefined();
  });
});
