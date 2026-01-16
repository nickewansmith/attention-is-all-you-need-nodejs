import { BadRequestException, Injectable, MessageEvent } from '@nestjs/common';
import { Observable } from 'rxjs';
import { TokenizerService } from '../common/tokenizer/tokenizer.service';
import { TransformerConfigService } from '../transformer-core/transformer-config.service';
import { TransformerModelService, type AttentionMaps } from '../transformer-core/transformer.service';
import { TrainingService } from '../training/training.service';
import { MetricsService } from '../observability/metrics.service';
import { TrainToyDto } from './dto/train-toy.dto';
import { TranslateBatchRequestDto } from './dto/translate-batch.dto';
import { TranslateRequestDto, TranslateResponseDto, TranslateStreamRequestDto } from './dto/translate.dto';

@Injectable()
export class ApiService {
  constructor(
    private readonly transformer: TransformerModelService,
    private readonly tokenizer: TokenizerService,
    private readonly configService: TransformerConfigService,
    private readonly trainingService: TrainingService,
    private readonly metricsService: MetricsService,
  ) {}

  translate(dto: TranslateRequestDto): TranslateResponseDto {
    const { maximumPositionEncoding } = this.configService.getConfig();
    const maxLen = Math.min(dto.maxLength ?? maximumPositionEncoding, maximumPositionEncoding);
    const encoderInput = [this.tokenizer.encode(dto.text, maxLen)];
    const decodeResult = this.transformer.decodeSequences(encoderInput, {
      maxLength: maxLen,
      strategy: dto.strategy,
      beamWidth: dto.beamWidth,
      includeAttention: dto.includeAttention,
    });
    const tokens = decodeResult.sequences[0];
    const translation = this.tokenizer.decode(tokens);
    const attention = dto.includeAttention ? decodeResult.attention?.[0] ?? undefined : undefined;
    this.metricsService.markInferenceRequest();
    return { translation, tokens, attention };
  }

  translateBatch(dto: TranslateBatchRequestDto) {
    const { maximumPositionEncoding } = this.configService.getConfig();
    const results: TranslateResponseDto[] = new Array(dto.items.length);

    const groups = new Map<
      string,
      {
        entries: Array<{ index: number; dto: TranslateRequestDto; maxLen: number }>;
        strategy: 'greedy' | 'beam';
        beamWidth?: number;
        includeAttention: boolean;
        maxLength: number;
      }
    >();

    dto.items.forEach((item, index) => {
      const maxLen = Math.min(item.maxLength ?? maximumPositionEncoding, maximumPositionEncoding);
      const strategy = item.strategy ?? 'greedy';
      const includeAttention = item.includeAttention ?? false;
      const beamWidth = strategy === 'beam' ? item.beamWidth ?? null : null;
      const groupKey = JSON.stringify({ strategy, beamWidth, maxLength: maxLen });
      const existing = groups.get(groupKey);
      if (existing) {
        existing.entries.push({ index, dto: item, maxLen });
        existing.includeAttention = existing.includeAttention || includeAttention;
      } else {
        groups.set(groupKey, {
          entries: [{ index, dto: item, maxLen }],
          strategy,
          beamWidth: item.beamWidth,
          includeAttention,
          maxLength: maxLen,
        });
      }
    });

    groups.forEach((group) => {
      const encoderInputs = group.entries.map((entry) => this.tokenizer.encode(entry.dto.text, group.maxLength));
      const decodeResult = this.transformer.decodeSequences(encoderInputs, {
        maxLength: group.maxLength,
        strategy: group.strategy,
        beamWidth: group.beamWidth,
        includeAttention: group.includeAttention,
      });

      group.entries.forEach((entry, idx) => {
        const tokens = decodeResult.sequences[idx] ?? [];
        const trimmedTokens = tokens.length > entry.maxLen ? tokens.slice(0, entry.maxLen) : tokens;
        const translation = this.tokenizer.decode(trimmedTokens);
        const rawAttention = entry.dto.includeAttention ? decodeResult.attention?.[idx] ?? undefined : undefined;
        const attention = rawAttention ? this.trimAttention(rawAttention, entry.maxLen) : undefined;
        results[entry.index] = { translation, tokens: trimmedTokens, attention };
      });
    });

    dto.items.forEach(() => this.metricsService.markInferenceRequest());
    return { results };
  }

  translateStream(dto: TranslateStreamRequestDto): Observable<MessageEvent> {
    const { maximumPositionEncoding } = this.configService.getConfig();
    const maxLen = Math.min(dto.maxLength ?? maximumPositionEncoding, maximumPositionEncoding);
    const encoderInput = this.tokenizer.encode(dto.text, maxLen);

    if (!dto.strategy || dto.strategy === 'greedy') {
      const endTimer = this.metricsService.startStreamTimer('greedy', false);
      return new Observable<MessageEvent>((subscriber) => {
        this.transformer
          .streamGreedy(encoderInput, maxLen, dto.includeAttention ?? false, async (event) => {
            subscriber.next({ data: { type: 'token', ...event } });
          })
          .then((finalResult) => {
            const translation = this.tokenizer.decode(finalResult.sequence);
            subscriber.next({
              data: {
                type: 'completed',
                translation,
                tokens: finalResult.sequence,
                attention: dto.includeAttention ? finalResult.attention : undefined,
              },
            });
            endTimer();
            this.metricsService.markInferenceRequest();
            subscriber.complete();
          })
          .catch((error) => {
            endTimer();
            subscriber.error(error);
          });
      });
    }

    if (dto.strategy === 'beam') {
      const beamWidth = dto.beamWidth ?? 4;
      const beamThrottleMs = dto.beamThrottleMs ?? 0;
      const includeAttention = dto.includeAttention ?? false;
      const endTimer = this.metricsService.startStreamTimer('beam', beamThrottleMs > 0);
      return new Observable<MessageEvent>((subscriber) => {
        this.transformer
          .streamBeam(encoderInput, maxLen, beamWidth, beamThrottleMs, includeAttention, async (event) => {
            subscriber.next({ data: { type: 'beam', ...event } });
          })
          .then((finalResult) => {
            const translation = this.tokenizer.decode(finalResult.sequence);
            subscriber.next({
              data: {
                type: 'completed',
                translation,
                tokens: finalResult.sequence,
                attention: includeAttention ? finalResult.attention : undefined,
              },
            });
            endTimer();
            this.metricsService.markInferenceRequest();
            subscriber.complete();
          })
          .catch((error) => {
            endTimer();
            subscriber.error(error);
          });
      });
    }

    throw new BadRequestException('Unsupported strategy for streaming.');
  }

  async trainToy(dto: TrainToyDto) {
    const epochs = dto.epochs ?? 1;
    const batchSize = dto.batchSize ?? 2;
    await this.trainingService.trainToyEpoch(epochs, batchSize);
  }

  private trimAttention(attention: AttentionMaps, targetLength: number): AttentionMaps {
    const trimSquareHeads = (heads: number[][][]) =>
      heads.map((head) => head.slice(0, targetLength).map((row) => row.slice(0, targetLength)));
    const trimTargetRows = (heads: number[][][]) =>
      heads.map((head) => head.slice(0, targetLength).map((row) => row.slice()));

    return {
      encoder: attention.encoder.map((snapshot) => ({
        layer: snapshot.layer,
        heads: snapshot.heads.map((head) => head.map((row) => row.slice())),
      })),
      decoderSelf: attention.decoderSelf.map((snapshot) => ({
        layer: snapshot.layer,
        heads: trimSquareHeads(snapshot.heads),
      })),
      decoderCross: attention.decoderCross.map((snapshot) => ({
        layer: snapshot.layer,
        heads: trimTargetRows(snapshot.heads),
      })),
    };
  }
}
