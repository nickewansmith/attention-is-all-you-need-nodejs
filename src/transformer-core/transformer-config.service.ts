import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { TokenizerService } from '../common/tokenizer/tokenizer.service';
import { DEFAULT_TRANSFORMER_CONFIG, TransformerConfig } from './types/transformer-config';

@Injectable()
export class TransformerConfigService {
  private readonly config: TransformerConfig;

  constructor(private readonly configService: ConfigService, tokenizer: TokenizerService) {
    this.config = {
      ...DEFAULT_TRANSFORMER_CONFIG,
      numLayers: this.getNumber('TRANSFORMER_LAYERS') ?? DEFAULT_TRANSFORMER_CONFIG.numLayers,
      dModel: this.getNumber('TRANSFORMER_D_MODEL') ?? DEFAULT_TRANSFORMER_CONFIG.dModel,
      numHeads: this.getNumber('TRANSFORMER_HEADS') ?? DEFAULT_TRANSFORMER_CONFIG.numHeads,
      dff: this.getNumber('TRANSFORMER_DFF') ?? DEFAULT_TRANSFORMER_CONFIG.dff,
      inputVocabSize: this.getNumber('TRANSFORMER_INPUT_VOCAB') ?? tokenizer.vocabSize,
      targetVocabSize: this.getNumber('TRANSFORMER_TARGET_VOCAB') ?? tokenizer.vocabSize,
      maximumPositionEncoding:
        this.getNumber('TRANSFORMER_MAX_POSITION') ?? DEFAULT_TRANSFORMER_CONFIG.maximumPositionEncoding,
      dropoutRate: this.getNumber('TRANSFORMER_DROPOUT') ?? DEFAULT_TRANSFORMER_CONFIG.dropoutRate,
      layerNormEpsilon:
        this.getNumber('TRANSFORMER_LAYER_NORM_EPS') ?? DEFAULT_TRANSFORMER_CONFIG.layerNormEpsilon,
    };
  }

  getConfig(): TransformerConfig {
    return this.config;
  }

  private getNumber(key: string) {
    const value = this.configService.get<string>(key);
    if (value === undefined || value === null) {
      return undefined;
    }

    const parsed = Number(value);
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid numeric configuration for ${key}: ${value}`);
    }

    return parsed;
  }
}
