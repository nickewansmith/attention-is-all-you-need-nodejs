import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CommonModule } from '../common/common.module';
import { TensorModule } from '../tensor/tensor.module';
import { CheckpointModule } from '../checkpoint/checkpoint.module';
import { FeedForwardNetworkService } from './feed-forward-network.service';
import { LayerNormalizationService } from './layer-normalization.service';
import { MultiHeadAttentionService } from './multi-head-attention.service';
import { PositionalEncodingService } from './positional-encoding.service';
import { TransformerConfigService } from './transformer-config.service';
import { TransformerModelService } from './transformer.service';

@Module({
  imports: [TensorModule, CommonModule, ConfigModule, CheckpointModule],
  providers: [
    PositionalEncodingService,
    LayerNormalizationService,
    MultiHeadAttentionService,
    FeedForwardNetworkService,
    TransformerModelService,
    TransformerConfigService,
  ],
  exports: [TransformerModelService, TransformerConfigService],
})
export class TransformerCoreModule {}
