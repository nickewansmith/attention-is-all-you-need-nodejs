import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CommonModule } from '../common/common.module';
import { TensorModule } from '../tensor/tensor.module';
import { TransformerCoreModule } from '../transformer-core/transformer-core.module';
import { CheckpointModule } from '../checkpoint/checkpoint.module';
import { ObservabilityModule } from '../observability/observability.module';
import { DatasetService } from './dataset.service';
import { NoamSchedulerService } from './noam-scheduler.service';
import { EvaluationService } from './evaluation.service';
import { TrainingService } from './training.service';

@Module({
  imports: [
    TransformerCoreModule,
    TensorModule,
    CommonModule,
    ConfigModule,
    CheckpointModule,
    ObservabilityModule,
  ],
  providers: [TrainingService, DatasetService, NoamSchedulerService, EvaluationService],
  exports: [TrainingService],
})
export class TrainingModule {}
