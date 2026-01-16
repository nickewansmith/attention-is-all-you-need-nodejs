import { Module } from '@nestjs/common';
import { CommonModule } from '../common/common.module';
import { TransformerCoreModule } from '../transformer-core/transformer-core.module';
import { TrainingModule } from '../training/training.module';
import { ObservabilityModule } from '../observability/observability.module';
import { ApiController } from './api.controller';
import { ApiService } from './api.service';

@Module({
  imports: [TransformerCoreModule, TrainingModule, CommonModule, ObservabilityModule],
  controllers: [ApiController],
  providers: [ApiService],
})
export class ApiModule {}
