import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { TransformerConfigService } from '../transformer-core/transformer-config.service';

@Injectable()
export class NoamSchedulerService {
  private readonly warmupSteps: number;

  constructor(private readonly configService: ConfigService, private readonly transformerConfig: TransformerConfigService) {
    this.warmupSteps = this.configService.get<number>('TRANSFORMER_WARMUP_STEPS') ?? 4000;
  }

  getLearningRate(step: number) {
    const { dModel } = this.transformerConfig.getConfig();
    const stepFloat = step;
    const arg1 = Math.pow(stepFloat, -0.5);
    const arg2 = stepFloat * Math.pow(this.warmupSteps, -1.5);
    return Math.pow(dModel, -0.5) * Math.min(arg1, arg2);
  }
}
