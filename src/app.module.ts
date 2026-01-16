import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'node:path';
import { ApiModule } from './api/api.module';
import { TransformerCoreModule } from './transformer-core/transformer-core.module';
import { TrainingModule } from './training/training.module';
import { ObservabilityModule } from './observability/observability.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    ServeStaticModule.forRoot({
      serveRoot: '/visualizer',
      rootPath: join(process.cwd(), 'docs', 'streaming-visualizer'),
    }),
    TransformerCoreModule,
    TrainingModule,
    ApiModule,
    ObservabilityModule,
  ],
})
export class AppModule {}
