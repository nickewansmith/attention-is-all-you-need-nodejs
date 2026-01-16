import { NestFactory } from '@nestjs/core';
import { AppModule } from '../app.module';
import { TrainingService } from '../training/training.service';

async function bootstrap() {
  const app = await NestFactory.createApplicationContext(AppModule, { logger: ['log', 'error', 'warn'] });
  const trainingService = app.get(TrainingService);
  const epochs = parseInt(process.argv[2] ?? '1', 10);
  const batchSize = parseInt(process.argv[3] ?? '2', 10);
  await trainingService.trainToyEpoch(epochs, batchSize);
  await app.close();
}

bootstrap().catch((error) => {
  // eslint-disable-next-line no-console
  console.error('Training CLI failed', error);
  process.exit(1);
});
