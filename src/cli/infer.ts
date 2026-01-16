import { NestFactory } from '@nestjs/core';
import { AppModule } from '../app.module';
import { ApiService } from '../api/api.service';
import { TranslateRequestDto } from '../api/dto/translate.dto';

function parseArgs(): TranslateRequestDto {
  const [, , text, ...rest] = process.argv;
  if (!text) {
    throw new Error('Usage: npm run cli:infer -- "your text" [maxLength] [strategy]');
  }

  const dto: TranslateRequestDto = { text } as TranslateRequestDto;
  if (rest[0]) {
    dto.maxLength = Number(rest[0]);
  }
  if (rest[1]) {
    dto.strategy = rest[1] as 'greedy' | 'beam';
  }
  if (rest[2]) {
    dto.beamWidth = Number(rest[2]);
  }
  dto.includeAttention = true;
  return dto;
}

async function bootstrap() {
  const dto = parseArgs();
  const app = await NestFactory.createApplicationContext(AppModule, { logger: false });
  const apiService = app.get(ApiService);
  const response = apiService.translate(dto);
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(response, null, 2));
  await app.close();
}

bootstrap().catch((error) => {
  // eslint-disable-next-line no-console
  console.error('Inference CLI failed', error);
  process.exit(1);
});
