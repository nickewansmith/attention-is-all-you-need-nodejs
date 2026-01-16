import { Logger } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bufferLogs: true });
  const corsOrigins = process.env.CORS_ALLOWED_ORIGINS;
  if (corsOrigins) {
    const allowedOrigins = corsOrigins
      .split(',')
      .map((origin) => origin.trim())
      .filter((origin) => origin.length > 0);
    app.enableCors({
      origin: allowedOrigins,
      methods: ['GET', 'POST', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization'],
    });
  }
  const port = process.env.PORT || 3000;
  await app.listen(port);
  Logger.log(`🚀 Transformer API running on http://localhost:${port}`);
}

void bootstrap();
