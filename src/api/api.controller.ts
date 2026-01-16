import { Body, Controller, HttpCode, HttpStatus, Post, Query, Sse, MessageEvent } from '@nestjs/common';
import type { Observable } from 'rxjs';
import { ApiService } from './api.service';
import { TrainToyDto } from './dto/train-toy.dto';
import { TranslateRequestDto, TranslateStreamRequestDto } from './dto/translate.dto';
import { TranslateBatchRequestDto } from './dto/translate-batch.dto';

@Controller('transformer')
export class ApiController {
  constructor(private readonly apiService: ApiService) {}

  @Post('translate')
  translate(@Body() dto: TranslateRequestDto) {
    return this.apiService.translate(dto);
  }

  @Post('translate/batch')
  translateBatch(@Body() dto: TranslateBatchRequestDto) {
    return this.apiService.translateBatch(dto);
  }

  @Sse('translate/stream')
  streamTranslate(@Query() dto: TranslateStreamRequestDto): Observable<MessageEvent> {
    return this.apiService.translateStream(dto);
  }

  @Post('train/toy')
  @HttpCode(HttpStatus.ACCEPTED)
  async trainToy(@Body() dto: TrainToyDto) {
    await this.apiService.trainToy(dto);
    return { status: 'training-started' };
  }
}
