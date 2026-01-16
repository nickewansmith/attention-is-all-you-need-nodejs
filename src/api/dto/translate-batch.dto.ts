import { Type } from 'class-transformer';
import { ArrayMinSize, IsArray, ValidateNested } from 'class-validator';
import { TranslateRequestDto, TranslateResponseDto } from './translate.dto';

export class TranslateBatchRequestDto {
  @IsArray()
  @ArrayMinSize(1)
  @ValidateNested({ each: true })
  @Type(() => TranslateRequestDto)
  items!: TranslateRequestDto[];
}

export interface TranslateBatchResponseDto {
  results: TranslateResponseDto[];
}
