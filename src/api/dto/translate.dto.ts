import { IsBoolean, IsIn, IsInt, IsOptional, IsString, Max, Min } from 'class-validator';
import { Type } from 'class-transformer';
import type { AttentionMaps } from '../../transformer-core/transformer.service';

export class TranslateRequestDto {
  @IsString()
  text!: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(4)
  @Max(256)
  maxLength?: number;

  @IsOptional()
  @IsIn(['greedy', 'beam'])
  strategy?: 'greedy' | 'beam';

  @IsOptional()
  @IsInt()
  @Type(() => Number)
  @Min(1)
  @Max(16)
  beamWidth?: number;

  @IsOptional()
  @IsBoolean()
  includeAttention?: boolean;
}

export class TranslateStreamRequestDto extends TranslateRequestDto {
  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  @Max(5000)
  beamThrottleMs?: number;
}

export interface TranslateResponseDto {
  translation: string;
  tokens: number[];
  attention?: AttentionMaps;
}
