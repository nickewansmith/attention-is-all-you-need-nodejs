import { IsInt, IsOptional, Max, Min } from 'class-validator';

export class TrainToyDto {
  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(100)
  epochs?: number;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(16)
  batchSize?: number;
}
