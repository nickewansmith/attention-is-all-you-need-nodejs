import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CheckpointService } from './checkpoint.service';

@Module({
  imports: [ConfigModule],
  providers: [CheckpointService],
  exports: [CheckpointService],
})
export class CheckpointModule {}
