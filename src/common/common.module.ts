import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TokenizerService } from './tokenizer/tokenizer.service';

@Module({
  imports: [ConfigModule],
  providers: [TokenizerService],
  exports: [TokenizerService],
})
export class CommonModule {}
