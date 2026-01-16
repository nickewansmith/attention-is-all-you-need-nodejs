import { Global, Module } from '@nestjs/common';
import { TensorService } from './tensor.service';

@Global()
@Module({
  providers: [TensorService],
  exports: [TensorService],
})
export class TensorModule {}
