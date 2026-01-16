import { Injectable } from '@nestjs/common';
import type { Tensor2D } from '@tensorflow/tfjs-node';
import { TensorService } from '../tensor/tensor.service';

@Injectable()
export class PositionalEncodingService {
  constructor(private readonly tensorService: TensorService) {}

  create(maxPosition: number, dModel: number): Tensor2D {
    const tf = this.tensorService.tf;
    const angleRates = [] as number[];

    for (let pos = 0; pos < maxPosition; pos += 1) {
      for (let i = 0; i < dModel; i += 1) {
        const angleRate = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        angleRates.push(angleRate);
      }
    }

    const angleTensor = tf.tensor2d(angleRates, [maxPosition, dModel], 'float32');
    const sinMask = tf.tensor1d(
      Array.from({ length: dModel }, (_, idx) => (idx % 2 === 0 ? 1 : 0)),
      'float32',
    );
    const cosMask = tf.tensor1d(
      Array.from({ length: dModel }, (_, idx) => (idx % 2 === 1 ? 1 : 0)),
      'float32',
    );

    const sinComponent = tf.sin(angleTensor).mul(sinMask);
    const cosComponent = tf.cos(angleTensor).mul(cosMask);

    return sinComponent.add(cosComponent) as Tensor2D;
  }
}
