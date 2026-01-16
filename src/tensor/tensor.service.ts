import { Injectable } from '@nestjs/common';
import { loadTensorflowBackend } from './backend-loader';
import type * as tfTypes from '@tensorflow/tfjs-node';
import type { Tensor2D, Tensor3D, Tensor4D } from '@tensorflow/tfjs-node';

const tf: typeof tfTypes = loadTensorflowBackend();

export type Tensor = tfTypes.Tensor;

@Injectable()
export class TensorService {
  readonly tf = tf;

  tensor(data: tfTypes.TensorLike, shape?: number[], dtype?: tfTypes.DataType) {
    return tf.tensor(data, shape, dtype);
  }

  tensor2d(data: number[][], shape?: [number, number], dtype?: tfTypes.DataType): Tensor2D {
    return tf.tensor2d(data, shape, dtype);
  }

  zeros(shape: number[], dtype?: tfTypes.DataType) {
    return tf.zeros(shape, dtype);
  }

  ones(shape: number[], dtype?: tfTypes.DataType) {
    return tf.ones(shape, dtype);
  }

  matMul(a: Tensor, b: Tensor, transposeA = false, transposeB = false) {
    return tf.matMul(a, b, transposeA, transposeB);
  }

  softmax(logits: Tensor, axis = -1) {
    return tf.softmax(logits, axis);
  }

  logSoftmax(logits: Tensor, axis = -1) {
    return tf.logSoftmax(logits, axis);
  }

  dropout(x: Tensor, rate: number, training: boolean) {
    if (!training || rate <= 0) {
      return x;
    }

    const keepProb = 1 - rate;
    return tf.tidy(() => {
      const randomTensor = tf.randomUniform(x.shape, 0, 1, 'float32');
      const keepMask = randomTensor.less(tf.scalar(keepProb)).cast(x.dtype);
      return x.mul(keepMask).div(tf.scalar(keepProb));
    });
  }

  assertTensor2D(tensor: Tensor, context = 'tensor'): asserts tensor is Tensor2D {
    if (tensor.rank !== 2) {
      throw new Error(`${context} expected rank 2 tensor, received rank ${tensor.rank}`);
    }
  }

  assertTensor3D(tensor: Tensor, context = 'tensor'): asserts tensor is Tensor3D {
    if (tensor.rank !== 3) {
      throw new Error(`${context} expected rank 3 tensor, received rank ${tensor.rank}`);
    }
  }

  assertTensor4D(tensor: Tensor, context = 'tensor'): asserts tensor is Tensor4D {
    if (tensor.rank !== 4) {
      throw new Error(`${context} expected rank 4 tensor, received rank ${tensor.rank}`);
    }
  }
}
