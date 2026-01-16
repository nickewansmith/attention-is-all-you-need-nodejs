import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import type { Tensor3D } from '@tensorflow/tfjs-node';
import { TensorService } from '../tensor/tensor.service';

export interface FeedForwardWeights {
  w1: tf.Variable;
  b1: tf.Variable;
  w2: tf.Variable;
  b2: tf.Variable;
}

@Injectable()
export class FeedForwardNetworkService {
  constructor(private readonly tensorService: TensorService) {}

  createWeights(dModel: number, dff: number): FeedForwardWeights {
    return {
      w1: tf.variable(tf.randomNormal([dModel, dff], 0, 0.02)),
      b1: tf.variable(tf.zeros([dff])),
      w2: tf.variable(tf.randomNormal([dff, dModel], 0, 0.02)),
      b2: tf.variable(tf.zeros([dModel])),
    };
  }

  apply(x: Tensor3D, weights: FeedForwardWeights, dropoutRate: number, training: boolean): Tensor3D {
    const result = tf.tidy(() => {
      const [batchSize, seqLen, depth] = x.shape as [number, number, number];
      const reshaped = x.reshape([-1, depth]);
      let hidden = reshaped.matMul(weights.w1).add(weights.b1);
      hidden = hidden.relu();
      hidden = hidden.reshape([batchSize, seqLen, weights.b1.shape[0]]);
      hidden = this.tensorService.dropout(hidden, dropoutRate, training);
      let output = hidden.reshape([-1, weights.b1.shape[0]]).matMul(weights.w2).add(weights.b2);
      output = output.reshape([batchSize, seqLen, weights.b2.shape[0]]);
      this.tensorService.assertTensor3D(output, 'feedForwardOutput');
      return output;
    });

    this.tensorService.assertTensor3D(result, 'feedForwardResult');
    return result;
  }
}
