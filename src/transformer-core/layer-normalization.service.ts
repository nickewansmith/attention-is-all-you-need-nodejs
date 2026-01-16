import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import type { Tensor3D } from '@tensorflow/tfjs-node';
import { TensorService } from '../tensor/tensor.service';

export interface LayerNormWeights {
  gamma: tf.Variable;
  beta: tf.Variable;
}

@Injectable()
export class LayerNormalizationService {
  constructor(private readonly tensorService: TensorService) {}

  createWeights(dim: number): LayerNormWeights {
    return {
      gamma: tf.variable(tf.ones([dim])),
      beta: tf.variable(tf.zeros([dim])),
    };
  }

  apply(x: Tensor3D, weights: LayerNormWeights, epsilon: number): Tensor3D {
    const result = tf.tidy(() => {
      const mean = tf.mean(x, -1, true);
      const variance = tf.mean(tf.square(x.sub(mean)), -1, true);
      const normalized = x.sub(mean).div(tf.sqrt(variance.add(tf.scalar(epsilon))));
      const output = normalized.mul(weights.gamma).add(weights.beta);
      this.tensorService.assertTensor3D(output, 'layerNormOutput');
      return output;
    });

    this.tensorService.assertTensor3D(result, 'layerNormResult');
    return result;
  }
}
