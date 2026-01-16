import * as tf from '@tensorflow/tfjs-node';

export class AdaptiveAdamOptimizer extends tf.AdamOptimizer {
  constructor(learningRate: number, beta1: number, beta2: number, epsilon?: number) {
    super(learningRate, beta1, beta2, epsilon);
  }

  updateLearningRate(learningRate: number) {
    this.learningRate = learningRate;
  }
}
