import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import type { Tensor3D, Tensor4D } from '@tensorflow/tfjs-node';
import { TensorService } from '../tensor/tensor.service';
import { TransformerConfig } from './types/transformer-config';

export interface AttentionWeights {
  wq: tf.Variable;
  wk: tf.Variable;
  wv: tf.Variable;
  wo: tf.Variable;
  bq: tf.Variable;
  bk: tf.Variable;
  bv: tf.Variable;
  bo: tf.Variable;
}

@Injectable()
export class MultiHeadAttentionService {
  constructor(private readonly tensorService: TensorService) {}

  createWeights(config: TransformerConfig): AttentionWeights {
    const { dModel } = config;
    return {
      wq: tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02)),
      wk: tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02)),
      wv: tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02)),
      wo: tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02)),
      bq: tf.variable(tf.zeros([dModel])),
      bk: tf.variable(tf.zeros([dModel])),
      bv: tf.variable(tf.zeros([dModel])),
      bo: tf.variable(tf.zeros([dModel])),
    };
  }

  apply(
    query: Tensor3D,
    key: Tensor3D,
    value: Tensor3D,
    mask: Tensor4D | null,
    weights: AttentionWeights,
    config: TransformerConfig,
    training: boolean,
    attentionCallback?: (weights: number[][][][]) => void,
  ): tf.Tensor3D {
    const tfInstance = this.tensorService.tf;
    const result = tfInstance.tidy(() => {
      const q = this.project(query, weights.wq, weights.bq);
      const k = this.project(key, weights.wk, weights.bk);
      const v = this.project(value, weights.wv, weights.bv);

      const [batchSize, seqLen] = q.shape as [number, number, number];
      const depth = config.dModel / config.numHeads;

      const qSplit = this.splitHeads(q, batchSize, config.numHeads, depth);
      const kSplit = this.splitHeads(k, batchSize, config.numHeads, depth);
      const vSplit = this.splitHeads(v, batchSize, config.numHeads, depth);

      const scaledAttention = this.scaledDotProductAttention(qSplit, kSplit, vSplit, mask, depth, attentionCallback);
      const scaledAttentionTransposed = tfInstance.transpose(scaledAttention, [0, 2, 1, 3]);
      this.tensorService.assertTensor4D(scaledAttentionTransposed, 'scaledAttentionTransposed');
      const concatAttention = scaledAttentionTransposed.reshape([batchSize, seqLen, config.dModel]);
      this.tensorService.assertTensor3D(concatAttention, 'concatAttention');

      let output = concatAttention.matMul(weights.wo).add(weights.bo);
      this.tensorService.assertTensor3D(output, 'attentionProjection');
      output = this.tensorService.dropout(output, config.dropoutRate, training);
      this.tensorService.assertTensor3D(output, 'attentionDropout');
      return output;
    });

    this.tensorService.assertTensor3D(result, 'multiHeadAttentionOutput');
    return result;
  }

  private project(x: Tensor3D, kernel: tf.Variable, bias: tf.Variable): Tensor3D {
    const [batchSize, seqLen, depth] = x.shape as [number, number, number];
    const reshaped = x.reshape([-1, depth]);
    const projected = reshaped.matMul(kernel).add(bias);
    const output = projected.reshape([batchSize, seqLen, kernel.shape[1]!]);
    this.tensorService.assertTensor3D(output, 'projectedAttention');
    return output;
  }

  private splitHeads(
    x: Tensor3D,
    batchSize: number,
    numHeads: number,
    depth: number,
  ): Tensor4D {
    const tfInstance = this.tensorService.tf;
    const reshaped = x.reshape([batchSize, -1, numHeads, depth]);
    const split = tfInstance.transpose(reshaped, [0, 2, 1, 3]);
    this.tensorService.assertTensor4D(split, 'splitHeads');
    return split;
  }

  private scaledDotProductAttention(
    query: Tensor4D,
    key: Tensor4D,
    value: Tensor4D,
    mask: Tensor4D | null,
    depth: number,
    attentionCallback?: (weights: number[][][][]) => void,
  ): Tensor4D {
    const tfInstance = this.tensorService.tf;
    let matmulQK = tfInstance.matMul(query, key, false, true);
    matmulQK = matmulQK.div(tfInstance.scalar(Math.sqrt(depth)));

    if (mask) {
      const negativeInfinity = tfInstance.scalar(-1e9);
      matmulQK = matmulQK.add(mask.mul(negativeInfinity));
    }

    const attentionWeights = tfInstance.softmax(matmulQK, -1);
    if (attentionCallback) {
      const snapshot = attentionWeights.arraySync() as number[][][][];
      attentionCallback(snapshot);
    }
    const output = tfInstance.matMul(attentionWeights, value);
    this.tensorService.assertTensor4D(output, 'scaledDotProductAttention');
    return output;
  }
}
