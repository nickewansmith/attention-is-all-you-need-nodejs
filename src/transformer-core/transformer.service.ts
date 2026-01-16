import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import type { Tensor2D, Tensor3D, Tensor4D } from '@tensorflow/tfjs-node';
import { TokenizerService } from '../common/tokenizer/tokenizer.service';
import { TensorService } from '../tensor/tensor.service';
import { CheckpointService } from '../checkpoint/checkpoint.service';
import { SerializedVariableMap } from '../checkpoint/checkpoint.types';
import { FeedForwardNetworkService, FeedForwardWeights } from './feed-forward-network.service';
import { LayerNormalizationService, LayerNormWeights } from './layer-normalization.service';
import { AttentionWeights, MultiHeadAttentionService } from './multi-head-attention.service';
import { PositionalEncodingService } from './positional-encoding.service';
import { TransformerConfig } from './types/transformer-config';
import { TransformerConfigService } from './transformer-config.service';

interface EncoderLayerWeights {
  attention: AttentionWeights;
  feedForward: FeedForwardWeights;
  norm1: LayerNormWeights;
  norm2: LayerNormWeights;
}

interface DecoderLayerWeights {
  selfAttention: AttentionWeights;
  encDecAttention: AttentionWeights;
  feedForward: FeedForwardWeights;
  norm1: LayerNormWeights;
  norm2: LayerNormWeights;
  norm3: LayerNormWeights;
}

interface AttentionLayerSnapshot {
  layer: number;
  heads: number[][][];
}

export interface AttentionMaps {
  encoder: AttentionLayerSnapshot[];
  decoderSelf: AttentionLayerSnapshot[];
  decoderCross: AttentionLayerSnapshot[];
}

interface DecodeOptions {
  maxLength: number;
  strategy?: 'greedy' | 'beam';
  beamWidth?: number;
  includeAttention?: boolean;
}

interface DecodeResult {
  sequences: number[][];
  attention?: Array<AttentionMaps | null>;
}

interface BeamDecodeResult {
  sequence: number[];
  attention?: AttentionMaps;
}

export interface StreamTokenEvent {
  step: number;
  tokenId: number;
  token: string;
  partialTranslation: string;
}

export interface StreamBeamUpdate {
  rank: number;
  score: number;
  tokens: number[];
  partialTranslation: string;
}

@Injectable()
export class TransformerModelService implements OnModuleInit {
  private readonly logger = new Logger(TransformerModelService.name);
  private config!: TransformerConfig;
  private positionalEncoding!: Tensor2D;
  private encoderLayers: EncoderLayerWeights[] = [];
  private decoderLayers: DecoderLayerWeights[] = [];
  private inputEmbedding!: tf.Variable;
  private targetEmbedding!: tf.Variable;
  private finalProjection!: tf.Variable;
  private finalBias!: tf.Variable;
  private built = false;
  private weightsLoaded = false;

  constructor(
    private readonly positionalEncodingService: PositionalEncodingService,
    private readonly multiHeadAttention: MultiHeadAttentionService,
    private readonly feedForwardService: FeedForwardNetworkService,
    private readonly layerNormService: LayerNormalizationService,
    private readonly tensorService: TensorService,
    private readonly checkpointService: CheckpointService,
    private readonly configService: TransformerConfigService,
    private readonly tokenizer: TokenizerService,
  ) {}

  onModuleInit() {
    this.build();
  }

  build() {
    if (this.built) {
      return;
    }

    this.config = this.configService.getConfig();
    this.positionalEncoding = this.positionalEncodingService.create(
      this.config.maximumPositionEncoding,
      this.config.dModel,
    );

    const vocabInput = Math.max(this.config.inputVocabSize, this.tokenizer.vocabSize);
    const vocabTarget = Math.max(this.config.targetVocabSize, this.tokenizer.vocabSize);

    this.inputEmbedding = tf.variable(tf.randomNormal([vocabInput, this.config.dModel], 0, 0.1));
    this.targetEmbedding = tf.variable(tf.randomNormal([vocabTarget, this.config.dModel], 0, 0.1));
    this.finalProjection = tf.variable(tf.randomNormal([this.config.dModel, vocabTarget], 0, 0.02));
    this.finalBias = tf.variable(tf.zeros([vocabTarget]));

    this.encoderLayers = Array.from({ length: this.config.numLayers }, () => this.createEncoderLayer());
    this.decoderLayers = Array.from({ length: this.config.numLayers }, () => this.createDecoderLayer());

    this.built = true;
    if (!this.weightsLoaded) {
      this.weightsLoaded = this.checkpointService.loadTransformerWeights(this);
    }
    this.logger.log('Transformer weights initialized');
  }

  forward(inputIds: number[][], targetIds: number[][], training: boolean): tf.Tensor3D {
    if (!this.built) {
      this.build();
    }

    const tfInstance = this.tensorService.tf;
    const result = tfInstance.tidy(() => {
      const encoderInput = this.tensorService.tensor2d(inputIds, undefined, 'int32');
      const decoderInput = this.tensorService.tensor2d(targetIds, undefined, 'int32');

      const encPaddingMask = this.createPaddingMask(encoderInput);
      const lookAheadMask = this.createCombinedMask(decoderInput);
      const decPaddingMask = this.createPaddingMask(encoderInput);

      const encOutput = this.runEncoder(encoderInput, encPaddingMask, training);
      const decOutput = this.runDecoder(decoderInput, encOutput, lookAheadMask, decPaddingMask, training);

      const batchSize = targetIds.length;
      const targetLength = targetIds[0]?.length ?? 0;
      const reshaped = decOutput.reshape([-1, this.config.dModel]);
      const logits = reshaped.matMul(this.finalProjection).add(this.finalBias);
      const output = logits.reshape([batchSize, targetLength, this.finalBias.shape[0]]);
      this.tensorService.assertTensor3D(output, 'decoderLogits');
      return output;
    });

    this.tensorService.assertTensor3D(result, 'forwardOutput');
    return result;
  }

  encode(encoderInput: number[][]): tf.Tensor3D {
    const tfInstance = this.tensorService.tf;
    const result = tfInstance.tidy(() => {
      const inputTensor = this.tensorService.tensor2d(encoderInput, undefined, 'int32');
      const encMask = this.createPaddingMask(inputTensor);
      return this.runEncoder(inputTensor, encMask, false);
    });

    this.tensorService.assertTensor3D(result, 'encodeOutput');
    return result;
  }

  decodeSequences(inputIds: number[][], options: DecodeOptions): DecodeResult {
    const strategy = options.strategy ?? 'greedy';

    if (strategy === 'beam') {
      const results = inputIds.map((sequence) =>
        this.runBeamSearchSingle(sequence, options.maxLength, options.beamWidth ?? 3, options.includeAttention),
      );
      const sequences = results.map((result) => result.sequence);
      const attention = options.includeAttention ? results.map((result) => result.attention ?? null) : undefined;
      return { sequences, attention };
    }

    if (options.includeAttention) {
      return this.runGreedyWithAttention(inputIds, options.maxLength);
    }

    const sequences = this.runGreedySearch(inputIds, options.maxLength);
    return { sequences };
  }

  greedyDecode(inputIds: number[][], maxLength: number) {
    return this.decodeSequences(inputIds, { maxLength, strategy: 'greedy' }).sequences;
  }

  private runGreedySearch(inputIds: number[][], maxLength: number) {
    if (!this.built) {
      this.build();
    }

    const tfInstance = this.tensorService.tf;
    return tfInstance.tidy(() => {
      const encoderInput = this.tensorService.tensor2d(inputIds, undefined, 'int32');
      const encPaddingMask = this.createPaddingMask(encoderInput);
      const encOutput = this.runEncoder(encoderInput, encPaddingMask, false);

      const batchSize = inputIds.length;
      const outputs = Array.from({ length: batchSize }, () => [this.tokenizer.startTokenId]);
      const completed = new Array(batchSize).fill(false);

      for (let i = 0; i < maxLength; i += 1) {
        const decoderInput = this.tensorService.tensor2d(outputs, undefined, 'int32');
        const combinedMask = this.createCombinedMask(decoderInput);
        const decOutput = this.runDecoder(decoderInput, encOutput, combinedMask, encPaddingMask, false);
        const seqLength = decOutput.shape[1]!;
        const lastLogits = decOutput
          .slice([0, seqLength - 1, 0], [batchSize, 1, this.config.targetVocabSize])
          .reshape([batchSize, this.config.targetVocabSize]);
        const nextIdsTensor = tfInstance.argMax(lastLogits, 1);
        const nextIds = Array.from(nextIdsTensor.dataSync());
        nextIdsTensor.dispose();

        nextIds.forEach((nextId, batchIndex) => {
          outputs[batchIndex].push(nextId);
          if (nextId === this.tokenizer.endTokenId) {
            completed[batchIndex] = true;
          }
        });

        if (completed.every(Boolean)) {
          break;
        }
      }

      return outputs;
    });
  }

  private runGreedyWithAttention(inputIds: number[][], maxLength: number): DecodeResult {
    const sequences: number[][] = [];
    const attention: Array<AttentionMaps | null> = [];

    inputIds.forEach((sentence) => {
      const result = this.runGreedySingle(sentence, maxLength);
      sequences.push(result.sequence);
      attention.push(result.attention ?? null);
    });

    return { sequences, attention };
  }

  private runGreedySingle(inputIds: number[], maxLength: number): BeamDecodeResult {
    if (!this.built) {
      this.build();
    }

    const encoderInput = this.tensorService.tensor2d([inputIds], undefined, 'int32');
    const encPaddingMask = this.createPaddingMask(encoderInput);
    const encoderAttention: AttentionLayerSnapshot[] = [];
    const encoderRecorder = this.createAttentionRecorder(encoderAttention);
    const encOutput = this.runEncoder(encoderInput, encPaddingMask, false, encoderRecorder);

    const outputs = [this.tokenizer.startTokenId];
    let completed = false;
    let decoderSelf: AttentionLayerSnapshot[] = [];
    let decoderCross: AttentionLayerSnapshot[] = [];

    for (let step = 0; step < maxLength; step += 1) {
      if (completed) {
        break;
      }

      const decoderInput = this.tensorService.tensor2d([outputs], undefined, 'int32');
      const combinedMask = this.createCombinedMask(decoderInput);
      const currentSelf: AttentionLayerSnapshot[] = [];
      const currentCross: AttentionLayerSnapshot[] = [];
      const decOutput = this.runDecoder(
        decoderInput,
        encOutput,
        combinedMask,
        encPaddingMask,
        false,
        this.createAttentionRecorder(currentSelf),
        this.createAttentionRecorder(currentCross),
      );
      const seqLength = decOutput.shape[1]!;
      const lastLogits = decOutput
        .slice([0, seqLength - 1, 0], [1, 1, this.config.targetVocabSize])
        .reshape([1, this.config.targetVocabSize]);
      const nextIdTensor = this.tensorService.tf.argMax(lastLogits, 1);
      const nextId = nextIdTensor.dataSync()[0];
      nextIdTensor.dispose();
      lastLogits.dispose();
      decOutput.dispose();
      combinedMask.dispose();
      decoderInput.dispose();

      outputs.push(nextId);
      decoderSelf = currentSelf;
      decoderCross = currentCross;

      if (nextId === this.tokenizer.endTokenId) {
        completed = true;
      }
    }

    encoderInput.dispose();
    encPaddingMask.dispose();
    encOutput.dispose();

    return {
      sequence: outputs,
      attention: {
        encoder: encoderAttention,
        decoderSelf,
        decoderCross,
      },
    };
  }

  async streamGreedy(
    inputIds: number[],
    maxLength: number,
    includeAttention: boolean,
    onToken: (event: StreamTokenEvent) => Promise<void> | void,
  ): Promise<BeamDecodeResult> {
    if (!this.built) {
      this.build();
    }

    const encoderInput = this.tensorService.tensor2d([inputIds], undefined, 'int32');
    const encPaddingMask = this.createPaddingMask(encoderInput);
    const encoderAttention: AttentionLayerSnapshot[] = [];
    const encoderRecorder = includeAttention ? this.createAttentionRecorder(encoderAttention) : undefined;
    const encOutput = this.runEncoder(encoderInput, encPaddingMask, false, encoderRecorder);

    const outputs = [this.tokenizer.startTokenId];
    let completed = false;
    let decoderSelf: AttentionLayerSnapshot[] = [];
    let decoderCross: AttentionLayerSnapshot[] = [];

    for (let step = 0; step < maxLength; step += 1) {
      if (completed) {
        break;
      }

      const decoderInput = this.tensorService.tensor2d([outputs], undefined, 'int32');
      const combinedMask = this.createCombinedMask(decoderInput);
      const currentSelf: AttentionLayerSnapshot[] = [];
      const currentCross: AttentionLayerSnapshot[] = [];
      const decOutput = this.runDecoder(
        decoderInput,
        encOutput,
        combinedMask,
        encPaddingMask,
        false,
        includeAttention ? this.createAttentionRecorder(currentSelf) : undefined,
        includeAttention ? this.createAttentionRecorder(currentCross) : undefined,
      );
      const seqLength = decOutput.shape[1]!;
      const lastLogits = decOutput
        .slice([0, seqLength - 1, 0], [1, 1, this.config.targetVocabSize])
        .reshape([1, this.config.targetVocabSize]);
      const nextIdTensor = this.tensorService.tf.argMax(lastLogits, 1);
      const nextId = nextIdTensor.dataSync()[0];
      nextIdTensor.dispose();
      lastLogits.dispose();
      decOutput.dispose();
      combinedMask.dispose();
      decoderInput.dispose();

      outputs.push(nextId);
      if (includeAttention) {
        decoderSelf = currentSelf;
        decoderCross = currentCross;
      }

      await Promise.resolve(
        onToken({
          step: outputs.length - 1,
          tokenId: nextId,
          token: this.tokenizer.tokenFromId(nextId),
          partialTranslation: this.tokenizer.decode(outputs),
        }),
      );

      if (nextId === this.tokenizer.endTokenId) {
        completed = true;
      }
    }

    encoderInput.dispose();
    encPaddingMask.dispose();
    encOutput.dispose();

    return {
      sequence: outputs,
      attention: includeAttention
        ? {
            encoder: encoderAttention,
            decoderSelf,
            decoderCross,
          }
        : undefined,
    };
  }

  async streamBeam(
    inputIds: number[],
    maxLength: number,
    beamWidth: number,
    throttleMs: number,
    includeAttention: boolean,
    onUpdate: (event: StreamBeamUpdate) => Promise<void> | void,
  ): Promise<BeamDecodeResult> {
    if (!this.built) {
      this.build();
    }

    const encoderInput = this.tensorService.tensor2d([inputIds], undefined, 'int32');
    const encPaddingMask = this.createPaddingMask(encoderInput);
    const encoderAttention: AttentionLayerSnapshot[] = [];
    const encoderRecorder = includeAttention ? this.createAttentionRecorder(encoderAttention) : undefined;
    const encOutput = this.runEncoder(encoderInput, encPaddingMask, false, encoderRecorder);
    const throttle = Math.max(0, Math.floor(throttleMs));
    let lastEmit = 0;
    type BeamSnapshotEntry = {
      rank: number;
      score: number;
      tokens: number[];
      partialTranslation: string;
    };
    let pendingSnapshot: BeamSnapshotEntry[] | null = null;

    const emitSnapshot = async (snapshot: BeamSnapshotEntry[]) => {
      await Promise.all(
        snapshot.map((beam) =>
          Promise.resolve(
            onUpdate({
              rank: beam.rank,
              score: beam.score,
              tokens: beam.tokens,
              partialTranslation: beam.partialTranslation,
            }),
          ),
        ),
      );
    };

    const flushSnapshot = async (force = false) => {
      if (!pendingSnapshot) {
        return;
      }
      const now = Date.now();
      if (force || throttle === 0 || now - lastEmit >= throttle) {
        const snapshot = pendingSnapshot;
        pendingSnapshot = null;
        await emitSnapshot(snapshot);
        lastEmit = now;
      }
    };

    interface BeamState {
      tokens: number[];
      score: number;
      completed: boolean;
      selfAttention: AttentionLayerSnapshot[];
      crossAttention: AttentionLayerSnapshot[];
    }

    let beams: BeamState[] = [
      {
        tokens: [this.tokenizer.startTokenId],
        score: 0,
        completed: false,
        selfAttention: [],
        crossAttention: [],
      },
    ];

    for (let step = 0; step < maxLength; step += 1) {
      const candidates: BeamState[] = [];

      for (const beam of beams) {
        if (beam.completed) {
          candidates.push(beam);
          continue;
        }

        const decoderInput = this.tensorService.tensor2d([beam.tokens], undefined, 'int32');
        const combinedMask = this.createCombinedMask(decoderInput);
        const currentSelf: AttentionLayerSnapshot[] = [];
        const currentCross: AttentionLayerSnapshot[] = [];
        const decOutput = this.runDecoder(
          decoderInput,
          encOutput,
          combinedMask,
          encPaddingMask,
          false,
          includeAttention ? this.createAttentionRecorder(currentSelf) : undefined,
          includeAttention ? this.createAttentionRecorder(currentCross) : undefined,
        );
        const seqLength = decOutput.shape[1]!;
        const lastLogits = decOutput
          .slice([0, seqLength - 1, 0], [1, 1, this.config.targetVocabSize])
          .reshape([1, this.config.targetVocabSize]);
        const logProbs = this.tensorService.logSoftmax(lastLogits, 1);
        const logProbValues = Array.from(logProbs.dataSync());
        logProbs.dispose();
        lastLogits.dispose();
        decOutput.dispose();
        combinedMask.dispose();
        decoderInput.dispose();

        this.topK(logProbValues, beamWidth).forEach(({ index, value }) => {
          candidates.push({
            tokens: [...beam.tokens, index],
            score: beam.score + value,
            completed: index === this.tokenizer.endTokenId,
            selfAttention: includeAttention ? currentSelf : beam.selfAttention,
            crossAttention: includeAttention ? currentCross : beam.crossAttention,
          });
        });
      }

      candidates.sort((a, b) => b.score - a.score);
      beams = candidates.slice(0, beamWidth);

      pendingSnapshot = beams.map((beam, idx) => ({
        rank: idx + 1,
        score: beam.score,
        tokens: [...beam.tokens],
        partialTranslation: this.tokenizer.decode(beam.tokens),
      }));
      await flushSnapshot();

      if (beams.every((beam) => beam.completed)) {
        break;
      }
    }

    await flushSnapshot(true);

    encoderInput.dispose();
    encPaddingMask.dispose();
    encOutput.dispose();

    const bestBeam = beams[0];
    if (!bestBeam) {
      return { sequence: [this.tokenizer.startTokenId], attention: undefined };
    }

    const attention = includeAttention
      ? {
          encoder: encoderAttention,
          decoderSelf: bestBeam.selfAttention,
          decoderCross: bestBeam.crossAttention,
        }
      : undefined;

    return { sequence: bestBeam.tokens, attention };
  }

  private runBeamSearchSingle(
    inputIds: number[],
    maxLength: number,
    beamWidth: number,
    captureAttention?: boolean,
  ): BeamDecodeResult {
    if (!this.built) {
      this.build();
    }

    const encoderInput = this.tensorService.tensor2d([inputIds], undefined, 'int32');
    const encPaddingMask = this.createPaddingMask(encoderInput);
    const encoderAttention: AttentionLayerSnapshot[] = [];
    const encoderRecorder = captureAttention ? this.createAttentionRecorder(encoderAttention) : undefined;
    const encOutput = this.runEncoder(encoderInput, encPaddingMask, false, encoderRecorder);

    interface BeamState {
      tokens: number[];
      score: number;
      completed: boolean;
      selfAttention: AttentionLayerSnapshot[];
      crossAttention: AttentionLayerSnapshot[];
    }

    let beams: BeamState[] = [
      {
        tokens: [this.tokenizer.startTokenId],
        score: 0,
        completed: false,
        selfAttention: [],
        crossAttention: [],
      },
    ];

    for (let step = 0; step < maxLength; step += 1) {
      const candidates: BeamState[] = [];

      for (const beam of beams) {
        if (beam.completed) {
          candidates.push(beam);
          continue;
        }

        const decoderInput = this.tensorService.tensor2d([beam.tokens], undefined, 'int32');
        const combinedMask = this.createCombinedMask(decoderInput);
        const currentSelf: AttentionLayerSnapshot[] = [];
        const currentCross: AttentionLayerSnapshot[] = [];
        const selfRecorder = captureAttention ? this.createAttentionRecorder(currentSelf) : undefined;
        const crossRecorder = captureAttention ? this.createAttentionRecorder(currentCross) : undefined;
        const decOutput = this.runDecoder(
          decoderInput,
          encOutput,
          combinedMask,
          encPaddingMask,
          false,
          selfRecorder,
          crossRecorder,
        );
        const seqLength = decOutput.shape[1]!;
        const lastLogits = decOutput
          .slice([0, seqLength - 1, 0], [1, 1, this.config.targetVocabSize])
          .reshape([1, this.config.targetVocabSize]);
        const logProbs = this.tensorService.logSoftmax(lastLogits, 1);
        const logProbValues = Array.from(logProbs.dataSync());

        logProbs.dispose();
        lastLogits.dispose();
        decOutput.dispose();
        combinedMask.dispose();
        decoderInput.dispose();

        this.topK(logProbValues, beamWidth).forEach(({ index, value }) => {
          candidates.push({
            tokens: [...beam.tokens, index],
            score: beam.score + value,
            completed: index === this.tokenizer.endTokenId,
            selfAttention: captureAttention ? currentSelf : [],
            crossAttention: captureAttention ? currentCross : [],
          });
        });
      }

      candidates.sort((a, b) => b.score - a.score);
      beams = candidates.slice(0, beamWidth);

      if (beams.every((beam) => beam.completed)) {
        break;
      }
    }

    const bestBeam = beams[0];
    encoderInput.dispose();
    encPaddingMask.dispose();
    encOutput.dispose();

    if (!bestBeam) {
      return { sequence: [this.tokenizer.startTokenId], attention: undefined };
    }

    const attention = captureAttention
      ? {
          encoder: encoderAttention,
          decoderSelf: bestBeam.selfAttention,
          decoderCross: bestBeam.crossAttention,
        }
      : undefined;

    return { sequence: bestBeam.tokens, attention };
  }

  private createAttentionRecorder(target: AttentionLayerSnapshot[]) {
    return (layerIndex: number, weights: number[][][][]) => {
      target.push({ layer: layerIndex, heads: this.extractAttentionHeads(weights) });
    };
  }

  private extractAttentionHeads(weights: number[][][][]) {
    const batchSlice = weights[0] ?? [];
    return batchSlice.map((head) => head.map((row) => row.slice()));
  }

  private topK(values: number[], k: number) {
    return values
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, k);
  }

  private runEncoder(
    inputTensor: Tensor2D,
    mask: Tensor4D,
    training: boolean,
    attentionRecorder?: (layerIndex: number, weights: number[][][][]) => void,
  ): tf.Tensor3D {
    const tfInstance = this.tensorService.tf;
    let x = this.embed(inputTensor, this.inputEmbedding);
    x = x.mul(tfInstance.scalar(Math.sqrt(this.config.dModel)));
    x = this.addPositionalEncoding(x);

    let output = x;
    this.encoderLayers.forEach((layer, layerIndex) => {
      const attnOutput = this.multiHeadAttention.apply(
        output,
        output,
        output,
        mask,
        layer.attention,
        this.config,
        training,
        attentionRecorder ? (weights) => attentionRecorder(layerIndex, weights) : undefined,
      );
      const out1 = this.layerNormService.apply(
        output.add(attnOutput),
        layer.norm1,
        this.config.layerNormEpsilon,
      );
      const ffnOutput = this.feedForwardService.apply(out1, layer.feedForward, this.config.dropoutRate, training);
      output = this.layerNormService.apply(out1.add(ffnOutput), layer.norm2, this.config.layerNormEpsilon);
    });

    return output;
  }

  private runDecoder(
    targetTensor: Tensor2D,
    encOutput: tf.Tensor3D,
    lookAheadMask: Tensor4D,
    paddingMask: Tensor4D,
    training: boolean,
    selfAttentionRecorder?: (layerIndex: number, weights: number[][][][]) => void,
    crossAttentionRecorder?: (layerIndex: number, weights: number[][][][]) => void,
  ): tf.Tensor3D {
    const tfInstance = this.tensorService.tf;
    let x = this.embed(targetTensor, this.targetEmbedding);
    x = x.mul(tfInstance.scalar(Math.sqrt(this.config.dModel)));
    x = this.addPositionalEncoding(x);

    let output = x;
    this.decoderLayers.forEach((layer, layerIndex) => {
      const attn1 = this.multiHeadAttention.apply(
        output,
        output,
        output,
        lookAheadMask,
        layer.selfAttention,
        this.config,
        training,
        selfAttentionRecorder ? (weights) => selfAttentionRecorder(layerIndex, weights) : undefined,
      );
      const out1 = this.layerNormService.apply(output.add(attn1), layer.norm1, this.config.layerNormEpsilon);

      const attn2 = this.multiHeadAttention.apply(
        out1,
        encOutput,
        encOutput,
        paddingMask,
        layer.encDecAttention,
        this.config,
        training,
        crossAttentionRecorder ? (weights) => crossAttentionRecorder(layerIndex, weights) : undefined,
      );
      const out2 = this.layerNormService.apply(out1.add(attn2), layer.norm2, this.config.layerNormEpsilon);

      const ffnOutput = this.feedForwardService.apply(out2, layer.feedForward, this.config.dropoutRate, training);
      output = this.layerNormService.apply(out2.add(ffnOutput), layer.norm3, this.config.layerNormEpsilon);
    });

    return output;
  }

  private embed(tokenTensor: Tensor2D, embedding: tf.Variable): tf.Tensor3D {
    const tfInstance = this.tensorService.tf;
    const [batchSize, seqLen] = tokenTensor.shape;
    const flat = tokenTensor.reshape([-1]);
    const oneHot = tfInstance.oneHot(flat, embedding.shape[0]);
    const embeddings = oneHot.matMul(embedding);
    const output = embeddings.reshape([batchSize, seqLen, this.config.dModel]);
    this.tensorService.assertTensor3D(output, 'embeddingOutput');
    return output;
  }

  private addPositionalEncoding(x: tf.Tensor3D): tf.Tensor3D {
    const seqLen = x.shape[1]!;
    const positionalSlice = this.positionalEncoding.slice([0, 0], [seqLen, this.config.dModel]);
    this.tensorService.assertTensor2D(positionalSlice, 'positionalSlice');
    const expanded = positionalSlice.expandDims(0);
    this.tensorService.assertTensor3D(expanded, 'positionalEncodingExpanded');
    return x.add(expanded);
  }

  private createEncoderLayer(): EncoderLayerWeights {
    return {
      attention: this.multiHeadAttention.createWeights(this.config),
      feedForward: this.feedForwardService.createWeights(this.config.dModel, this.config.dff),
      norm1: this.layerNormService.createWeights(this.config.dModel),
      norm2: this.layerNormService.createWeights(this.config.dModel),
    };
  }

  private createDecoderLayer(): DecoderLayerWeights {
    return {
      selfAttention: this.multiHeadAttention.createWeights(this.config),
      encDecAttention: this.multiHeadAttention.createWeights(this.config),
      feedForward: this.feedForwardService.createWeights(this.config.dModel, this.config.dff),
      norm1: this.layerNormService.createWeights(this.config.dModel),
      norm2: this.layerNormService.createWeights(this.config.dModel),
      norm3: this.layerNormService.createWeights(this.config.dModel),
    };
  }

  serializeVariables(): SerializedVariableMap {
    const map = this.collectVariables();
    const serialized: SerializedVariableMap = {};
    Object.entries(map).forEach(([name, variable]) => {
      const values = Array.from(variable.dataSync());
      serialized[name] = { shape: variable.shape.slice(), values };
    });
    return serialized;
  }

  loadSerializedVariables(serialized: SerializedVariableMap) {
    const map = this.collectVariables();
    Object.entries(serialized).forEach(([name, snapshot]) => {
      const variable = map[name];
      if (!variable) {
        return;
      }
      const tensor = this.tensorService.tf.tensor(snapshot.values, snapshot.shape, 'float32');
      variable.assign(tensor);
      tensor.dispose();
    });
  }

  private collectVariables(): Record<string, tf.Variable> {
    const map: Record<string, tf.Variable> = {
      inputEmbedding: this.inputEmbedding,
      targetEmbedding: this.targetEmbedding,
      finalProjection: this.finalProjection,
      finalBias: this.finalBias,
    };

    this.encoderLayers.forEach((layer, idx) => {
      map[`encoder.${idx}.attention.wq`] = layer.attention.wq;
      map[`encoder.${idx}.attention.wk`] = layer.attention.wk;
      map[`encoder.${idx}.attention.wv`] = layer.attention.wv;
      map[`encoder.${idx}.attention.wo`] = layer.attention.wo;
      map[`encoder.${idx}.attention.bq`] = layer.attention.bq;
      map[`encoder.${idx}.attention.bk`] = layer.attention.bk;
      map[`encoder.${idx}.attention.bv`] = layer.attention.bv;
      map[`encoder.${idx}.attention.bo`] = layer.attention.bo;
      map[`encoder.${idx}.ffn.w1`] = layer.feedForward.w1;
      map[`encoder.${idx}.ffn.b1`] = layer.feedForward.b1;
      map[`encoder.${idx}.ffn.w2`] = layer.feedForward.w2;
      map[`encoder.${idx}.ffn.b2`] = layer.feedForward.b2;
      map[`encoder.${idx}.norm1.gamma`] = layer.norm1.gamma;
      map[`encoder.${idx}.norm1.beta`] = layer.norm1.beta;
      map[`encoder.${idx}.norm2.gamma`] = layer.norm2.gamma;
      map[`encoder.${idx}.norm2.beta`] = layer.norm2.beta;
    });

    this.decoderLayers.forEach((layer, idx) => {
      map[`decoder.${idx}.selfAttention.wq`] = layer.selfAttention.wq;
      map[`decoder.${idx}.selfAttention.wk`] = layer.selfAttention.wk;
      map[`decoder.${idx}.selfAttention.wv`] = layer.selfAttention.wv;
      map[`decoder.${idx}.selfAttention.wo`] = layer.selfAttention.wo;
      map[`decoder.${idx}.selfAttention.bq`] = layer.selfAttention.bq;
      map[`decoder.${idx}.selfAttention.bk`] = layer.selfAttention.bk;
      map[`decoder.${idx}.selfAttention.bv`] = layer.selfAttention.bv;
      map[`decoder.${idx}.selfAttention.bo`] = layer.selfAttention.bo;

      map[`decoder.${idx}.encDecAttention.wq`] = layer.encDecAttention.wq;
      map[`decoder.${idx}.encDecAttention.wk`] = layer.encDecAttention.wk;
      map[`decoder.${idx}.encDecAttention.wv`] = layer.encDecAttention.wv;
      map[`decoder.${idx}.encDecAttention.wo`] = layer.encDecAttention.wo;
      map[`decoder.${idx}.encDecAttention.bq`] = layer.encDecAttention.bq;
      map[`decoder.${idx}.encDecAttention.bk`] = layer.encDecAttention.bk;
      map[`decoder.${idx}.encDecAttention.bv`] = layer.encDecAttention.bv;
      map[`decoder.${idx}.encDecAttention.bo`] = layer.encDecAttention.bo;

      map[`decoder.${idx}.ffn.w1`] = layer.feedForward.w1;
      map[`decoder.${idx}.ffn.b1`] = layer.feedForward.b1;
      map[`decoder.${idx}.ffn.w2`] = layer.feedForward.w2;
      map[`decoder.${idx}.ffn.b2`] = layer.feedForward.b2;

      map[`decoder.${idx}.norm1.gamma`] = layer.norm1.gamma;
      map[`decoder.${idx}.norm1.beta`] = layer.norm1.beta;
      map[`decoder.${idx}.norm2.gamma`] = layer.norm2.gamma;
      map[`decoder.${idx}.norm2.beta`] = layer.norm2.beta;
      map[`decoder.${idx}.norm3.gamma`] = layer.norm3.gamma;
      map[`decoder.${idx}.norm3.beta`] = layer.norm3.beta;
    });

    return map;
  }

  private createPaddingMask(seq: Tensor2D): Tensor4D {
    const tfInstance = this.tensorService.tf;
    const mask = seq.equal(tfInstance.scalar(this.tokenizer.padTokenId, 'int32')).cast('float32');
    const expanded = mask.expandDims(1).expandDims(1);
    this.tensorService.assertTensor4D(expanded, 'paddingMask');
    return expanded;
  }

  private createLookAheadMask(size: number): Tensor4D {
    const tfInstance = this.tensorService.tf;
    const maskData: number[] = [];
    for (let row = 0; row < size; row += 1) {
      for (let col = 0; col < size; col += 1) {
        maskData.push(col > row ? 1 : 0);
      }
    }
    const mask = tfInstance.tensor2d(maskData, [size, size], 'float32');
    const expanded = mask.reshape([1, 1, size, size]);
    this.tensorService.assertTensor4D(expanded, 'lookAheadMask');
    return expanded;
  }

  private createCombinedMask(seq: Tensor2D): Tensor4D {
    const lookAheadMask = this.createLookAheadMask(seq.shape[1]!);
    const paddingMask = this.createPaddingMask(seq);
    const combined = lookAheadMask.maximum(paddingMask);
    this.tensorService.assertTensor4D(combined, 'combinedMask');
    return combined;
  }
}
