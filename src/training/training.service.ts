import { Injectable, Logger } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import { TensorService } from '../tensor/tensor.service';
import { TransformerModelService } from '../transformer-core/transformer.service';
import { TransformerConfigService } from '../transformer-core/transformer-config.service';
import { TokenizerService } from '../common/tokenizer/tokenizer.service';
import { CheckpointService } from '../checkpoint/checkpoint.service';
import { MetricsService } from '../observability/metrics.service';
import { DatasetService, TrainingBatch } from './dataset.service';
import { NoamSchedulerService } from './noam-scheduler.service';
import { EvaluationService } from './evaluation.service';
import { AdaptiveAdamOptimizer } from './optimizers/adaptive-adam.optimizer';

@Injectable()
export class TrainingService {
  private readonly logger = new Logger(TrainingService.name);
  private readonly optimizer: AdaptiveAdamOptimizer;
  private globalStep = 1;
  private checkpointLoaded = false;

  constructor(
    private readonly transformer: TransformerModelService,
    private readonly datasetService: DatasetService,
    private readonly scheduler: NoamSchedulerService,
    private readonly tensorService: TensorService,
    private readonly configService: TransformerConfigService,
    private readonly tokenizer: TokenizerService,
    private readonly checkpointService: CheckpointService,
    private readonly evaluationService: EvaluationService,
    private readonly metricsService: MetricsService,
  ) {
    this.optimizer = new AdaptiveAdamOptimizer(0.001, 0.9, 0.98, 1e-9);
  }

  async trainToyEpoch(epochs = 1, batchSize = 2) {
    await this.ensureCheckpointLoaded();
    const { maximumPositionEncoding } = this.configService.getConfig();
    const batches = this.datasetService.createBatches(batchSize, maximumPositionEncoding);

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      this.logger.log(`Starting epoch ${epoch + 1}/${epochs}`);

      for (const batch of batches) {
        const loss = await this.trainBatch(batch);
        const perplexity = Math.exp(loss);
        this.logger.log(`Step ${this.globalStep} | Loss: ${loss.toFixed(4)} | PPL: ${perplexity.toFixed(4)}`);
        this.metricsService.markTrainingStep();
        this.globalStep += 1;
      }

      await this.evaluateSample(Math.min(5, batchSize));
      await this.checkpointService.save(this.transformer, this.optimizer, this.globalStep);
    }
  }

  private async trainBatch(batch: TrainingBatch) {
    const decoderInput = batch.decoderInputs.map((seq) => seq.slice(0, -1));
    const decoderTarget = batch.decoderInputs.map((seq) => seq.slice(1));

    const learningRate = this.scheduler.getLearningRate(this.globalStep);
    this.optimizer.updateLearningRate(learningRate);

    const lossTensor = this.optimizer.minimize(() => {
      const predictions = this.transformer.forward(batch.encoderInputs, decoderInput, true);
      return this.computeLoss(predictions, decoderTarget);
    }, true);

    const lossValue = lossTensor ? lossTensor.dataSync()[0] : 0;
    lossTensor?.dispose();
    return lossValue;
  }

  private computeLoss(predictions: tf.Tensor3D, targetSequences: number[][]): tf.Scalar {
    const tfInstance = this.tensorService.tf;
    const { targetVocabSize } = this.configService.getConfig();
    const labelsTensor = this.tensorService.tensor2d(targetSequences, undefined, 'int32');
    const flatLabels = labelsTensor.reshape([-1]);
    const logits = predictions.reshape([-1, targetVocabSize]);
    const oneHotLabels = tfInstance.oneHot(flatLabels, targetVocabSize);
    const losses = tfInstance.losses.softmaxCrossEntropy(oneHotLabels, logits);
    return tfInstance.mean(losses) as tf.Scalar;
  }

  private async ensureCheckpointLoaded() {
    if (this.checkpointLoaded) {
      return;
    }

    const result = await this.checkpointService.restoreForTraining(this.transformer, this.optimizer);
    if (result?.globalStep) {
      this.globalStep = result.globalStep;
    }
    this.checkpointLoaded = true;
  }

  private async evaluateSample(sampleSize: number) {
    const { maximumPositionEncoding } = this.configService.getConfig();
    const sample = this.datasetService.sample(sampleSize);
    if (sample.length === 0) {
      return;
    }

    const translations = sample.map((example) => {
      const encoded = this.tokenizer.encode(example.source, maximumPositionEncoding);
      const decodedTokens = this.transformer.greedyDecode([encoded], maximumPositionEncoding)[0];
      return this.tokenizer.decode(decodedTokens);
    });

    const references = sample.map((example) => example.target);
    const bleu = this.evaluationService.computeBleu(references, translations);
    this.logger.log(`Validation BLEU@${sample.length}: ${bleu.toFixed(4)}`);
  }
}
