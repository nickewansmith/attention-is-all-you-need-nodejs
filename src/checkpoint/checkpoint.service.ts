import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import { TransformerModelService } from '../transformer-core/transformer.service';
import { CheckpointPayload, SerializedOptimizerWeight } from './checkpoint.types';

const DEFAULT_CHECKPOINT_PATH = 'checkpoints/latest.json';

@Injectable()
export class CheckpointService {
  private readonly logger = new Logger(CheckpointService.name);
  private readonly defaultSavePath: string;
  private readonly defaultLoadPath?: string;

  constructor(private readonly configService: ConfigService) {
    this.defaultSavePath = this.configService.get<string>('CHECKPOINT_SAVE_PATH') ?? DEFAULT_CHECKPOINT_PATH;
    const configuredLoadPath = this.configService.get<string>('CHECKPOINT_LOAD_PATH');
    this.defaultLoadPath = configuredLoadPath ?? this.defaultSavePath;
  }

  async save(transformer: TransformerModelService, optimizer: tf.AdamOptimizer, globalStep: number, filePath?: string) {
    const snapshot = await this.createPayload(transformer, optimizer, globalStep);
    const targetPath = filePath ?? this.defaultSavePath;
    this.ensureDirectory(targetPath);
    fs.writeFileSync(targetPath, JSON.stringify(snapshot));
    this.logger.log(`Checkpoint saved to ${targetPath} (step ${globalStep})`);
  }

  loadTransformerWeights(transformer: TransformerModelService, filePath?: string) {
    const pathToLoad = filePath ?? this.defaultLoadPath;
    if (!pathToLoad) {
      return false;
    }

    const payload = this.readCheckpoint(pathToLoad);
    if (!payload) {
      return false;
    }

    transformer.loadSerializedVariables(payload.variables);
    this.logger.log(`Loaded transformer weights from checkpoint (step ${payload.globalStep})`);
    return true;
  }

  async restoreForTraining(
    transformer: TransformerModelService,
    optimizer: tf.AdamOptimizer,
    filePath?: string,
  ): Promise<{ globalStep: number } | null> {
    const pathToLoad = filePath ?? this.defaultLoadPath;
    if (!pathToLoad) {
      return null;
    }

    const payload = this.readCheckpoint(pathToLoad);
    if (!payload) {
      return null;
    }

    transformer.loadSerializedVariables(payload.variables);
    if (payload.optimizer && payload.optimizer.length > 0) {
      await this.applyOptimizerWeights(payload.optimizer, optimizer);
    }
    this.logger.log(`Restored checkpoint for training (step ${payload.globalStep})`);
    return { globalStep: payload.globalStep };
  }

  private async createPayload(
    transformer: TransformerModelService,
    optimizer: tf.AdamOptimizer,
    globalStep: number,
  ): Promise<CheckpointPayload> {
    const optimizerWeights = await this.serializeOptimizerWeights(optimizer);
    return {
      createdAt: new Date().toISOString(),
      globalStep,
      variables: transformer.serializeVariables(),
      optimizer: optimizerWeights,
    };
  }

  private readCheckpoint(filePath?: string): CheckpointPayload | null {
    if (!filePath) {
      return null;
    }

    if (!fs.existsSync(filePath)) {
      if (filePath !== this.defaultSavePath) {
        this.logger.warn(`Checkpoint file not found at ${filePath}`);
      }
      return null;
    }

    try {
      const raw = fs.readFileSync(filePath, 'utf8');
      return JSON.parse(raw) as CheckpointPayload;
    } catch (error) {
      this.logger.error(`Failed to load checkpoint at ${filePath}: ${error}`);
      return null;
    }
  }

  private async serializeOptimizerWeights(optimizer: tf.AdamOptimizer) {
    const weights = await optimizer.getWeights();
    const serialized: SerializedOptimizerWeight[] = [];
    for (const named of weights) {
      const values = Array.from(await named.tensor.data());
      serialized.push({ name: named.name, shape: named.tensor.shape.slice(), values });
      named.tensor.dispose();
    }
    return serialized;
  }

  private async applyOptimizerWeights(weights: SerializedOptimizerWeight[], optimizer: tf.AdamOptimizer) {
    const namedTensors = weights.map((weight) => ({
      name: weight.name,
      tensor: tf.tensor(weight.values, weight.shape, 'float32'),
    }));
    await optimizer.setWeights(namedTensors);
    namedTensors.forEach((named) => named.tensor.dispose());
  }

  private ensureDirectory(targetPath: string) {
    const dir = path.dirname(targetPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }
}
