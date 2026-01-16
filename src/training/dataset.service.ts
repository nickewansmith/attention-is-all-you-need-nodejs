import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { existsSync, readFileSync } from 'fs';
import { TokenizerService } from '../common/tokenizer/tokenizer.service';

export interface ParallelTextExample {
  source: string;
  target: string;
}

export interface TrainingBatch {
  encoderInputs: number[][];
  decoderInputs: number[][];
}

const TOY_DATA: ParallelTextExample[] = [
  { source: 'translate english to german', target: 'übersetze englisch zu deutsch' },
  { source: 'attention is all you need', target: 'aufmerksamkeit ist alles was du brauchst' },
  { source: 'deep learning is fun', target: 'tiefes lernen macht spass' },
  { source: 'machine translation with transformers', target: 'maschinenübersetzung mit transformern' },
];

@Injectable()
export class DatasetService {
  private readonly logger = new Logger(DatasetService.name);
  private readonly dataset: ParallelTextExample[];

  constructor(
    private readonly tokenizer: TokenizerService,
    private readonly configService: ConfigService,
  ) {
    this.dataset = this.initializeDataset();

    if (!this.tokenizer.usesExternalModel()) {
      this.tokenizer.fitOnTexts(this.dataset.flatMap((example) => [example.source, example.target]));
    }
  }

  createBatches(batchSize: number, maxSeqLength: number): TrainingBatch[] {
    if (this.dataset.length === 0) {
      throw new Error('Dataset is empty. Provide DATASET_SOURCE_PATH and DATASET_TARGET_PATH.');
    }

    const batches: TrainingBatch[] = [];
    for (let i = 0; i < this.dataset.length; i += batchSize) {
      const slice = this.dataset.slice(i, i + batchSize);
      const encoderInputs = slice.map((example) => this.tokenizer.encode(example.source, maxSeqLength));
      const decoderInputs = slice.map((example) => this.tokenizer.encode(example.target, maxSeqLength));
      batches.push({ encoderInputs, decoderInputs });
    }

    return batches;
  }

  sample(count: number) {
    if (count <= 0) {
      return [];
    }
    return this.dataset.slice(0, Math.min(count, this.dataset.length));
  }

  private initializeDataset(): ParallelTextExample[] {
    const dataset = this.loadFromFiles();
    if (dataset.length > 0) {
      this.logger.log(`Loaded ${dataset.length} examples from configured dataset files.`);
      return dataset;
    }

    this.logger.warn('Dataset paths not configured; falling back to toy dataset.');
    return TOY_DATA;
  }

  private loadFromFiles(): ParallelTextExample[] {
    const sourcePath = this.configService.get<string>('DATASET_SOURCE_PATH');
    const targetPath = this.configService.get<string>('DATASET_TARGET_PATH');
    if (!sourcePath || !targetPath) {
      return [];
    }

    if (!existsSync(sourcePath) || !existsSync(targetPath)) {
      this.logger.warn(`Dataset files not found at ${sourcePath} / ${targetPath}`);
      return [];
    }

    const sourceLines = this.readLines(sourcePath);
    const targetLines = this.readLines(targetPath);
    const limit = Math.min(sourceLines.length, targetLines.length);
    if (limit === 0) {
      return [];
    }

    const maxSamples = this.parseNumber('DATASET_MAX_SAMPLES');
    const finalCount = maxSamples ? Math.min(limit, maxSamples) : limit;
    const dataset: ParallelTextExample[] = [];
    for (let i = 0; i < finalCount; i += 1) {
      dataset.push({ source: sourceLines[i], target: targetLines[i] });
    }
    return dataset;
  }

  private readLines(filePath: string) {
    return readFileSync(filePath, 'utf8')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  }

  private parseNumber(key: string) {
    const raw = this.configService.get<string>(key);
    if (raw === undefined || raw === null) {
      return undefined;
    }

    const parsed = Number(raw);
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid numeric configuration for ${key}: ${raw}`);
    }
    return parsed;
  }
}
