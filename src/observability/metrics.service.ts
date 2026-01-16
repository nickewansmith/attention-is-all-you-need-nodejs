import { Injectable } from '@nestjs/common';
import { Counter, Histogram, register } from 'prom-client';

@Injectable()
export class MetricsService {
  private readonly inferenceCounter: Counter<string>;
  private readonly trainingCounter: Counter<string>;
  private readonly streamLatencyHistogram: Histogram<string>;

  constructor() {
    this.inferenceCounter = new Counter({
      name: 'inference_requests_total',
      help: 'Total number of inference requests handled',
    });

    this.trainingCounter = new Counter({
      name: 'training_steps_total',
      help: 'Total number of training optimization steps executed',
    });

    this.streamLatencyHistogram = new Histogram({
      name: 'translation_stream_duration_ms',
      help: 'Wall-clock duration (ms) for streaming translation requests',
      buckets: [50, 100, 250, 500, 1000, 2000, 5000, 10000],
      labelNames: ['strategy', 'throttle'],
    });
  }

  markInferenceRequest() {
    this.inferenceCounter.inc();
  }

  markTrainingStep() {
    this.trainingCounter.inc();
  }

  startStreamTimer(strategy: 'greedy' | 'beam', throttled: boolean) {
    const endTimer = this.streamLatencyHistogram.startTimer({
      strategy,
      throttle: throttled ? 'throttled' : 'unthrottled',
    });
    return () => endTimer();
  }

  async getMetrics() {
    return register.metrics();
  }

  getMetricsContentType() {
    return register.contentType;
  }
}
