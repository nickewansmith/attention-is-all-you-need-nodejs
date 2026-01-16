import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const GPU_BACKEND = '@tensorflow/tfjs-node-gpu';
const CPU_BACKEND = '@tensorflow/tfjs-node';

function hasCuda(): boolean {
  const nvidiaSmis = ['nvidia-smi', '/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi'];
  const binary = nvidiaSmis.find((path) => existsSync(path));
  if (!binary) {
    return false;
  }
  const result = spawnSync(binary, ['-L'], { stdio: 'ignore' });
  return result.status === 0;
}

export function loadTensorflowBackend() {
  if (process.env.TFJS_FORCE_CPU === '1') {
    // eslint-disable-next-line global-require, import/no-dynamic-require
    return require(CPU_BACKEND);
  }

  if (hasCuda()) {
    try {
      // eslint-disable-next-line global-require, import/no-dynamic-require
      return require(GPU_BACKEND);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn('Failed to load tfjs-node-gpu, falling back to CPU backend:', error);
    }
  }

  // eslint-disable-next-line global-require, import/no-dynamic-require
  return require(CPU_BACKEND);
}
