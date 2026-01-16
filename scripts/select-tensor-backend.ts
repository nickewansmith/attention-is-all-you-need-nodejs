import * as os from 'node:os';

const tfGpu = '@tensorflow/tfjs-node-gpu';
const tfCpu = '@tensorflow/tfjs-node';

function detectCuda() {
  const platform = os.platform();
  if (platform !== 'linux' && platform !== 'darwin') {
    return null;
  }
  const nvidiaExists = ['nvidia-smi', '/usr/bin/nvidia-smi', '/bin/nvidia-smi'].some((binary) =>
    require('fs').existsSync(binary),
  );
  return nvidiaExists ? tfGpu : null;
}

const preferred = detectCuda();
process.stdout.write(preferred ?? tfCpu);
