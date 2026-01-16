const { execSync } = require('child_process');
const fs = require('fs');
const hasCuda = () => {
  try {
    execSync('nvidia-smi -L', { stdio: 'ignore' });
    return true;
  } catch (err) {
    return false;
  }
};
const target = hasCuda() ? '@tensorflow/tfjs-node-gpu' : '@tensorflow/tfjs-node';
fs.writeFileSync('tf-backend.txt', target);
console.log(target);
