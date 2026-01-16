export interface TransformerConfig {
  numLayers: number;
  dModel: number;
  numHeads: number;
  dff: number;
  inputVocabSize: number;
  targetVocabSize: number;
  maximumPositionEncoding: number;
  dropoutRate: number;
  layerNormEpsilon: number;
}

export const DEFAULT_TRANSFORMER_CONFIG: TransformerConfig = {
  numLayers: 4,
  dModel: 256,
  numHeads: 8,
  dff: 1024,
  inputVocabSize: 4096,
  targetVocabSize: 4096,
  maximumPositionEncoding: 256,
  dropoutRate: 0.1,
  layerNormEpsilon: 1e-6,
};
