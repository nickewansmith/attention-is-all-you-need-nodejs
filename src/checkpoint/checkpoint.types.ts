export interface SerializedTensor {
  shape: number[];
  values: number[];
}

export type SerializedVariableMap = Record<string, SerializedTensor>;

export interface SerializedOptimizerWeight extends SerializedTensor {
  name: string;
}

export interface CheckpointPayload {
  createdAt: string;
  globalStep: number;
  variables: SerializedVariableMap;
  optimizer?: SerializedOptimizerWeight[];
}
