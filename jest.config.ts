import type { Config } from 'jest';

const config: Config = {
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/test'],
  moduleFileExtensions: ['js', 'json', 'ts'],
  transform: {
    '^.+\\.(t|j)s$': 'ts-jest',
  },
  moduleNameMapper: {
    '^@app/(.*)$': '<rootDir>/src/$1',
    '^@core/(.*)$': '<rootDir>/src/transformer-core/$1',
    '^@training/(.*)$': '<rootDir>/src/training/$1',
    '^@api/(.*)$': '<rootDir>/src/api/$1',
    '^@tensor/(.*)$': '<rootDir>/src/tensor/$1',
  },
  collectCoverageFrom: ['src/**/*.(t|j)s'],
  coverageDirectory: 'coverage',
};

export default config;
