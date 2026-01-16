import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Tokenizer } from '@huggingface/tokenizers';
import { existsSync, readFileSync } from 'fs';
import { dirname, join } from 'path';

const DEFAULT_MAX_LENGTH = 64;

@Injectable()
export class TokenizerService {
  private readonly logger = new Logger(TokenizerService.name);
  private hfTokenizer?: Tokenizer;
  private readonly usingExternalModel: boolean;
  private readonly defaultMaxLength: number;

  private startToken: string;
  private endToken: string;
  private padToken: string;
  private unkToken: string;

  private startTokenIdValue = 0;
  private endTokenIdValue = 0;
  private padTokenIdValue = 0;
  private unkTokenIdValue = 0;

  private externalVocabSize = 0;

  private wordToId = new Map<string, number>();
  private idToWord = new Map<number, string>();

  constructor(private readonly configService: ConfigService) {
    this.startToken = this.configService.get<string>('TOKENIZER_START_TOKEN') ?? '<s>';
    this.endToken = this.configService.get<string>('TOKENIZER_END_TOKEN') ?? '</s>';
    this.padToken = this.configService.get<string>('TOKENIZER_PAD_TOKEN') ?? '<pad>';
    this.unkToken = this.configService.get<string>('TOKENIZER_UNK_TOKEN') ?? '<unk>';
    this.defaultMaxLength = this.parseNumberConfig('TOKENIZER_MAX_LENGTH') ?? DEFAULT_MAX_LENGTH;

    this.usingExternalModel = this.tryLoadExternalTokenizer();

    if (!this.usingExternalModel) {
      this.bootstrapVocabulary();
      this.startTokenIdValue = this.wordToId.get(this.startToken) ?? 0;
      this.endTokenIdValue = this.wordToId.get(this.endToken) ?? 0;
      this.padTokenIdValue = this.wordToId.get(this.padToken) ?? 0;
      this.unkTokenIdValue = this.wordToId.get(this.unkToken) ?? 0;
    }
  }

  get vocabSize(): number {
    if (this.usingExternalModel) {
      return this.externalVocabSize;
    }
    return this.wordToId.size;
  }

  get startTokenId(): number {
    return this.startTokenIdValue;
  }

  get endTokenId(): number {
    return this.endTokenIdValue;
  }

  get padTokenId(): number {
    return this.padTokenIdValue;
  }

  get unkTokenId(): number {
    return this.unkTokenIdValue;
  }

  usesExternalModel(): boolean {
    return this.usingExternalModel;
  }

  fitOnTexts(texts: string[]) {
    if (this.usingExternalModel) {
      return;
    }
    texts.forEach((text) => this.addTokens(this.tokenize(text)));
  }

  encode(text: string, maxLength = this.defaultMaxLength): number[] {
    const targetLength = Math.max(2, maxLength); // ensure we can place start/end tokens

    if (this.usingExternalModel && this.hfTokenizer) {
      const encoding = this.hfTokenizer.encode(text, { add_special_tokens: false });
      const ids = [this.startTokenId, ...encoding.ids, this.endTokenId];
      return this.applyPadding(ids, targetLength);
    }

    const tokens = this.tokenize(text);
    const ids = [
      this.startTokenId,
      ...tokens.map((token) => this.wordToId.get(token) ?? this.unkTokenId),
      this.endTokenId,
    ];
    return this.applyPadding(ids, targetLength);
  }

  decode(tokenIds: number[]): string {
    const filtered = tokenIds.filter(
      (id) => id !== this.padTokenId && id !== this.startTokenId && id !== this.endTokenId,
    );

    if (this.usingExternalModel && this.hfTokenizer) {
      return this.hfTokenizer.decode(filtered, { skip_special_tokens: true }).trim();
    }

    const tokens = filtered.map((id) => this.idToWord.get(id) ?? this.unkToken);
    return tokens.join(' ');
  }

  tokenFromId(id: number): string {
    if (this.usingExternalModel && this.hfTokenizer) {
      return this.hfTokenizer.id_to_token(id) ?? this.unkToken;
    }
    return this.idToWord.get(id) ?? this.unkToken;
  }

  private applyPadding(ids: number[], maxLength: number) {
    if (ids.length > maxLength) {
      ids.splice(maxLength - 1);
      ids[maxLength - 1] = this.endTokenId;
    }

    while (ids.length < maxLength) {
      ids.push(this.padTokenId);
    }

    return ids;
  }

  private tryLoadExternalTokenizer() {
    const tokenizerPath = this.configService.get<string>('TOKENIZER_JSON_PATH');
    if (!tokenizerPath) {
      return false;
    }

    if (!existsSync(tokenizerPath)) {
      this.logger.warn(`Tokenizer file not found at ${tokenizerPath}. Falling back to basic tokenizer.`);
      return false;
    }

    try {
      const tokenizerJson = JSON.parse(readFileSync(tokenizerPath, 'utf8'));
      const tokenizerConfig = this.loadTokenizerConfig(tokenizerPath);
      const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig ?? {});
      this.externalVocabSize = this.deriveVocabSize(tokenizerJson);
      this.startTokenIdValue = this.resolveTokenId(tokenizer, 'TOKENIZER_START_ID', this.startToken);
      this.endTokenIdValue = this.resolveTokenId(tokenizer, 'TOKENIZER_END_ID', this.endToken);
      this.padTokenIdValue = this.resolveTokenId(tokenizer, 'TOKENIZER_PAD_ID', this.padToken);
      this.unkTokenIdValue = this.resolveTokenId(tokenizer, 'TOKENIZER_UNK_ID', this.unkToken);
      this.hfTokenizer = tokenizer;
      this.logger.log(`Loaded tokenizer from ${tokenizerPath}`);
      return true;
    } catch (error) {
      this.logger.error(`Failed to load tokenizer from ${tokenizerPath}: ${error}`);
      return false;
    }
  }

  private resolveTokenId(tokenizer: Tokenizer, envKey: string, token: string) {
    const envValue = this.parseNumberConfig(envKey);
    if (envValue !== undefined) {
      return envValue;
    }

    const id = tokenizer.token_to_id(token);
    if (id === undefined) {
      throw new Error(`Token ${token} missing from tokenizer. Provide ${envKey} in environment variables.`);
    }
    return id;
  }

  private loadTokenizerConfig(tokenizerPath: string) {
    const configuredPath = this.configService.get<string>('TOKENIZER_CONFIG_PATH');
    const candidatePath = configuredPath ?? join(dirname(tokenizerPath), 'tokenizer_config.json');
    if (!candidatePath || !existsSync(candidatePath)) {
      return undefined;
    }
    try {
      return JSON.parse(readFileSync(candidatePath, 'utf8'));
    } catch (error) {
      this.logger.warn(`Failed to parse tokenizer config at ${candidatePath}: ${error}`);
      return undefined;
    }
  }

  private deriveVocabSize(tokenizerJson: Record<string, unknown>) {
    const model = tokenizerJson?.['model'];
    if (model && typeof model === 'object') {
      const vocab = (model as Record<string, unknown>)['vocab'];
      if (Array.isArray(vocab)) {
        return vocab.length;
      }
      if (vocab && typeof vocab === 'object') {
        return Object.keys(vocab as Record<string, unknown>).length;
      }
      const vocabSize = (model as Record<string, unknown>)['vocabSize'];
      if (typeof vocabSize === 'number') {
        return vocabSize;
      }
    }
    return 0;
  }

  private parseNumberConfig(key: string) {
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

  private bootstrapVocabulary() {
    [this.padToken, this.startToken, this.endToken, this.unkToken].forEach((token) => this.registerToken(token));

    const seedWords = 'abcdefghijklmnopqrstuvwxyz'.split('').concat(['.', ',', '?', '!', "'", '"']);
    this.addTokens(seedWords);
  }

  private tokenize(text: string): string[] {
    const matches = text
      .toLowerCase()
      .trim()
      .match(/[\p{L}\p{N}'-]+|[^\s\p{L}\p{N}]/gu);
    if (!matches) {
      return [];
    }
    return matches;
  }

  private addTokens(tokens: string[]) {
    tokens.forEach((token) => this.registerToken(token));
  }

  private registerToken(token: string) {
    if (this.wordToId.has(token)) {
      return;
    }

    const nextId = this.wordToId.size;
    this.wordToId.set(token, nextId);
    this.idToWord.set(nextId, token);
  }
}
