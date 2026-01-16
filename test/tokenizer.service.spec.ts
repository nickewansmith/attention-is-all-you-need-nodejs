import { ConfigService } from '@nestjs/config';
import { TokenizerService } from '../src/common/tokenizer/tokenizer.service';

describe('TokenizerService', () => {
  it('encodes and decodes roundtrip text', () => {
    const tokenizer = new TokenizerService(new ConfigService());
    tokenizer.fitOnTexts(['hello world']);
    const tokens = tokenizer.encode('hello world', 10);
    const decoded = tokenizer.decode(tokens);
    expect(decoded).toContain('hello');
    expect(decoded).toContain('world');
  });
});
