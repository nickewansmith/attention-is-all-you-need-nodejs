import { Injectable } from '@nestjs/common';

const EPSILON = 1e-9;

@Injectable()
export class EvaluationService {
  computeBleu(references: string[], hypotheses: string[], maxOrder = 4): number {
    if (references.length === 0 || references.length !== hypotheses.length) {
      return 0;
    }

    const precisions: number[] = [];
    for (let order = 1; order <= maxOrder; order += 1) {
      let matchCount = 0;
      let totalCount = 0;

      references.forEach((reference, idx) => {
        const refNgrams = this.countNgrams(reference, order);
        const hypNgrams = this.countNgrams(hypotheses[idx], order);
        hypNgrams.forEach((count, key) => {
          const referenceCount = refNgrams.get(key) ?? 0;
          matchCount += Math.min(count, referenceCount);
          totalCount += count;
        });
      });

      if (totalCount === 0) {
        precisions.push(1);
      } else {
        precisions.push(matchCount / totalCount);
      }
    }

    const logPrecision = precisions.reduce((sum, precision) => sum + Math.log(precision + EPSILON), 0) / maxOrder;
    const geoMean = Math.exp(logPrecision);

    const hypLength = hypotheses.reduce((sum, text) => sum + this.tokenize(text).length, 0);
    const refLength = references.reduce((sum, text) => sum + this.tokenize(text).length, 0);
    const brevityPenalty = hypLength > refLength ? 1 : Math.exp(1 - refLength / Math.max(hypLength, 1));

    return brevityPenalty * geoMean;
  }

  private countNgrams(text: string, order: number) {
    const tokens = this.tokenize(text);
    const counts = new Map<string, number>();
    if (tokens.length === 0) {
      return counts;
    }

    for (let i = 0; i <= tokens.length - order; i += 1) {
      const ngram = tokens.slice(i, i + order).join(' ');
      counts.set(ngram, (counts.get(ngram) ?? 0) + 1);
    }
    return counts;
  }

  private tokenize(text: string) {
    return text
      .trim()
      .split(/\s+/)
      .filter((token) => token.length > 0);
  }
}
