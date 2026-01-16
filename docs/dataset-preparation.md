# Dataset & Tokenizer Preparation Guide

This guide walks through preparing a real translation dataset and tokenizer for the NestJS Transformer implementation. Use it to replace the toy dataset with something production-worthy (e.g., IWSLT or WMT).

## 1. Obtain a Parallel Corpus

Choose a dataset with aligned source/target sentences:

- [IWSLT TED Talks](https://wit3.fbk.eu/) – smaller, good for prototyping.
- [WMT News Translation](https://www.statmt.org/wmt23/translation-task.html) – large-scale.

Download the source (`train.en`) and target (`train.de`) files, ensuring they have the same number of lines.

```
mkdir -p data
curl -o data/train.en https://example.com/path/to/train.en
curl -o data/train.de https://example.com/path/to/train.de
```

(Optional) Split into train/val/test subsets using standard tools (Moses scripts, sacreBLEU, etc.).

## 2. Clean the Data

Common preprocessing steps:

- Normalize Unicode, strip control characters.
- Lowercase or truecase depending on target style.
- Remove very long sentences (e.g., > 200 tokens) or empty lines.

You can use `sacremoses`, `sentencepiece`, or custom Python scripts. Example Python snippet:

```python
from pathlib import Path

def clean_line(line):
    line = line.strip()
    return line.replace('\u2028', ' ')

def clean_file(path):
    lines = [clean_line(line) for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]

eng = clean_file('data/train.en')
deu = clean_file('data/train.de')
assert len(eng) == len(deu)
```

## 3. Train or Download a Tokenizer

The project loads Hugging Face `tokenizer.json` files. You can either:

### Option A – Download an existing tokenizer

Many Hugging Face models provide `tokenizer.json`. Example:

```
curl -o tokenizer/tokenizer.json https://huggingface.co/t5-small/resolve/main/tokenizer.json
curl -o tokenizer/tokenizer_config.json https://huggingface.co/t5-small/resolve/main/tokenizer_config.json
```

Update env vars:

```
TOKENIZER_JSON_PATH=./tokenizer/tokenizer.json
TOKENIZER_CONFIG_PATH=./tokenizer/tokenizer_config.json
```

### Option B – Train SentencePiece

```
pip install sentencepiece
python - <<'PY'
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/train.en,data/train.de',
    model_prefix='tokenizer/spm',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'
)
PY
```

Then convert to HF JSON:

```
pip install tokenizers transformers
python - <<'PY'
from tokenizers import SentencePieceBPETokenizer
from pathlib import Path

model = SentencePieceBPETokenizer(
    vocab_file='tokenizer/spm.vocab',
    merges_file='tokenizer/spm.model'
)
model.save('tokenizer', 'custom')
PY
```

Ensure special tokens exist (pad/start/end/unk). Override IDs via env vars if necessary (`TOKENIZER_PAD_ID`, etc.).

## 4. Configure the App

Set env vars in `.env` or export them:

```
DATASET_SOURCE_PATH=./data/train.en
DATASET_TARGET_PATH=./data/train.de
DATASET_MAX_SAMPLES=200000  # optional subset
TOKENIZER_JSON_PATH=./tokenizer/custom-tokenizer.json
TOKENIZER_CONFIG_PATH=./tokenizer/custom-tokenizer_config.json
CHECKPOINT_SAVE_PATH=./checkpoints/latest.json
```

Restart the NestJS server or rerun the CLI training command. The `TrainingService` will ingest the new dataset, log BLEU/perplexity samples, and save checkpoints after each epoch.

## 5. Sanity Checks

Before launching long training runs:

1. Use the CLI inference to ensure tokenization + checkpoints work:
   ```
   npm run cli:infer "hello world"
   ```
2. Run a single-epoch training pass:
   ```
   npm run cli:train 1 64
   ```
3. Inspect `checkpoints/latest.json` to confirm weights/optimizer state were saved.

## 6. Troubleshooting

- **Mismatched line counts** – ensure source/target files have the same number of lines. Use `wc -l` or custom scripts.
- **Tokenizer missing special tokens** – set `TOKENIZER_*_ID` env vars or insert tokens via Hugging Face tokenizers API.
- **Out-of-memory** – reduce `DATASET_MAX_SAMPLES`, `maxLength`, or `TRANSFORMER_D_MODEL`. You can also increase batch size gradually.

With these steps complete, you’re ready to conduct real experiments using the NestJS Transformer stack.
