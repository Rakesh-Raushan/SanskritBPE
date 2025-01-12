# Sanskrit BPE Tokenizer

A byte pair encoding (BPE) tokenizer specifically designed for Sanskrit text processing.

## Features

- Vocabulary size: 4,999 tokens (under the 5,000 token limit)
- Compression ratio: 3.3
- Specialized for Sanskrit text processing
- Efficient encoding and decoding

## Usage
python
from bpe_tokenizer import SanskritBPETokenizer
Initialize tokenizer
tokenizer = SanskritBPETokenizer(vocab_size=4999)
Train on your data
tokenizer.fit(texts)
Encode text
encoded = tokenizer.encode("your sanskrit text here")
Decode tokens
decoded = tokenizer.decode(encoded)

### Example Texts

The repository includes two example Sanskrit texts:

**Example 1:**
```sanskrit
यथा ह्येकेन चक्रेण न रथस्य गतिर्भवेत्।
एवं परुषकारेण विना दैवं न सिद्ध्यति।।
```

**Example 2:**
```sanskrit
उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः! न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः !
```

## HuggingFace Space

You can try out the tokenizer in our [HuggingFace Space](your_space_link_here)

## Training Data

The tokenizer was trained on [describe your Sanskrit dataset here]
