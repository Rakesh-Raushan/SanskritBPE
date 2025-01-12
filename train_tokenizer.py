import pandas as pd
from pathlib import Path
from datasets import load_dataset
from bpe_tokenizer import SanskritBPETokenizer
import json, re
from typing import List

def load_sanskrit_data() -> List[str]:
    # # Load your Sanskrit dataset here
    # # This is a placeholder - you'll need to adapt this to your actual data source
    # texts = []
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     texts = f.readlines()

    ds = load_dataset("VinitT/Sanskrit-Llama")
    raw_text = "".join(ds['train']['input'])

    return raw_text

# def calculate_compression_ratio(original_texts: List[str], tokenizer: SanskritBPETokenizer) -> float:
#     original_size = sum(len(text.encode('utf-8')) for text in original_texts)
#     tokenized_size = sum(len(tokenizer.encode(text)) for text in original_texts)
#     return original_size / tokenized_size

def main():
    # Load data
    raw_text = load_sanskrit_data()
    
    # Initialize and train tokenizer
    tokenizer = SanskritBPETokenizer(vocab_size=4999)
    tokenizer.fit(raw_text)
    
    print(f"Compression ratio: {tokenizer.compression_ratio}")
    
    # Convert tuple keys to strings for JSON serialization
    serializable_merges = {f"{k[0]}|{k[1]}": v for k, v in tokenizer.merges.items()}
    
    # Save tokenizer vocabulary and merges
    with open('tokenizer_config.json', 'w', encoding='utf-8') as f:
        json.dump({
            'vocab': tokenizer.vocab,
            'merges': serializable_merges,
            'inverse_vocab': tokenizer.inverse_vocab
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 