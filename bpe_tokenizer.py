import collections
import re
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

class SanskritBPETokenizer:
    def __init__(self, vocab_size: int = 4999):
        self.vocab_size = vocab_size # this vocab is vocab of tokens; we are going to have these many unique tokens
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        self.compression_ratio = None
        self.pattern = r'[\u0900-\u097F\u1CD0-\u1CFF\s]+|[॥।ʼ.,!?()"]' #valid_chars_pattern
        
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    

    def merge(self, ids, pair, idx, reverse_map):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                reverse_map[idx] = reverse_map[pair[0]] + reverse_map[pair[1]]
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def decode_ids(self, ids, reverse_map):
        """Convert token IDs back to their character or merged string representation."""
        return [reverse_map[token] for token in ids]
    

    def fit(self, raw_texts: List[str]) -> None:
        texts = re.sub(f'[^{self.pattern}]', '', raw_texts)
        # Initialize vocabulary with characters
        base_vocab = {char: idx for idx, char in enumerate(set(texts), start=0)}
        base_reverse_vocab = {idx: char for char, idx in base_vocab.items()}

        tokens = [base_vocab[char] for char in texts]

        num_merges = self.vocab_size - len(base_vocab)

        ids = tokens[:]
        merges = {}
        reverse_merges = base_reverse_vocab.copy()  # Start with the base vocabulary

        for i in tqdm(range(num_merges)):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = max(reverse_merges.keys()) + 1
            ids = self.merge(ids, pair, idx, reverse_merges)
            merges[pair] = idx

        self.merges = merges
        self.vocab = {v:k for k,v in reverse_merges.items()}
        self.inverse_vocab = reverse_merges
        self.compression_ratio = round(len(tokens)/len(ids),1)
    
    def merge_tokens(self, ids, positions=None):
        """Helper method to apply merges consistently"""
        i = 0
        while i < len(ids) - 1:
            merged = False
            for pair, merge_idx in self.merges.items():
                if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    if positions is not None:
                        # Update positions if tracking them
                        merged_start = positions[i][0]
                        merged_end = positions[i + 1][1]
                        positions[i] = (merged_start, merged_end)
                        positions.pop(i + 1)
                    
                    # Update ids
                    ids[i] = merge_idx
                    ids.pop(i + 1)
                    
                    merged = True
                    break
            if not merged:
                i += 1
        return ids

    def encode(self, raw_texts: str) -> List[int]:
        texts = re.sub(f'[^{self.pattern}]', '', raw_texts)
        
        # Use the existing vocabulary instead of creating a new one
        ids = []
        for char in texts:
            # Find the character in the inverse_vocab
            char_id = None
            for id, token in self.inverse_vocab.items():
                if token == char:
                    char_id = id
                    break
            if char_id is not None:
                ids.append(char_id)
            else:
                # Handle unknown characters
                continue
        
        # Apply merges using the shared method
        return self.merge_tokens(ids)
    
    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inverse_vocab[id] for id in ids)  
    
    def encode_with_positions(self, raw_texts: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        texts = re.sub(f'[^{self.pattern}]', '', raw_texts)
        
        # Initialize with character positions
        ids = []
        positions = []  # List of (start, end) positions
        current_pos = 0
        
        # First pass: encode individual characters and track their positions
        for char in texts:
            char_id = None
            for id, token in self.inverse_vocab.items():
                if token == char:
                    char_id = id
                    break
            if char_id is not None:
                ids.append(char_id)
                positions.append((current_pos, current_pos + len(char)))
            current_pos += len(char)
        
        # Apply merges using the shared method
        self.merge_tokens(ids, positions)
        return ids, positions