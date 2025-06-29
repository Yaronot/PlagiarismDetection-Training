#!/usr/bin/env python3
"""
Plagiarism Data Inspector - Show real examples of model inputs
Usage: python inspect_plagiarism_data.py --corpus_path /path/to/corpus --csv_file mappings.csv [options]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import random
from tqdm import tqdm


class PlagiarismDataInspector:
    """Inspector class to examine plagiarism detection data"""

    def __init__(self, csv_file: str, corpus_path: str):
        self.csv_file = csv_file
        self.corpus_path = corpus_path

    def extract_passages_python(self, source_id: str, suspicious_id: str,
                                source_offset: int, source_length: int,
                                suspicious_offset: int, suspicious_length: int) -> Tuple[str, str]:
        """Extract text passages from corpus files"""
        try:
            # Calculate part numbers
            source_num = int(source_id.split('document')[1].split('.')[0])
            source_part = (source_num - 1) // 500 + 1

            suspicious_num = int(suspicious_id.split('document')[1])
            suspicious_part = (suspicious_num - 1) // 500 + 1

            # Build file paths
            source_file = f"{self.corpus_path}/source-document/part{source_part}/{source_id}"
            suspicious_file = f"{self.corpus_path}/suspicious-document/part{suspicious_part}/{suspicious_id}.txt"

            # Read and extract passages
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_text = f.read()
                source_passage = source_text[source_offset:source_offset + source_length]

            with open(suspicious_file, 'r', encoding='utf-8', errors='ignore') as f:
                suspicious_text = f.read()
                suspicious_passage = suspicious_text[suspicious_offset:suspicious_offset + suspicious_length]

            return source_passage.strip(), suspicious_passage.strip()

        except Exception as e:
            print(f"Error extracting passages: {e}")
            return None, None

    def inspect_raw_data(self, num_examples: int = 5) -> List[Dict]:
        """Inspect raw data from CSV and show examples"""
        print("=" * 60)
        print("INSPECTING RAW CSV DATA")
        print("=" * 60)

        df = pd.read_csv(self.csv_file)
        print(f"Total rows in CSV: {len(df)}")
        print(f"CSV columns: {list(df.columns)}")
        print("\nFirst few rows of CSV:")
        print(df.head())

        print(f"\nExamining {num_examples} random examples...")
        examples = []

        # Sample random rows
        sample_indices = random.sample(range(len(df)), min(num_examples, len(df)))

        for i, idx in enumerate(sample_indices):
            row = df.iloc[idx]
            print(f"\n{'=' * 40}")
            print(f"EXAMPLE {i + 1}")
            print(f"{'=' * 40}")
            print(f"Source ID: {row['source_id']}")
            print(f"Suspicious ID: {row['suspicious_id']}")
            print(f"Source offset: {row['source_offset']}")
            print(f"Source length: {row['source_length']}")
            print(f"Suspicious offset: {row['suspicious_offset']}")
            print(f"Suspicious length: {row['suspicious_length']}")
            print(f"Obfuscation: {row['obfuscation']}")

            # Extract actual passages
            source_passage, suspicious_passage = self.extract_passages_python(
                row['source_id'], row['suspicious_id'],
                int(row['source_offset']), int(row['source_length']),
                int(row['suspicious_offset']), int(row['suspicious_length'])
            )

            if source_passage and suspicious_passage:
                print(f"\nSOURCE PASSAGE ({len(source_passage)} chars):")
                print("-" * 50)
                print(source_passage[:500] + "..." if len(source_passage) > 500 else source_passage)

                print(f"\nSUSPICIOUS PASSAGE ({len(suspicious_passage)} chars):")
                print("-" * 50)
                print(suspicious_passage[:500] + "..." if len(suspicious_passage) > 500 else suspicious_passage)

                examples.append({
                    'source_id': row['source_id'],
                    'suspicious_id': row['suspicious_id'],
                    'source_passage': source_passage,
                    'suspicious_passage': suspicious_passage,
                    'obfuscation': row['obfuscation'],
                    'source_length': len(source_passage),
                    'suspicious_length': len(suspicious_passage)
                })
            else:
                print("\nFailed to extract passages!")

        return examples

    def analyze_token_distribution(self, examples: List[Dict], model_name: str = 'bert-base-uncased') -> None:
        """Analyze token length distribution across all examples"""
        print("\n" + "=" * 60)
        print("TOKEN LENGTH DISTRIBUTION ANALYSIS")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        source_lengths = []
        suspicious_lengths = []

        print("Analyzing token lengths for all examples...")
        for example in tqdm(examples):
            source_tokens = tokenizer(example['source_passage'])['input_ids']
            suspicious_tokens = tokenizer(example['suspicious_passage'])['input_ids']
            source_lengths.append(len(source_tokens))
            suspicious_lengths.append(len(suspicious_tokens))

        source_lengths = np.array(source_lengths)
        suspicious_lengths = np.array(suspicious_lengths)

        print(f"\nSOURCE PASSAGES:")
        print(f"  Mean tokens: {np.mean(source_lengths):.1f}")
        print(f"  Median tokens: {np.median(source_lengths):.1f}")
        print(f"  95th percentile: {np.percentile(source_lengths, 95):.1f}")
        print(f"  Max tokens: {np.max(source_lengths)}")
        print(f"  Passages > 512 tokens: {np.sum(source_lengths > 512)} ({np.mean(source_lengths > 512) * 100:.1f}%)")
        print(
            f"  Passages > 1024 tokens: {np.sum(source_lengths > 1024)} ({np.mean(source_lengths > 1024) * 100:.1f}%)")

        print(f"\nSUSPICIOUS PASSAGES:")
        print(f"  Mean tokens: {np.mean(suspicious_lengths):.1f}")
        print(f"  Median tokens: {np.median(suspicious_lengths):.1f}")
        print(f"  95th percentile: {np.percentile(suspicious_lengths, 95):.1f}")
        print(f"  Max tokens: {np.max(suspicious_lengths)}")
        print(
            f"  Passages > 512 tokens: {np.sum(suspicious_lengths > 512)} ({np.mean(suspicious_lengths > 512) * 100:.1f}%)")
        print(
            f"  Passages > 1024 tokens: {np.sum(suspicious_lengths > 1024)} ({np.mean(suspicious_lengths > 1024) * 100:.1f}%)")

        # Calculate truncation impact
        avg_source_loss = np.mean(np.maximum(0, source_lengths - 512))
        avg_suspicious_loss = np.mean(np.maximum(0, suspicious_lengths - 512))

        print(f"\nTRUNCATION IMPACT (at 512 tokens):")
        print(f"  Average tokens lost per source passage: {avg_source_loss:.1f}")
        print(f"  Average tokens lost per suspicious passage: {avg_suspicious_loss:.1f}")
        print(f"  Total information loss: {(avg_source_loss + avg_suspicious_loss) / 2:.1f} tokens per pair")

    def inspect_tokenized_data(self, examples: List[Dict], model_name: str = 'bert-base-uncased',
                               max_length: int = 512) -> None:
        """Show how the data looks after tokenization"""
        print("\n" + "=" * 60)
        print("INSPECTING TOKENIZED DATA")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, example in enumerate(examples[:3]):  # Show first 3 examples
            print(f"\n{'=' * 40}")
            print(f"TOKENIZED EXAMPLE {i + 1}")
            print(f"{'=' * 40}")

            # Tokenize both passages
            source_tokens = tokenizer(
                example['source_passage'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            suspicious_tokens = tokenizer(
                example['suspicious_passage'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            print(f"Source passage tokenized:")
            print(f"  - Input IDs shape: {source_tokens['input_ids'].shape}")
            print(f"  - Attention mask shape: {source_tokens['attention_mask'].shape}")
            print(f"  - Number of actual tokens (non-padding): {source_tokens['attention_mask'].sum().item()}")

            print(f"\nSuspicious passage tokenized:")
            print(f"  - Input IDs shape: {suspicious_tokens['input_ids'].shape}")
            print(f"  - Attention mask shape: {suspicious_tokens['attention_mask'].shape}")
            print(f"  - Number of actual tokens (non-padding): {suspicious_tokens['attention_mask'].sum().item()}")

            # Show first few tokens decoded
            source_ids = source_tokens['input_ids'][0]
            suspicious_ids = suspicious_tokens['input_ids'][0]

            print(f"\nFirst 10 source tokens:")
            first_10_source = source_ids[:10]
            print(f"  Token IDs: {first_10_source.tolist()}")
            print(f"  Decoded: {tokenizer.decode(first_10_source, skip_special_tokens=False)}")

            print(f"\nFirst 10 suspicious tokens:")
            first_10_suspicious = suspicious_ids[:10]
            print(f"  Token IDs: {first_10_suspicious.tolist()}")
            print(f"  Decoded: {tokenizer.decode(first_10_suspicious, skip_special_tokens=False)}")

            # Show how much was truncated
            source_full_tokens = tokenizer(example['source_passage'])['input_ids']
            suspicious_full_tokens = tokenizer(example['suspicious_passage'])['input_ids']

            print(f"\nTruncation info:")
            print(f"  Source: {len(source_full_tokens)} -> {max_length} tokens")
            print(f"  Suspicious: {len(suspicious_full_tokens)} -> {max_length} tokens")
            if len(source_full_tokens) > max_length:
                print(f"  Source was truncated by {len(source_full_tokens) - max_length} tokens")
            if len(suspicious_full_tokens) > max_length:
                print(f"  Suspicious was truncated by {len(suspicious_full_tokens) - max_length} tokens")

    def create_model_input_example(self, examples: List[Dict], model_name: str = 'bert-base-uncased',
                                   max_length: int = 512) -> Dict:
        """Create an exact example of what the model receives as input"""
        print("\n" + "=" * 60)
        print("MODEL INPUT EXAMPLE")
        print("=" * 60)

        if not examples:
            print("No examples available!")
            return {}

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        example = examples[0]  # Use first example

        print(f"Creating model input for:")
        print(f"  Source: {example['source_passage'][:100]}...")
        print(f"  Suspicious: {example['suspicious_passage'][:100]}...")

        # This mimics exactly what PlagiarismDataset.__getitem__ returns
        encoding1 = tokenizer(
            example['source_passage'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        encoding2 = tokenizer(
            example['suspicious_passage'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        model_input = {
            'input_ids_1': encoding1['input_ids'].flatten(),
            'attention_mask_1': encoding1['attention_mask'].flatten(),
            'input_ids_2': encoding2['input_ids'].flatten(),
            'attention_mask_2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(1.0, dtype=torch.float)  # Positive example
        }

        print(f"\nMODEL INPUT TENSOR SHAPES:")
        for key, tensor in model_input.items():
            print(f"  {key}: {tensor.shape} (dtype: {tensor.dtype})")

        print(f"\nSAMPLE VALUES:")
        print(f"  input_ids_1[:20]: {model_input['input_ids_1'][:20].tolist()}")
        print(f"  attention_mask_1[:20]: {model_input['attention_mask_1'][:20].tolist()}")
        print(f"  input_ids_2[:20]: {model_input['input_ids_2'][:20].tolist()}")
        print(f"  attention_mask_2[:20]: {model_input['attention_mask_2'][:20].tolist()}")
        print(f"  label: {model_input['label'].item()}")

        return model_input

    def analyze_obfuscation_types(self) -> None:
        """Analyze different types of obfuscation in the dataset"""
        print("\n" + "=" * 60)
        print("OBFUSCATION TYPE ANALYSIS")
        print("=" * 60)

        df = pd.read_csv(self.csv_file)
        obfuscation_counts = df['obfuscation'].value_counts()

        print("Obfuscation types in dataset:")
        for obf_type, count in obfuscation_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {obf_type}: {count} instances ({percentage:.1f}%)")

        print(f"\nTotal instances: {len(df)}")

        # Show examples of each obfuscation type
        print(f"\nExamples by obfuscation type:")
        for obf_type in obfuscation_counts.index[:3]:  # Show top 3 types
            print(f"\n{'-' * 30}")
            print(f"OBFUSCATION TYPE: {obf_type}")
            print(f"{'-' * 30}")

            examples = df[df['obfuscation'] == obf_type].head(1)
            for _, row in examples.iterrows():
                source_passage, suspicious_passage = self.extract_passages_python(
                    row['source_id'], row['suspicious_id'],
                    int(row['source_offset']), int(row['source_length']),
                    int(row['suspicious_offset']), int(row['suspicious_length'])
                )

                if source_passage and suspicious_passage:
                    print(f"Source: {source_passage[:200]}...")
                    print(f"Suspicious: {suspicious_passage[:200]}...")

    def save_examples_to_json(self, examples: List[Dict], output_file: str = "plagiarism_examples.json"):
        """Save examples to JSON file for later inspection"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(examples)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Inspect plagiarism detection data')
    parser.add_argument('--corpus_path', required=True, help='Path to corpus directory')
    parser.add_argument('--csv_file', required=True, help='Path to plagiarism mappings CSV')
    parser.add_argument('--model_name', default='bert-base-uncased', help='BERT model name for tokenization')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to inspect')
    parser.add_argument('--save_json', help='Save examples to JSON file')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus path {args.corpus_path} does not exist")
        sys.exit(1)

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} does not exist")
        sys.exit(1)

    # Set random seed for reproducible examples
    random.seed(42)

    print("Plagiarism Detection Data Inspector")
    print("=" * 60)
    print(f"Corpus path: {args.corpus_path}")
    print(f"CSV file: {args.csv_file}")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")

    # Initialize inspector
    inspector = PlagiarismDataInspector(args.csv_file, args.corpus_path)

    # Inspect raw data
    examples = inspector.inspect_raw_data(args.num_examples)

    if examples:
        # Analyze token distribution first (gives context for truncation)
        inspector.analyze_token_distribution(examples, args.model_name)

        # Inspect tokenized data
        inspector.inspect_tokenized_data(examples, args.model_name, args.max_length)

        # Create model input example
        inspector.create_model_input_example(examples, args.model_name, args.max_length)

        # Analyze obfuscation types
        inspector.analyze_obfuscation_types()

        # Save examples if requested
        if args.save_json:
            inspector.save_examples_to_json(examples, args.save_json)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()