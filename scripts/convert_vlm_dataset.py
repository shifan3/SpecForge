#!/usr/bin/env python3
"""
Convert VLM dataset to SpecForge format.

This script filters out multi-image entries and converts the dataset
to the format expected by SpecForge's VLM preprocessing:
- Top-level 'image' field (single image path)
- 'conversations' field with simple string content

Supports multiple input formats:
- Format 1: 'conversations' field with [{role, content}, ...]
- Format 2: 'messages' field with [{role, content}, ...] (e.g., full_page_reader dataset)

Usage:
    python convert_vlm_dataset.py --input /path/to/dataset.json --output /path/to/output.jsonl
"""

import argparse
import json
import os


def convert_dataset(input_path: str, output_path: str, keep_multi_image: bool = False, max_entries: int = None):
    """
    Convert VLM dataset to SpecForge format.

    Args:
        input_path: Path to input dataset (JSON or JSONL)
        output_path: Path to output JSONL file
        keep_multi_image: If True, keep only the first image from multi-image entries
                         If False, skip multi-image entries entirely
        max_entries: Maximum number of entries to convert
    """
    # Read source dataset
    if input_path.endswith('.jsonl'):
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    print(f"Total entries: {len(data)}")

    

    # Filter and convert
    converted = []
    skipped_multi_image = 0
    skipped_no_image = 0
    kept_first_image = 0
    skipped_too_long = 0
    for entry in data:
        images = entry.get('images', [])

        # Handle entries with no images
        if len(images) == 0:
            skipped_no_image += 1
            continue

        # Handle multi-image entries
        if len(images) > 1:
            if keep_multi_image:
                kept_first_image += 1
            else:
                skipped_multi_image += 1
                continue

        # Extract single image (first one if multi-image and keep_multi_image=True)
        image_path = images[0]

        # Convert conversations to simple format
        # SpecForge expects: conversations = [{role, content(string)}, ...]
        # Support both 'conversations' and 'messages' fields
        source_messages = entry.get('conversations') or entry.get('messages', [])
        new_conversations = []
        brk = False
        for msg in source_messages:
            role = msg['role']
            content = msg['content']

            # If content is a list (multi-part), extract text parts
            if isinstance(content, list):
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                content_str = ''.join(text_parts)
            else:
                content_str = content
            content_str = content_str.replace('<image>', '')
            if len(content_str) >= 1024 and False:
                print(f"Skipped (too long): {len(content_str)}")
                skipped_too_long += 1
                brk = True
                break
            new_conversations.append({
                'role': role,
                'content': content_str
            })
        if brk:
            continue
        converted.append({
            'image': image_path,
            'conversations': new_conversations
        })
    
    if max_entries is not None:
        converted = converted[:max_entries]

    print(f"Skipped (multi-image): {skipped_multi_image}")
    print(f"Skipped (no image): {skipped_no_image}")
    print(f"Skipped (too long): {skipped_too_long}")
    if keep_multi_image:
        print(f"Kept first image only: {kept_first_image}")
    print(f"Converted entries: {len(converted)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save as JSONL
    with open(output_path, 'w') as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nSaved to {output_path}")

    # Show sample
    if converted:
        print("\nSample entry:")
        print(json.dumps(converted[0], indent=2, ensure_ascii=False)[:500])

    return len(converted)


def main():
    parser = argparse.ArgumentParser(description='Convert VLM dataset to SpecForge format')
    parser.add_argument('--input', '-i', required=True, help='Input dataset path (JSON or JSONL)')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL path')
    parser.add_argument('--keep-multi-image', action='store_true',
                       help='Keep multi-image entries (use first image only)')
    parser.add_argument('--max-entries', type=int, default=None, help='Maximum number of entries to convert')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    count = convert_dataset(args.input, args.output, args.keep_multi_image, args.max_entries)

    if count == 0:
        print("Warning: No entries were converted!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
