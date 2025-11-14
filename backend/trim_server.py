#!/usr/bin/env python3
"""
Script to trim server.py by removing multi-stage endpoints and functions.
"""

import re

def find_function_end(lines, start_idx):
    """Find the end of a function definition."""
    indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())

    for i in range(start_idx + 1, len(lines)):
        line = lines[i]

        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('#'):
            continue

        # Check if we've reached a new top-level definition
        current_indent = len(line) - len(line.lstrip())

        # If we're back to the same or lesser indent and it's a decorator or definition
        if current_indent <= indent_level and (line.strip().startswith('@') or
                                                line.strip().startswith('def ') or
                                                line.strip().startswith('async def ') or
                                                line.strip().startswith('class ')):
            return i

    return len(lines)

def remove_endpoints(input_file, output_file):
    """Remove unwanted endpoints from server.py"""

    # Endpoints to remove (function names)
    endpoints_to_remove = [
        'def generate(req: GenerateRequest)',  # Original multi-stage generate (NOT generate-fast!)
        'def generate_cumulative_tags(',
        'def cumulative_tags_feedback(',
        'def cumulative_tags_next_stage(',
        'def cumulative_tags_select_concept(',
        'def generate_final_progressive(',
        'def final_progressive_feedback(',
        'def get_mode_selection(',
        'def generate_final(mode:',  # Specific to final mode selection
        'def generate_final_cumulative_tags(',
        'def generate_stage_refinement(',
        'def run_final_mode1(',
        'def run_final_mode2(',
        'def run_final_mode3(',
        'def run_final_mode4(',
        'def run_final_mode1_optimized(',
        'def run_final_mode2_optimized(',
        'def run_final_mode3_optimized(',
        'def run_final_mode4_optimized(',
        'def handle_parallel_to_final(',
        'def safe_generate_images(',  # Only used by removed endpoints
    ]

    # Functions to KEEP (do NOT remove even if they contain keywords)
    functions_to_keep = [
        'def generate_fast(',  # Mode 1 entry point - KEEP!
        'async def upload_session(',  # Mode 2 entry point - KEEP!
        'def list_sessions(',  # Mode 2 support - KEEP!
        'def load_session(',  # Mode 2 support - KEEP!
        'def load_stage_data(',  # Mode 2 support - KEEP!
    ]

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Track sections to remove
    sections_to_remove = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # First check if this is a function we want to KEEP
        should_keep = False
        for keep_func in functions_to_keep:
            if keep_func in line:
                should_keep = True
                break

        # If it's a keep function, skip it
        if should_keep:
            i += 1
            continue

        # Check if this line starts a function we want to remove
        should_remove = False
        for endpoint in endpoints_to_remove:
            if endpoint in line:
                should_remove = True
                break

        if should_remove:
            # Find the start of this function (including decorators)
            start_idx = i
            # Look backwards for decorators
            while start_idx > 0 and (lines[start_idx - 1].strip().startswith('@') or
                                     lines[start_idx - 1].strip() == ''):
                start_idx -= 1

            # Find the end of this function
            end_idx = find_function_end(lines, i)

            sections_to_remove.append((start_idx, end_idx))
            print(f"Marking for removal: lines {start_idx + 1} to {end_idx} ({line.strip()[:60]}...)")
            i = end_idx
        else:
            i += 1

    # Remove sections in reverse order to maintain line numbers
    sections_to_remove.reverse()
    for start, end in sections_to_remove:
        del lines[start:end]

    # Write the trimmed file
    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"\nRemoved {len(sections_to_remove)} functions/endpoints")
    print(f"Original lines: {len(open(input_file).readlines())}")
    print(f"New lines: {len(lines)}")
    print(f"Lines removed: {len(open(input_file).readlines()) - len(lines)}")

if __name__ == '__main__':
    input_file = 'server.py'
    output_file = 'server_trimmed.py'

    remove_endpoints(input_file, output_file)
    print(f"\nTrimmed server saved to {output_file}")
    print("Review the file, then rename it to server.py if it looks good")
