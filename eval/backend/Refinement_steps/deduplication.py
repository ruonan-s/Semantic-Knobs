"""
Stage 1: Deduplication
Remove exact/near-duplicate tag phrasings while preserving semantic distinctions.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from collections import Counter


@dataclass
class TagGroup:
    """A group of tags that are essentially the same phrase."""
    canonical: str              # The representative phrasing we'll use
    variations: list[str]       # All phrasings found (including canonical)
    frequency: int              # Total occurrence count across all variations


@dataclass 
class DeduplicationResult:
    """Result of the deduplication stage."""
    unique_tags: list[str]                    # List of canonical tags
    tag_groups: list[TagGroup]                # Full group information
    tag_mapping: dict[str, str]               # Maps any variation → canonical
    original_count: int                       # Input tag count
    deduplicated_count: int                   # Output tag count
    duplicates_removed: list[tuple[str, str]] # (removed_tag, merged_into)


def normalize_tag(tag: str) -> str:
    """Normalize tag for comparison."""
    tag = tag.lower().strip()
    tag = re.sub(r'\s+', ' ', tag)  # Collapse whitespace
    return tag


def are_duplicates(tag1: str, tag2: str, threshold: float = 0.88) -> bool:
    """
    Determine if two tags are duplicates (same concept, different phrasing).
    
    Conservative matching - only true duplicates, not similar concepts.
    
    Examples of DUPLICATES (should merge):
        - "natural light" ≈ "natural lighting" 
        - "open layout" ≈ "open floor plan"
        - "contemporary style" ≈ "contemporary aesthetics"
    
    Examples of NON-DUPLICATES (should keep separate):
        - "natural light" ≠ "bright atmosphere"
        - "hanging plants" ≠ "lush greenery"  
        - "cozy seating" ≠ "comfortable furniture"
    """
    norm1 = normalize_tag(tag1)
    norm2 = normalize_tag(tag2)
    
    # Exact match
    if norm1 == norm2:
        return True
    
    # One is substring with only generic suffix difference
    # e.g., "contemporary" vs "contemporary style"
    generic_suffixes = {
        'style', 'design', 'aesthetic', 'aesthetics', 
        'look', 'feel', 'vibe', 'theme', 'elements'
    }
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    # Check if difference is only generic words
    diff = words1.symmetric_difference(words2)
    if diff and diff.issubset(generic_suffixes):
        # Verify the core words match
        core1 = words1 - generic_suffixes
        core2 = words2 - generic_suffixes
        if core1 == core2:
            return True
    
    # High string similarity for typos and minor variations
    # e.g., "lighting" vs "light"
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Also check if one is contained in the other
    if norm1 in norm2 or norm2 in norm1:
        # Only merge if the extra part is short/generic
        longer = norm1 if len(norm1) > len(norm2) else norm2
        shorter = norm2 if len(norm1) > len(norm2) else norm1
        extra = longer.replace(shorter, '').strip()
        
        if len(extra) <= 10 or extra in generic_suffixes:
            return True
    
    return similarity >= threshold


def select_canonical_form(variations: list[str], tag_counts: dict[str, int]) -> str:
    """
    Select the best canonical form from duplicate variations.
    
    Priority:
    1. Most frequent in original data
    2. More descriptive (moderate length)
    3. Proper capitalization preserved
    """
    if len(variations) == 1:
        return variations[0]
    
    def score(tag: str) -> tuple:
        freq = tag_counts.get(tag, 1)
        # Prefer moderate length (10-30 chars)
        length_score = -abs(len(tag) - 20)
        # Prefer more words (more descriptive)
        word_count = len(tag.split())
        return (freq, word_count, length_score)
    
    return max(variations, key=score)


def deduplicate_tags(raw_tags: list[str]) -> DeduplicationResult:
    """
    Main deduplication function.
    
    Args:
        raw_tags: List of positive tags from exploration (may have duplicates)
        
    Returns:
        DeduplicationResult with unique tags and mapping information
    """
    # Count frequencies
    tag_counts = Counter(raw_tags)
    unique_raw = list(set(raw_tags))
    
    # Find duplicate groups
    groups = []
    used = set()
    duplicates_removed = []
    
    # Sort by frequency (most frequent first)
    sorted_tags = sorted(unique_raw, key=lambda t: tag_counts[t], reverse=True)
    
    for tag in sorted_tags:
        if normalize_tag(tag) in used:
            continue
        
        # Find all duplicates of this tag
        group_members = [tag]
        used.add(normalize_tag(tag))
        
        for other_tag in sorted_tags:
            if normalize_tag(other_tag) in used:
                continue
            
            if are_duplicates(tag, other_tag):
                group_members.append(other_tag)
                used.add(normalize_tag(other_tag))
        
        # Select canonical form
        canonical = select_canonical_form(group_members, tag_counts)
        
        # Track removed duplicates
        for member in group_members:
            if member != canonical:
                duplicates_removed.append((member, canonical))
        
        groups.append(TagGroup(
            canonical=canonical,
            variations=group_members,
            frequency=sum(tag_counts[m] for m in group_members)
        ))
    
    # Build mapping
    tag_mapping = {}
    for group in groups:
        for variation in group.variations:
            tag_mapping[variation] = group.canonical
            tag_mapping[normalize_tag(variation)] = group.canonical
    
    unique_tags = [g.canonical for g in groups]
    
    return DeduplicationResult(
        unique_tags=unique_tags,
        tag_groups=groups,
        tag_mapping=tag_mapping,
        original_count=len(raw_tags),
        deduplicated_count=len(unique_tags),
        duplicates_removed=duplicates_removed
    )