#!/usr/bin/env python3
import json

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def build_meta_map(data_list, prefix):
    """
    Build a map from image_id to metadata.
    Image IDs follow the format: prefix_i_0
    """
    meta_map = {}
    for i, item in enumerate(data_list):
        image_id = f"{prefix}_{i}_0"
        meta_map[image_id] = {
            "concept_name": item["concept_name"],
            "core_strategy": item["core_strategy"]
        }
    return meta_map

def extract_full_entry(data_list, selected_id):
    """
    Given a data list and a selected_id like 'impression_3_0',
    return the full JSON entry for that ID.
    """
    index = int(selected_id.split("_")[1])
    return data_list[index].copy()

def generate_user_preference(impression_path, spatial_path, objects_path, ambient_path, preferences_path):
    """
    Generate the combined user_preference object by:
      - Loading data files
      - Extracting selections
      - Aggregating tag feedback into per-layer preferences
      - Formatting other preferences for unselected images
    """
    # Load data
    impression_data = load_json(impression_path)
    spatial_data    = load_json(spatial_path)
    objects_data    = load_json(objects_path)
    ambient_data    = load_json(ambient_path)
    preferences     = load_json(preferences_path)

    # Build meta map for unselected images
    meta_map = {}
    meta_map.update(build_meta_map(impression_data, "impression"))
    meta_map.update(build_meta_map(spatial_data,    "spatial"))
    meta_map.update(build_meta_map(objects_data,    "objects"))
    meta_map.update(build_meta_map(ambient_data,    "ambient"))

    # Get selections and tag feedback
    selected     = preferences["selections"]
    tag_feedback = preferences["tags"]["parallel"]

    # Initialize per-layer tag preferences
    layer_tags = {
        "impression": {"prefer_to_include": [], "prefer_to_avoid": []},
        "spatial":    {"prefer_to_include": [], "prefer_to_avoid": []},
        "ambient":    {"prefer_to_include": [], "prefer_to_avoid": []},
    }
    
    # Collect tags by source image and sentiment for grouping
    other_tags_by_source = {}  # source -> {"positive": [tags], "negative": [tags]}

    # Process each tag entry
    for entry in tag_feedback:
        tag       = entry["tag"]
        sentiment = entry["preference"]
        source    = entry["source_image"]

        # Assign to selected layer or collect for other_preference grouping
        if source == selected["impression"]:
            layer = "impression"
        elif source == selected["spatial"]:
            layer = "spatial"
        elif source == selected["ambient"]:
            layer = "ambient"
        else:
            # Collect tags for grouping by source image
            if source not in other_tags_by_source:
                other_tags_by_source[source] = {"positive": [], "negative": []}
            
            if sentiment == "positive":
                other_tags_by_source[source]["positive"].append(tag)
            else:
                other_tags_by_source[source]["negative"].append(tag)
            continue

        # Add to the layer's preferences
        if sentiment == "positive":
            layer_tags[layer]["prefer_to_include"].append(tag)
        else:
            layer_tags[layer]["prefer_to_avoid"].append(tag)
    
    # Format grouped tags for other_preference
    other_preference = {"preferred": [], "avoid": []}
    for source, tag_groups in other_tags_by_source.items():
        meta = meta_map.get(source)
        if meta:
            # Group positive tags
            if tag_groups["positive"]:
                grouped_tags = ", ".join(tag_groups["positive"])
                formatted = f"{grouped_tags} in {meta['concept_name']} ({meta['core_strategy']})"
                other_preference["preferred"].append(formatted)
            
            # Group negative tags  
            if tag_groups["negative"]:
                grouped_tags = ", ".join(tag_groups["negative"])
                formatted = f"{grouped_tags} in {meta['concept_name']} ({meta['core_strategy']})"
                other_preference["avoid"].append(formatted)

    # Build final user_preference object
    user_preference = {
        "impression": extract_full_entry(impression_data, selected["impression"]),
        "spatial":    extract_full_entry(spatial_data,    selected["spatial"]),
        "objects":    extract_full_entry(objects_data,    selected["objects"]),
        "ambient":    extract_full_entry(ambient_data,    selected["ambient"]),
    }
    
    for layer in ["impression", "spatial", "objects", "ambient"]:
        user_preference[layer]["visual_elements_preference"] = layer_tags[layer]
    user_preference["other_preference"] = other_preference

    return user_preference

if __name__ == "__main__":
    # File paths (adjust if needed)
    IMPRESSION_PATH = "impression.json"
    SPATIAL_PATH    = "spatial.json"
    AMBIENT_PATH    = "ambient.json"
    PREF_PATH       = "preferences.json"

    user_pref = generate_user_preference(
        IMPRESSION_PATH,
        SPATIAL_PATH,
        AMBIENT_PATH,
        PREF_PATH
    )
    print(json.dumps(user_pref, indent=2))
