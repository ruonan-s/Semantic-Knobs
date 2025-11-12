# Current Tag Clustering Method Summary

## Overview

Your system uses **K-means clustering on CLIP embeddings** to group similar tags into semantic concepts. This is a **purely embedding-based approach** with no predefined semantic rules or manual grouping.

## Pipeline

### 1. Tag Collection & Normalization
```python
for image_id, tags in image_tags.items():
    for tag_text in tags:
        normalized = normalize_text(tag_text)      # Lowercase, clean punctuation
        lemmatized = simple_lemmatize(normalized)  # "chairs" → "chair"
        all_texts.append(lemmatized)
```

**Purpose:** Standardize text for better embedding quality

### 2. CLIP Embedding Generation
```python
embeddings = get_batch_embeddings(all_texts)  # CLIP ViT-L/14
```

**Model:** CLIP ViT-L/14 (Large, 336M parameters)  
**Device:** CUDA if available, else CPU  
**Output:** 768-dimensional unit-normalized vectors  
**Why CLIP?** Pre-trained on 400M image-text pairs, excellent for visual concepts

### 3. Optimal K Estimation (Elbow Method)
```python
K = estimate_optimal_k(n_tags, embeddings, min_k=5, max_k=80)
```

**Algorithm:**
1. Test K-means with K from 5 to min(80, 70% of n_tags)
2. Compute inertia (within-cluster sum of squares) for each K
3. Find "elbow point" where adding more clusters has diminishing returns
4. Use second derivative to detect sharp change in slope

**Adaptive Range:**
- Small datasets (<10 tags): K ≤ n_tags // 2
- Medium datasets (10-150 tags): Test every 1-2 values
- Large datasets (>150 tags): Test every 3 values
- Maximum tested: 25-30 K values to find elbow

**Fallback:** If elbow detection fails → K = n_tags // 2 (target ~2 tags per cluster)

### 4. K-means Clustering
```python
kmeans = KMeans(
    n_clusters=K,
    random_state=42,    # Reproducible results
    n_init=10,          # 10 initializations, pick best
    max_iter=300        # Convergence iterations
)
cluster_labels = kmeans.fit_predict(embeddings)
```

**What it does:**
- Groups tags with similar CLIP embeddings
- Minimizes within-cluster variance
- Maximizes between-cluster separation

**Result:** Each tag assigned to one cluster (concept)

### 5. Concept Creation
For each cluster:
```python
# Compute centroid (average embedding)
centroid = mean(embeddings_in_cluster)
centroid = normalize(centroid)  # Unit norm

# Choose label: most frequent + shortest tag
label = most_common_tag_in_cluster

# Create concept
concept = Concept(
    id="concept_N",
    label=label,
    centroid=centroid,
    member_tag_ids=[...]
)
```

**Label Selection Logic:**
1. Count frequency of each unique tag text in cluster
2. Sort by: frequency (desc) → length (asc)
3. Use top result as concept label

**Example:**
- Cluster has: ["warm lighting", "warm lighting", "cozy light"]
- Label: "warm lighting" (most frequent, shorter than "cozy light")

### 6. State Initialization
```python
K = len(concepts)
initial_weight = 1.0 / K  # Uniform distribution

for concept in concepts:
    concept_states[concept.id] = ConceptState(
        w=initial_weight,        # Raw weight
        ema_w=initial_weight,    # EMA smoothed weight
        like_count=0,
        dislike_count=0,
        score=0
    )
```

**Initial state:** All concepts equally weighted (simplex constraint)

## Key Parameters

```python
MIN_CLUSTERS = 5          # Minimum K
MAX_CLUSTERS = 80         # Maximum K (allows fine granularity)
TARGET_TAGS_PER_CLUSTER = 2  # Heuristic fallback target
```

**Tuning:**
- Increase `TARGET_TAGS_PER_CLUSTER` → Fewer, larger concepts (more merging)
- Decrease `TARGET_TAGS_PER_CLUSTER` → More, smaller concepts (finer distinction)

## Example Flow

**Input:** 48 tags from 3 images
```
impression_0: ["warm lighting", "wooden desk", "potted plants", ...]
impression_1: ["minimalist design", "white walls", "open layout", ...]
impression_2: ["cozy atmosphere", "soft shadows", "warm lighting", ...]
```

**Step 1: Normalization**
```
"Warm Lighting" → "warm lighting" → "warm light"
"Wooden Desk" → "wooden desk" → "wooden desk"
```

**Step 2: CLIP Embeddings**
```
"warm light" → [0.231, -0.124, 0.567, ..., 0.089]  # 768 dims
"wooden desk" → [0.445, 0.223, -0.321, ..., 0.156]
...
```

**Step 3: Elbow Method**
```
Testing K from 5 to 33 (70% of 48)
Inertias: [1250, 980, 820, 710, 640, 590, 560, 540, ...]
                                     ↑ Elbow around K=17
```

**Step 4: K-means with K=17**
```
Cluster 0: ["warm light", "cozy light", "soft glow"]
Cluster 1: ["wooden desk", "wood furniture"]
Cluster 2: ["potted plants", "indoor plants", "greenery"]
Cluster 3: ["minimalist design", "minimal aesthetic"]
...
```

**Step 5: Concept Creation**
```
concept_0: 
  label: "warm light" (most frequent)
  members: 3 tags
  centroid: mean of 3 embeddings

concept_1:
  label: "wooden desk"
  members: 2 tags
  centroid: mean of 2 embeddings
...
```

**Output:** 17 concepts, each with ~2-3 tags

## Strengths

✅ **Semantic Grouping** - CLIP understands meaning, not just text similarity  
✅ **No Manual Rules** - Fully automatic, adapts to any domain  
✅ **Scalable** - Works with 10 to 1000+ tags  
✅ **Reproducible** - Fixed random seed ensures consistency  
✅ **Optimal K** - Elbow method finds natural number of concepts  
✅ **Multi-lingual Ready** - CLIP handles multiple languages  

## Limitations

⚠️ **K-means assumptions:**
- Assumes spherical clusters (may split elongated semantic groups)
- Sensitive to initialization (mitigated with n_init=10)
- Equal variance assumption (some concepts tighter than others)

⚠️ **Elbow detection:**
- Can be ambiguous (multiple "elbows")
- Falls back to heuristic if detection fails

⚠️ **Label selection:**
- Uses frequency, not semantic centrality
- May pick non-representative tag if frequency tied

## Alternatives Considered (Not Currently Used)

1. **Hierarchical Clustering** - Would allow nested concepts
2. **DBSCAN** - Would find arbitrary-shaped clusters
3. **Agglomerative Clustering** - Would create concept hierarchy
4. **Manual Thresholds** - Would use fixed cosine similarity cutoffs

**Current choice (K-means) is optimal for:**
- Fixed number of concepts needed for weight normalization
- Fast computation
- Interpretable results

## Code Location

- **Main logic:** `backend/concept_refinement.py:251-340` (`build_concepts()`)
- **Elbow method:** `backend/concept_refinement.py:192-248` (`estimate_optimal_k()`)
- **Initialization:** `backend/concept_refinement.py:631-702` (`initialize_from_tags()`)
- **CLIP model:** `backend/concept_refinement.py:24-28` (loaded globally)
- **Text normalization:** `backend/concept_refinement.py:80-120`

## Summary

**Your current method:** K-means clustering on CLIP ViT-L/14 embeddings with automatic K selection via elbow method, creating 5-80 semantic concepts from raw image tags, with concept labels chosen by frequency and length.

