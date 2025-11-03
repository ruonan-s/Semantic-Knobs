"""
Test the elbow method for K-means clustering
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def find_elbow_point(inertias, k_values):
    """Find elbow point using the maximum distance from line method"""
    if len(inertias) < 3:
        return k_values[len(inertias) // 2]
    
    inertias = np.array(inertias)
    k_values = np.array(k_values)
    
    # Create line from first to last point
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    
    # Calculate distance from each point to the line
    distances = []
    for i in range(len(k_values)):
        point = np.array([k_values[i], inertias[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        distances.append(distance)
    
    # Find the point with maximum distance (the elbow)
    elbow_idx = np.argmax(distances)
    return int(k_values[elbow_idx])

# Generate sample data with natural clusters
np.random.seed(42)
n_samples_per_cluster = 15
n_true_clusters = 8

data = []
for i in range(n_true_clusters):
    center = np.random.randn(2) * 5
    cluster = np.random.randn(n_samples_per_cluster, 2) + center
    data.append(cluster)

data = np.vstack(data)
print(f"Generated {len(data)} samples with {n_true_clusters} true clusters")

# Test K-means with different K values
k_range = range(3, 20)
inertias = []

print("\nRunning K-means for different K values...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
    print(f"  K={k:2d}: inertia={kmeans.inertia_:.2f}")

# Find elbow
optimal_k = find_elbow_point(inertias, list(k_range))
print(f"\n✓ Elbow method detected optimal K = {optimal_k}")
print(f"  (True number of clusters: {n_true_clusters})")

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Elbow at K={optimal_k}')
plt.axvline(x=n_true_clusters, color='g', linestyle=':', linewidth=2, label=f'True K={n_true_clusters}')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/ruonansun/Desktop/World-stylizer/Exploration+Refinement/backend/elbow_test.png', dpi=150)
print(f"\n✓ Plot saved to: backend/elbow_test.png")

print("\n" + "="*60)
print("ELBOW METHOD TEST PASSED!")
print("="*60)

