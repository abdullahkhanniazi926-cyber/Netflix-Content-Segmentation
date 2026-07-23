import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import re

df = pd.read_csv('Dataset.csv')

# --- Handle duration the same way as Task 3 ---
def extract_minutes(row):
    if row['type'] == 'Movie' and pd.notna(row['duration']):
        match = re.search(r'(\d+)', row['duration'])
        return int(match.group(1)) if match else 0
    return 0

def extract_seasons(row):
    if row['type'] == 'TV Show' and pd.notna(row['duration']):
        match = re.search(r'(\d+)', row['duration'])
        return int(match.group(1)) if match else 0
    return 0

df['movie_minutes'] = df.apply(extract_minutes, axis=1)
df['tv_seasons'] = df.apply(extract_seasons, axis=1)

# --- Encode categorical features ---
le_type = LabelEncoder()
le_rating = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'])
df['rating_encoded'] = le_rating.fit_transform(df['rating'])

# --- TF-IDF on genres ---
tfidf = TfidfVectorizer(stop_words='english', max_features=30)
genre_matrix = tfidf.fit_transform(df['listed_in']).toarray()

# --- Scale the numeric features (critical for K-Means) ---
numeric_cols = ['type_encoded', 'rating_encoded', 'release_year', 'movie_minutes', 'tv_seasons']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(df[numeric_cols])

# --- Combine scaled numeric features with genre features into one array ---
import numpy as np
X = np.hstack([scaled_numeric, genre_matrix])

print("Final feature matrix shape:", X.shape)
print("Sample of scaled numeric features (first row):", scaled_numeric[0])
from sklearn.cluster import KMeans

# Try k from 2 to 10, record inertia (how tight/compact the clusters are) for each
inertia_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    print(f"k={k}: inertia={kmeans.inertia_:.2f}")
    # Based on the elbow method, k=5 gives the best balance of compactness vs. simplicity
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X)

df['cluster'] = cluster_labels

print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index())
print("\n" + "="*60)
print("CLUSTER PROFILES")
print("="*60)

for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]

    print(f"\n--- Cluster {cluster_id} ({len(cluster_data)} titles) ---")
    print(f"Type split: {cluster_data['type'].value_counts().to_dict()}")
    print(f"Avg release year: {cluster_data['release_year'].mean():.0f}")
    print(f"Top ratings: {cluster_data['rating'].value_counts().head(3).to_dict()}")

    # Get the most common individual genre words across this cluster
    all_genres = cluster_data['listed_in'].str.split(', ').explode()
    top_genres = all_genres.value_counts().head(5)
    print(f"Top genres:\n{top_genres.to_string()}")
    from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Netflix Content Clusters (PCA-reduced to 2D)')
plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Explained variance by 2 components:", pca.explained_variance_ratio_.sum())
print("\n" + "="*60)
print("CLUSTER INTERPRETATION SUMMARY")
print("="*60)

interpretations = {
    0: "Classic Movies — older films (avg. 1985), Dramas/Action-heavy, minimal TV content.",
    1: "Modern Mature Movies — recent films (avg. 2017), strongly TV-MA rated, International/Drama-heavy.",
    2: "Niche/Family TV Shows — smallest cluster, mixed comedies/dramas with notable Kids' TV presence.",
    3: "International TV Shows — dominant genre is International TV Shows, largest TV Show cluster.",
    4: "General Audience Movies — mid-2010s films, more balanced TV-14/R ratings, broad genre mix."
}

for cluster_id, description in interpretations.items():
    size = df[df['cluster'] == cluster_id].shape[0]
    pct = size / len(df) * 100
    print(f"\nCluster {cluster_id} ({size} titles, {pct:.1f}% of catalog):")
    print(f"  {description}")

print("\n" + "="*60)
print("KEY FINDING")
print("="*60)
print("Clustering naturally separated content along two main axes:")
print("1. Content Type (Movie vs TV Show) - the dominant divide")
print("2. Release Era / Maturity Rating - the secondary divide")
print("This happened without explicitly telling the model to prioritize")
print("'type' - it emerged purely from feature similarity patterns.")

df.to_csv('final_clustered_dataset.csv', index=False)