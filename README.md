\# Netflix Content Segmentation



An unsupervised clustering model that groups Netflix titles into meaningful content segments using K-Means — built as Task 4 of the Auspify Machine Learning internship.



\## What It Does



Groups the entire Netflix catalog into 5 distinct content clusters based on genre, type, release year, rating, and duration — without using any predefined labels — then interprets what each cluster represents.



\## How It Works



1\. \*\*Feature Preparation\*\* — Combined scaled numeric features (`type`, `rating`, `release\_year`, movie minutes, TV seasons) with TF-IDF genre vectors (capped at 30 terms to reduce dimensionality for clustering).

2\. \*\*Data Scaling\*\* — Applied `StandardScaler` to all numeric features so no single feature (e.g. release year) dominates distance calculations due to its raw scale.

3\. \*\*Optimal K Selection\*\* — Used the Elbow Method, testing k=2 through k=10 and comparing inertia drop-off, to justify choosing k=5.

4\. \*\*Clustering\*\* — Applied K-Means with k=5 and 10 random initializations to avoid poor local minima.

5\. \*\*Cluster Identification\*\* — Profiled each cluster's type split, average release year, top ratings, and top genres to determine what each group represents.

6\. \*\*Visualization\*\* — Reduced the 35-dimensional feature space to 2D using PCA and plotted the clusters (66.76% of variance explained by the 2 components).

7\. \*\*Interpretation\*\* — Named and summarized each cluster in plain language.



\## Tech Stack



\- Python

\- pandas, numpy

\- scikit-learn (`KMeans`, `StandardScaler`, `LabelEncoder`, `TfidfVectorizer`, `PCA`)

\- scipy

\- matplotlib



\## Dataset



`Dataset.csv` — 8,790 Netflix titles (same dataset as Tasks 1–3).



\## How to Run



pip install pandas numpy scikit-learn scipy matplotlib

python Segmentation.py





Note: the script opens a matplotlib plot window during Step 4 — the script pauses until you close that window, then continues to print the final interpretation summary.



\## Results — Cluster Profiles



| Cluster | Size | % of Catalog | Description |

|---|---|---|---|

| 0 | 477 | 5.4% | Classic Movies — avg. 1985, Dramas/Action-heavy |

| 1 | 2,910 | 33.1% | Modern Mature Movies — avg. 2017, strongly TV-MA |

| 2 | 252 | 2.9% | Niche/Family TV Shows — smallest cluster, Kids' TV present |

| 3 | 2,402 | 27.3% | International TV Shows — dominant genre is International TV |

| 4 | 2,749 | 31.3% | General Audience Movies — mid-2010s, balanced TV-14/R |



\## Key Finding



Clustering naturally separated content along two axes: \*\*Content Type\*\* (Movie vs TV Show) as the dominant divide, and \*\*Release Era / Maturity Rating\*\* as the secondary divide — this emerged purely from feature similarity, without explicitly telling the model to prioritize `type`.



\## Known Limitations



\- PCA visualization captures \~67% of the original variance — a useful approximation, not a complete picture of the 35-dimensional clustering.

\- K-Means assumes roughly spherical, similarly-sized clusters, which may not perfectly reflect the true shape of content groupings.

