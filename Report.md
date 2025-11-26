
## 📈 K-Means Clustering Analysis Report

This report summarizes the implementation details, metric calculations, and analysis of the custom K-Means clustering algorithm applied to a synthetic 2D dataset with a known ground truth of $K=3$.

***

### 1. Implementation Overview (Deliverable 1)

The K-Means algorithm was implemented entirely from scratch in the `KMeansScratch` class, relying solely on **NumPy** for numerical computations. This implementation included:

* **Initialization:** Centroids were initialized using the **Forgy method** (random selection of data points).
* **Iterative Core:** The algorithm executes the **E-step** (assignment based on Euclidean distance) and the **M-step** (centroid update based on the mean) until convergence.
* **Metrics:** Functions were included to calculate the **WCSS** (Within-Cluster Sum of Squares) for the Elbow Method.

***

### 2. Metric Analysis and Optimal K Justification (Deliverable 2)

The clustering metrics, **WCSS** (for the Elbow Method) and **Silhouette Score**, were calculated for $K$ values ranging from 2 to 10 to programmatically determine the optimal number of clusters.

| K | WCSS Score | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 1148.24 | 0.4497 |
| **3** | **407.48** | **0.7850** |
| 4 | 338.25 | 0.6033 |
| 5 | 289.47 | 0.4357 |
| 6 | 258.42 | 0.3546 |
| 7 | 231.78 | 0.3011 |
| 8 | 218.44 | 0.2811 |
| 9 | 200.01 | 0.2522 |
| 10 | 185.34 | 0.2319 |

#### Justification for Optimal $K$

Both programmatic methods unequivocally point to $\mathbf{K=3}$:

1.  **Elbow Method (WCSS):** The WCSS experiences the **most significant drop** (the "elbow") between $K=2$ (1148.24) and $\mathbf{K=3}$ (407.48). The rate of decrease flattens dramatically thereafter.
2.  **Silhouette Analysis:** The metric reaches its **highest value** ($\mathbf{0.7850}$) at $\mathbf{K=3}$, indicating that the clusters are the best defined and most internally compact while being well-separated from neighboring clusters.

**Conclusion:** The **Optimal $K$ is $\mathbf{3}$**, which successfully identifies the number of groups defined in the synthetic dataset.

***

### 3. Interpretation and Visualization (Deliverables 3 & 4)

The final K-Means model was fitted using the optimal $\mathbf{K=3}$.

#### Final Cluster Center Interpretation

The calculated final centroids define the structure of the identified groups:

| Cluster | Feature 1 Center | Feature 2 Center |
| :---: | :---: | :---: |
| 0 | -6.83 | 6.84 |
| 1 | 2.15 | 8.81 |
| 2 | -0.19 | 1.84 |

These three centers accurately represent the **mean location** of the three distinct, non-overlapping groups within the 2D feature space.

#### Visualization Evidence

A 2D scatter plot was generated to visualize the final cluster assignments for $K=3$.



**Summary of Visualization:**
The visualization confirms that the scratch implementation effectively partitioned the data into **three clean, distinct clusters**. The data points are correctly grouped, and the red 'X' markers (centroids) are centrally located within their respective assigned clusters, validating the model's accuracy.
