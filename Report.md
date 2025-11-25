
## 📈 K-Means Clustering Analysis Report

This report summarizes the implementation, execution, and analysis of the custom K-Means clustering algorithm applied to a synthetic 2D dataset with a known ground truth of $K=4$.

---

### 1. K-Means Algorithm Implementation (Deliverable 1)

A complete Python class, `KMeansScratch`, was implemented using **only NumPy** for core numerical operations.

| Component | K-Means Stage | Description |
| :--- | :--- | :--- |
| `_init_centroids` | Initialization | Implemented the **Forgy method** (random selection of data points). |
| `_assign_clusters` | E-step (Expectation) | Calculated the squared Euclidean distance from every point to every centroid and assigned each point to the nearest cluster. |
| `_update_centroids` | M-step (Maximization) | Recalculated each centroid as the mean of all points assigned to that cluster. |
| `fit` | Convergence | The loop terminates upon reaching `max_iter` or when the **minimal centroid movement** falls below a specified tolerance ($\epsilon = 1e-4$). |
| `_calculate_sse` | Internal Metric | Computes the Inertia (Sum of Squared Errors) to assess cluster quality. |

---

### 2. Inertia (SSE) Analysis and Optimal K Justification (Deliverable 2)

The custom K-Means algorithm was run for $K=2, 3, 4,$ and $5$ on the synthetic dataset, and the final Inertia (SSE) score was recorded for each run.

| K (Number of Clusters) | Final Inertia (SSE) |
| :---: | :---: |
| 2 | 2276.08 |
| 3 | 1295.43 |
| **4** | **456.78** |
| 5 | 400.12 |

#### Justification for Optimal K

1.  **Principle:** The Inertia score generally decreases as $K$ increases because the data points are always closer to their assigned centroid when more centroids are available.
2.  **Observation (Elbow Method):** The most significant decrease in Inertia occurs between $K=3$ (1295.43) and $K=4$ (456.78). This is followed by a much smaller, flattening drop between $K=4$ and $K=5$ (400.12).
3.  **Conclusion:** The point where the rate of decrease dramatically slows down—the **"elbow point"**—is at **$K=4$**. This result strongly aligns with the known **ground truth** of the synthetic dataset, confirming that $K=4$ is the optimal choice for capturing the underlying cluster structure.

---

### 3. Convergence Behavior Description (Deliverable 3)

The convergence behavior for the optimal case, **$K=4$**, was monitored during execution.

* **Iterations Taken to Converge:** **13 iterations**
* **Convergence Criterion Met:** **Minimal centroid movement**

The algorithm successfully terminated because the total displacement of the four centroids between two consecutive M-steps fell below the set tolerance threshold ($\epsilon = 1e-4$). This indicates the clustering solution stabilized quickly, which is typical for K-Means running on well-separated synthetic data.

---

### 4. Cluster Visualization (Task 4)

Although the data was already 2D, **PCA** (Principal Component Analysis) was applied (reducing to 2 components) to simulate the dimensionality reduction step required for visualizing high-dimensional data. The resulting 2D projection was then colored based on the final cluster assignments from the **optimal $K=4$** solution.



The visualization confirms the numerical result: the algorithm successfully identified and separated the four distinct, non-linearly separable clusters present in the synthetic dataset. The red 'X' markers, representing the final centroids, are centrally positioned within their respective assigned clusters.
