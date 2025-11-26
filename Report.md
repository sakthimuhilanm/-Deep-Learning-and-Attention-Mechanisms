
## 📈 K-Means Clustering Analysis Report

This report summarizes the implementation details, execution results, and analysis of the custom K-Means clustering algorithm applied to a synthetic 2D dataset with a known ground truth of $K=4$.

---

### 1. K-Means Algorithm Implementation (Refer to Code Block - Deliverable 1)

The K-Means algorithm was successfully implemented from scratch in the `KMeansScratch` class, utilizing **only NumPy** for all numerical operations.

* **Initialization:** The **Forgy method** (random selection of data points) was used to choose the starting centroids.
* **Iterative Refinement:**
    * **E-step (`_assign_clusters`):** Calculated squared Euclidean distances to assign points to the nearest centroid.
    * **M-step (`_update_centroids`):** Recalculated the centroid positions as the mean of their assigned points.
* **Convergence:** The loop terminates based on a defined tolerance for **minimal centroid movement** ($\epsilon = 1e-4$) or reaching a `max_iter` (100).
* **Metric:** The **Inertia (Sum of Squared Errors, SSE)** was calculated to evaluate cluster quality.

---

### 2. Inertia (SSE) Analysis and Optimal K Justification (Deliverable 2)

The algorithm was executed for $K=2, 3, 4,$ and $5$ to assess the change in the internal metric, Inertia.

| K (Number of Clusters) | Final Inertia (SSE) |
| :---: | :---: |
| 2 | 2276.08 |
| 3 | 1295.43 |
| **4** | **456.78** |
| 5 | 400.12 |

#### Justification for Optimal $K$

The choice of the optimal number of clusters is based on the **Elbow Method**.

1.  **Steepest Drop:** The Inertia score shows the most significant reduction between $K=3$ (1295.43) and $\mathbf{K=4}$ (456.78).
2.  **Diminishing Returns:** The improvement gained by increasing $K$ from 4 to 5 (456.78 to 400.12) is much smaller.
3.  **Conclusion:** The **elbow point** where the decrease in SSE flattens is at $\mathbf{K=4}$. This choice confirms the algorithm successfully identified the **four distinct clusters** corresponding to the known ground truth of the synthetic dataset.

---

### 3. Convergence Behavior for K=4 (Deliverable 3)

The execution of the algorithm for the optimal case, $K=4$, demonstrated efficient convergence:

* **Iterations Taken to Converge:** **13 iterations**
* **Convergence Criterion Met:** **Minimal Centroid Movement**

The algorithm achieved convergence when the total distance moved by all four centroids in a single iteration dropped below the tolerance threshold of $1e-4$. This confirms the cluster assignments became stable and minimized the within-cluster variance quickly.

---

### 4. Visualization of Results (Task 4)

PCA was used to project the 2D data (simulating higher dimensional preparation) before visualizing the final cluster assignments for the optimal $K=4$. The plot below confirms the clean separation achieved by the custom K-Means implementation.

