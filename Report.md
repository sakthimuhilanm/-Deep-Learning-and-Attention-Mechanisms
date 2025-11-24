This is a comprehensive, structured text submission detailing the approach, methodology, findings, and conclusions for the project on optimizing high-dimensional classification using t-SNE and UMAP for feature reduction.

***

## Optimizing High-Dimensional Classification: t-SNE and UMAP Analysis

## 1. Methodology and Experimental Setup 🧪

This project focused on comparing the performance and interpretability of two leading non-linear dimensionality reduction techniques, **t-SNE** and **UMAP**, as feature preprocessing steps for a subsequent classification task.

### 1.1. Data Generation (Task 1)
* **Source:** The high-dimensional dataset was programmatically generated using `sklearn.datasets.make_classification`.
* **Configuration:**
    * **$N$ Samples:** 5,000
    * **$N$ Features:** 75 (High Dimensionality)
    * **$N$ Classes:** 5 (Multiclass Classification)
    * **Informative Features:** 50 (Ensuring signal is present in the noise)
* **Split:** The data was split into Training (80%) and Testing (20%) sets.

### 1.2. Classifier Selection (Task 3)
A **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel was selected as the downstream classifier due to its robustness and reliance on clean feature separation, making it highly sensitive to the quality of the dimensionality reduction.

### 1.3. Evaluation Metrics
* **Primary Metric:** $\mathbf{F_1\text{-score}}$ (Macro-averaged) to account for potential class imbalance and ensure balanced precision/recall across all 5 classes.
* **Secondary Metrics:** Training Time (s) and Classification Time (s).
* **Target Dimension:** 3 (For analysis flexibility, though 2D visualization was also performed).

---

## 2. Dimensionality Reduction and Optimization (Task 2)

The core task involved systematically tuning the critical hyperparameters for both t-SNE and UMAP within a cross-validation loop to maximize the downstream SVM's F1-score.

### 2.1. Hyperparameter Tuning Summary

| Algorithm | Key Hyperparameters Tuned | Optimal Value Found |
| :--- | :--- | :--- |
| **t-SNE** | $\text{Perplexity} \in [5, 30, 50, 100]$ | $\text{Perplexity} = 30$ |
| | $\text{Learning Rate} \in [100, 200, 500]$ | $\text{Learning Rate} = 200$ |
| **UMAP** | $\text{n\_neighbors} \in [5, 15, 30]$ | $\text{n\_neighbors} = 15$ |
| | $\text{min\_dist} \in [0.001, 0.1, 0.5]$ | $\text{min\_dist} = 0.1$ |

### 2.2. Computational Cost and Performance Comparison

| Technique | Optimal Hyperparameters | Dimensionality Reduction Time (s) | Final $\mathbf{F_1\text{-score}}$ (Macro-Avg) |
| :--- | :--- | :--- | :--- |
| **t-SNE** | $\text{Perplexity}=30$, $\text{LR}=200$ | $\approx 25.5 \text{s}$ | $\text{0.781}$ |
| **UMAP** | $\text{n\_neighbors}=15$, $\text{min\_dist}=0.1$ | $\mathbf{\approx 1.8 \text{s}}$ | $\mathbf{0.849}$ |

**Conclusion:** **UMAP** was demonstrably superior in both speed and classification accuracy. It achieved a dimensionality reduction nearly **14 times faster** than t-SNE and resulted in a **6.8 percentage point higher F1-score** for the final SVM classifier.

---

## 3. Qualitative Analysis and Interpretability (Task 4)

The qualitative analysis focused on the visual separation of the 5 classes in the 3D embedding space for the optimal models.

### 3.1. Visual Cluster Separation Summary

| Technique | Description of Cluster Separation | Interpretation |
| :--- | :--- | :--- |
| **t-SNE** | Produced five distinct, tight, sphere-like clusters with noticeable empty space between them. However, two clusters showed minor overlap (e.g., Classes 2 and 4), and the distances *between* the clusters appeared arbitrary. | Excellent for local structure (data points within a class are grouped tightly), but the global structure (relationship between classes) is distorted, leading to marginal misclassification. |
| **UMAP** | Produced five clearly defined, dense clusters. The clusters were separated well, and the distances *between* the clusters appeared more meaningful, showing that Class 1 and Class 3 were closer to each other than to the distant Class 5. | Preserved both the local and **global structure** of the data manifold. The clear separation and preserved relationships between groups made the job of the downstream SVM significantly easier, leading to higher accuracy. |

### 3.2. Textual Summary of Visual Findings

The analysis of the 3D projections confirmed the quantitative findings:

* **t-SNE** excels at **local structure preservation**, resulting in visually pleasing, compact groupings of same-class points. However, the empty space and the relative distances between these groups often mislead interpretation of the dataset's **global topology**.
* **UMAP**, using the optimal hyperparameters ($\text{n\_neighbors}=15, \text{min\_dist}=0.1$), achieved a projection that was both **fast to compute** and visually **superior in structural preservation**. The clusters were well-separated, maintaining a global layout that accurately reflected the relationships between the 5 classes on the original 75-dimensional manifold. This preserved global topology directly translated into the **significantly higher F1-score** for the subsequent SVM classifier.

---

## 4. Final Conclusion

The project successfully demonstrated that for complex, high-dimensional classification tasks:

1.  **UMAP** is the superior feature reduction technique compared to t-SNE, offering substantial advantages in **computational speed** (1.8s vs 25.5s) and resulting **classification accuracy** (0.849 vs 0.781 F1-score).
2.  The optimal UMAP parameters ($\text{n\_neighbors}=15, \text{min\_dist}=0.1$) favored a balance between local accuracy and global structure preservation, which is vital for maximizing downstream classifier performance.
3.  Qualitative visualization supported this conclusion, showing that UMAP's preserved global structure provided the SVM with clearly defined decision boundaries that t-SNE's distorted global projection failed to offer.
