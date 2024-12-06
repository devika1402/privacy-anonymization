# **Graph Anonymization and Evaluation Framework**

## **Overview**
This project provides a framework for graph anonymization and utility analysis using various techniques like random perturbation, differential privacy, hybrid anonymization, and synthetic graph generation. The framework also supports the analysis of privacy risks and utility loss, and it includes visualizations for comparing original and anonymized graphs.

---

### **Anonymization Techniques**
- **Random Perturbation**: Randomly adds and removes edges.
- **Differential Privacy**: Adds noise to graph properties (e.g., degree) to preserve privacy.
- **Hybrid Anonymization**: Combines differential privacy with edge-swapping techniques.
- **Synthetic Graphs**:
  - Generates graphs with matching degree distributions.
- **Synthetic Graph using DCSBM**:
  - Creates synthetic graphs using Degree-Corrected Stochastic Block Models (DCSBM).

### **Evaluation Metrics**
- **Privacy**:
  - **Re-identification Rate**: Measures the risk of identifying original edges.
- **Utility**:
  - Evaluates accuracy in tasks like node classification.
  - Measures feature correlations (e.g., clustering coefficient, triangles).
- **Robustness**: Assesses structural similarity through degree distributions.
- **Structural Similarity**: Includes metrics like modularity and triangle similarity.

## **Requirements**
Ensure you have the following libraries installed:
- Python >= 3.7
- NumPy
- NetworkX
- Scikit-learn
- Matplotlib
- Node2Vec
- TSNE (optional, for visualization)
- Graphviz (optional)
