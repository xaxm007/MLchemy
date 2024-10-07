# 💻 MLchemy

A repository showcasing all my Jupyter notebooks and datasets used to implement various machine learning models during my learning journey.

---

## 📋 Table of Contents

- [Folder Structure](#folder-structure)
- [Topics Covered](#topics-covered)
- [How to Use](#how-to-use)
- [Requirements](#requirements)

---

## 📂 Folder Structure

```
MLchemy/
│
├── Regression/
│   ├── data
│   │   └── dataset.csv
│   └── notebook
│       ├── simple_linear_regression.ipynb
│       ├── support_vector_regression.ipynb
│       └── decision_tree_regression.ipynb
│
├── Classification/
│   ├── data
│   │   └── dataset.csv
│   └── notebook
│       ├── k_nearest_neighbours.ipynb
│       ├── support_vector_machine.ipynb
│       └── naive_bayes.ipynb
│
└── Dimensionality Reduction (PCA)/
    ├── data
    │   └── dataset.csv
    └── notebook
        ├── principal_component_analysis.ipynb
        └── kernel_pca.ipynb
```

---

## 📎 Topics Covered

Machine Learning Implementations covered in this repository:

- **[Regression](./Regression/)**: 
   - Simple Linear Regression
   - Multiple Linear Regression
   - Polynomial Regression
   - Support Vector Regression
   - Decision Tree Regression
   - Random Forest Regression

- **[Classification](./Classification/)**: 
   - Logistic Regression
   - K Nearest Neighbors 
   - Support Vector Machines
   - Kernel SVM
   - Naive Bayes
   - Decision Tree Classification
   - Random Forest Classification

- **[Clustering](./Clustering/)**: 
   - K-Means Clustering
   - Hierarchical Clustering

- **[Dimensionality Reduction](./PCA/)**: 
   - Principal Component Analysis (PCA)
   - Linear Discriminant Analysis (LDA)
   - Kernel PCA

---

## 👉 How to Use

To run the notebooks locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/MLchemy.git
   cd MLchemy
   ```

   `Recommended`, Follow steps to use conda env:

   1. **Create a conda env**: 

      ```bash
      conda create -n <your_environment_name>
      ```

   2. **Install Ipykernel for using Jupyter Notebook**:

      ```bash
      conda install ipykernel
      ```

   3. **Connect new ipykernel to the conda env**:

      ```bash
      python -m ipykernel install --user --name <your_env_name> --display-name "<new_name_for_your_kernel"
      ```

2. **Install the required dependencies** (see [Requirements](#requirements)):

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebooks**:

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Navigate to the relevant notebook under the respective folder and run the cells.

---

## 🛠️ Requirements

To run the code in this repository, you need the following:

- Numpy
- Pandas 
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Required packages are listed in `requirements.txt`. Install them using:

  ```bash
  pip install -r requirements.txt
  ```
