# Breast Cancer Classification using Artificial Neural Networks

This project uses Artiicial Neural Networks (ANN), LSTM, and 1D Convolutional Neural Networks (CNN) to classify breast cancer as malignant or benign using the Breast Cancer Wisconsin dataset.


## Project Overview

The aim of this project is to explore and compare different deep learning architectures (ANN, LSTM, and 1D CNN) for the classification of breast cancer tumors. The dataset is preprocessed and fed into neural network models built using TensorFlow and Keras, with performance evaluated using accuracy and loss metrics.

## Dataset

- **Primary:** [`breast-cancer.csv`](breast-cancer.csv) (uploaded locally in Colab)
- **Alternative:** [scikit-learn built-in breast cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

Features include mean values, standard errors, and worst values for attributes such as radius, texture, perimeter, area, smoothness, etc.

The target column is encoded as:
- `0`: Benign
- `1`: Malignant

## Project Structure

- `ANN.ipynb` — Main Jupyter Notebook with all code for preprocessing, EDA, visualization, model training and evaluation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xainasx/ANN-Project.git
   cd ANN-Project
   ```
2. **Install required libraries:**
   - Python >= 3.6
   - [TensorFlow](https://www.tensorflow.org/)
   - scikit-learn
   - pandas
   - matplotlib
   - seaborn
   ```bash
   pip install tensorflow scikit-learn pandas matplotlib seaborn
   ```

## Usage

1. **Open the notebook:**
   - You can run directly in [Google Colab](https://colab.research.google.com/github/xainasx/ANN-Project/blob/main/ANN.ipynb) or locally in Jupyter Notebook.

2. **Upload the Dataset:**
   - When prompted, upload your `breast-cancer.csv` file.

3. **Run all cells:**
   - The notebook will perform EDA, preprocessing, and train multiple deep learning models.

## Model Architectures

- **Feed-forward ANN**
  - Dense layers with activation functions (ReLU, sigmoid)
- **LSTM**
  - Single LSTM layer followed by dense output
- **1D Convolutional Neural Network**
  - Conv1D, MaxPooling1D, Dense layers
  
Each model uses appropriate activation and loss functions for binary or categorical classification.

## Results

- Training and validation accuracy/loss are plotted for every model.
- Model evaluations and comparison are provided at the end of the notebook.
- Typically, all models achieve high accuracy (>90%) on the breast cancer dataset.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

**Author:** [xainasx](https://github.com/xainasx)
