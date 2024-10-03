# Intrusion Detection System (IDS) Project

## Project Overview
This project is an **Intrusion Detection System (IDS)** built using machine learning models to detect anomalies and malicious activities in network traffic. The system is based on the **CICIDS2018** dataset and other network traffic datasets to classify network behavior into normal and attack types.

The project is designed to handle real-time network traffic and can be integrated into a production environment to detect intrusions based on predefined network features.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Pipeline](#modeling-pipeline)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)


---

## Project Structure

```bash
├── artifacts
│   ├── data.csv                # Raw dataset used for training
│   ├── model_trained.pkl        # Final trained machine learning model
│   ├── preprocessor.pkl         # Preprocessor pipeline (scalers, encoders, etc.)
│   ├── test.csv                 # Test data
│   ├── train.csv                # Training data
│   └── train_data.csv           # Another version of training data
│
├── dataset
│   └── train_data.csv           # Raw dataset files
│
├── logs                         # Logging directory for errors and process tracking
│
├── src
│   ├── components
│   │   ├── data_ingestion.py     # Script for data ingestion from source
│   │   ├── data_transformation.py # Script for data preprocessing and transformation
│   │   └── model_trainer.py      # Script for model training
│   │
│   ├── pipeline
│   │   ├── exception.py          # Custom exception handling
│   │   ├── logger.py             # Logging utility
│   │   └── utils.py              # Helper functions
│
├── venv                         # Virtual environment directory
│
├── .gitignore                   # Git ignore file
├── README.md                    # Readme file for the project (you are here!)
├── requirements.txt             # Python dependencies
├── setup.py                     # Project setup file
└── README.md                    # Project documentation (this file)
```
---

## Datasets

This project is built on the **CICIDS2018** dataset, a comprehensive dataset for cybersecurity research.

### **CICIDS2018 Dataset**
- **Description**: The dataset includes labeled network traffic data, with various attacks (DoS, brute force, SQL injection, botnets, etc.) and normal network traffic.
- **Link**: You can download the dataset [here](https://www.unb.ca/cic/datasets/ids-2018.html).
- **Size**: Large datasets, split into multiple CSV files, exceeding hundreds of MB.

The training and test datasets used for this project are stored in the `dataset/` folder. The main dataset used is `train_data.csv`.

---

## Installation
To set up this project on your local machine:

### Clone the Repository:

```bash
git clone https://github.com/yourusername/ids-project.git
cd ids-project
```

### Create and Activate Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### Install the Required Packages:
```bash
pip install -r requirements.txt
```

---
## Usage

### Data Ingestion:

The `data_ingestion.py` script is responsible for fetching and loading the dataset.

### Data Transformation:

The `data_transformation.py` script preprocesses and transforms the raw data for use in the model.

### Model Training:

The `model_trainer.py` script handles training the model using the transformed data.

### Run the Project:

```bash
python src/components/data_ingestion.py    # Ingest data
```
---
## Modeling Pipeline
The project follows a pipeline structure, where:

- **Data Ingestion**: The dataset is loaded from the source CSV files.
- **Data Transformation**: The dataset is preprocessed (handling missing values, feature scaling, encoding).
- **Model Training**: The preprocessed data is used to train a machine learning model for intrusion detection.
  
The model is then saved to the `artifacts/` folder as `model_trained.pkl`, and the preprocessor pipeline is saved as `preprocessor.pkl`.

---

## Results

The model was evaluated on both the training and testing datasets. Below are the performance metrics:

- **Testing Accuracy Score**: 89.75%
- **Training Accuracy Score**: 89.87%
- **Testing F1 Score**: 88.27%
- **Training F1 Score**: 88.40%
- **Testing Recall Score**: 89.75%
- **Training Recall Score**: 89.87%
- **Testing Precision Score**: 89.08%
- **Training Precision Score**: 89.31%
- **Balanced Accuracy Score**: 86.55%
- **ROC AUC (Testing)**: 99.17%
- **ROC AUC (Training)**: 99.21%

---
### Conclusion

The model demonstrates strong performance with high accuracy, precision, recall, and F1 scores on both training and testing datasets. The ROC AUC score of 99% on both training and testing datasets indicates the model's excellent ability to distinguish between different classes.

However, the slight difference between training and testing scores suggests that the model generalizes well to unseen data and is not overfitting. With a balanced accuracy of 86.55%, the model performs well across all classes, even in the presence of imbalanced data.


---

## Contributing

If you'd like to contribute to this project, feel free to create a pull request or open an issue to discuss changes.

---


