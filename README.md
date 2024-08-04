# SentimentAnalysisNLP

## Overview

Welcome to the SentimentAnalysisNLP project! This repository contains a comprehensive sentiment analysis pipeline that processes and classifies tweets into different sentiment categories using Natural Language Processing (NLP) techniques and machine learning models.

The goal of this project is to accurately classify tweets into one of four sentiment categories:
- Positive
- Negative
- Neutral
- Irrelevant

![visualization1](https://github.com/user-attachments/assets/8145f959-1dcb-48ad-a32b-d0df11a4fc93)

## Features

- **Data Preprocessing**: Tokenization, lowercasing, punctuation removal, stopword removal, and lemmatization.
- **Vectorization**: TF-IDF and Count Vectorizer methods.
- **Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost.
- **Model Evaluation**: Accuracy, precision, recall, F1 score, ROC AUC score, and classification report.
- **Visualization**: Word cloud and bar plot of the most common words.
- **Saved Models**: Pre-trained models available for direct use.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Clone the Repository

```sh
git clone https://github.com/Chralsh/SentimentAnalysisNLP.git
cd SentimentAnalysisNLP
```

### Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

## Usage

### Data Preparation

Ensure you have your datasets in the `data` directory. You should have two CSV files: `twitter_training.csv` and `twitter_validation.csv`.

### Running the Jupyter Notebook

Launch Jupyter Notebook and open the `SentimentAnalysisNLP.ipynb` file located in the `notebooks` directory.

```sh
jupyter notebook notebooks/SentimentAnalysisNLP.ipynb
```

### Training and Evaluating Models

Follow the steps in the notebook to:

- **Preprocess the data:** Clean and process the tweet text.
- **Visualize the data:** Generate word clouds and bar plots for common words.
- **Vectorize the text:** Transform text data using TF-IDF and Count Vectorizer.
- **Train and evaluate models:** Train Logistic Regression, Random Forest, and XGBoost models. Evaluate their performance using various metrics.
- **Save and load models:** Save the trained models to the models directory and load them for predictions.

![wordcloud](https://github.com/user-attachments/assets/830bf0b4-7943-49a6-b52f-1454c543ec32)

### Making Predictions

Use the trained models to make predictions on new tweet data. Example tweets and their sentiment predictions are provided in the notebook.

## Output form the best model
![output](https://github.com/user-attachments/assets/227e7220-e337-46b5-8adf-fc5d2fd72f4e)

## Additional Information

- **Code Comments:** Every step in the notebook is thoroughly commented. Each code cell contains comments explaining the purpose and functionality of the code.
- **Multiple models and dataset processing used:** I used more than one specific workflow and model just to demonstrate different solutions and methods.

## Contributing

I welcome contributions to the SentimentAnalysisNLP project! If you have any ideas, bug reports, or improvements, feel free to submit a pull request or open an issue.
