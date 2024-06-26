# Fight Forecaster AI

This project aims to predict the outcomes of UFC fights using machine learning techniques, specifically XGBoost.

## Overview

The system uses historical UFC fight data to train a model that can predict the winner of future fights. It employs advanced techniques such as:

- XGBoost classifier with GPU acceleration
- Hyperparameter optimization using Optuna
- SHAP (SHapley Additive exPlanations) for model interpretability
- Dynamic feature selection
- Web scraping for up-to-date fight odds

## Features

- Data preprocessing and cleaning
- Web scraping of fight odds from BestFightOdds
- Model training with XGBoost
- Hyperparameter tuning with Optuna
- Model evaluation using accuracy and AUC-ROC scores
- Visualization of learning curves
- SHAP analysis for feature importance
- Ability to retrain the model with top N most important features
- Interactive pausing and resuming of the training process
- Prediction of fight outcomes for specific matchups

## Requirements

- Python 3.x
- NumPy
- Pandas
- Optuna
- XGBoost
- Scikit-learn
- Matplotlib
- SHAP
- Selenium (for web scraping)

## Installation

1. Clone this repository
2. Install the required packages:
   
   pip install -r requirements.txt
   

## Usage

### Data Preparation

1. Run the data preparation script:
   
   python data_cleaner.py
   
   This will process the raw UFC stats and create the necessary datasets.

2. To scrape fight odds:
   
   python odds_scraper.py
   

### Model Training

Run the main training script:

python main.py

During execution, you can:
- Press 'p' to pause/resume the optimization process
- Press 'q' to quit the program

### Making Predictions

To predict the outcome of a specific matchup:

1. Run the prediction script:
   
   python predict.py
   

2. Enter the names of the two fighters when prompted.

3. The script will output the predicted winner and the probability of winning.

## Structure

- data_cleaner.py: Contains functions for data preprocessing and feature engineering
- odds_scraper.py: Web scraping script for collecting fight odds
- main.py: Main script for model training and optimization
- predict.py: Script for making predictions on specific matchups

## Model Performance

The model's performance is evaluated using accuracy and AUC-ROC scores. Learning curves are plotted to visualize the training process.

## Feature Importance

SHAP values are used to determine the most influential features in predicting fight outcomes. This information can be used to refine the model and gain insights into what factors most affect fight results.

## Future Improvements

- Predict if fight will go the distance
- Predict how will fight end

## Copyright and License

Copyright (c) 2024 William Qin Shen. All rights reserved.

This software and associated documentation files (Fight Forecaster AI) are the exclusive property of William Qin Shen. The Software is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties.

No part of this Software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright owner, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law.

For permission requests, please contact the copyright owner at:

williambillqinshen@gmail.com

Unauthorized use, reproduction, or distribution of this Software, or any portion of it, may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under law.

## Contact

William Qin Shen

williambillqinshen@gmail.com
