# Fight Forecaster AI UFC

Fight Forecaster AI is a tooling stack for preparing UFC fight data, training predictive models, and reviewing betting scenarios. It centralizes scraping, cleaning, feature engineering, model training, and evaluation so the full workflow can be run from one repository.

## Project scope

The repository focuses on practical fight forecasting tasks:

- **Data acquisition** – Scrapers under `src/data_processing/scraping` collect fighter statistics, bout histories, and betting lines from public sources.
- **Data preparation** – Cleaning and feature modules in `src/data_processing/cleaning` and `src/data_processing/features` merge raw tables, engineer Elo-style ratings, flag recent form, and attach betting context.
- **Modeling** – Code in `src/models` builds and tunes XGBoost models with Optuna-based cross-validation, then serializes artifacts for reuse.
- **Matchup scoring** – CLI utilities assemble head-to-head matchup files, score them with saved models, and generate bankroll projections.
- **Reporting** – Evaluation scripts summarize calibration, bankroll growth, and other diagnostics for comparison across runs.

## Repository layout

```
├── data/
│   ├── raw/            # Raw scrapes directly from data sources
│   ├── processed/      # Curated tables produced by the processors
│   ├── train_test/     # Train/test splits ready for model training
│   └── live_data/      # Latest cards and ad-hoc matchup files
├── notebooks/          # Exploratory analyses and scraper prototypes
├── outputs/            # Plots, calibration reports, bankroll summaries
├── saved_models/       # Persisted encoders and trained model artifacts
└── src/
    ├── data_processing/
    │   ├── scraping/   # Selenium and HTTP scrapers
    │   ├── cleaning/   # FightDataProcessor and helpers
    │   └── features/   # Feature engineering utilities
    └── models/
        ├── create_predictions/
        ├── model_evaluation/
        ├── predict_testset/
        └── xgboost_optimizer/
```

## Environment setup

- Python 3.10 or newer is recommended.
- Chrome or Chromium with a matching ChromeDriver binary must be installed for Selenium-based scrapers.
- GPU hardware is optional but shortens Optuna searches for the XGBoost models.

Create and populate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost optuna matplotlib seaborn rich tqdm rapidfuzz selenium selenium-stealth beautifulsoup4 requests
```

Install any optional visualization packages (for example `shap` or `plotly`) as needed.

Ensure the `data` directory includes the expected subfolders (`raw`, `processed`, `train_test`, `live_data`) before running the processors so intermediate files have a destination.

## Typical workflow

1. **Scrape source data**
   - Run the modules in `src/data_processing/scraping` to capture fighter statistics, results, and odds.
   - Selenium scrapers such as `BestfightoddsScraper.py` and `Tapology_scraping.py` require ChromeDriver.
   - HTTP-only scripts such as `scrape_ufc_stats_library.py` rely on `requests` and `BeautifulSoup`.

2. **Clean and engineer features**
   - Use `FightDataProcessor` in `src/data_processing/cleaning/data_cleaner.py` to merge raw tables, create fight-level aggregates, compute Elo ratings, and generate curated CSV outputs under `data/processed` and `data/train_test`.

3. **Create matchup files**
   - Build head-to-head inputs for upcoming fights with:

     ```bash
     python src/models/create_predictions/main.py \
         --fighters "Fighter A" "Fighter B" \
         --odds 110 -120 100 -110 \
         --date 2025-08-09
     ```

   - The command looks up fighter histories, applies encoders, and writes a matchup file ready for scoring.

4. **Train models**
   - Launch nested cross-validation and hyperparameter search with:

     ```bash
     python src/models/xgboost_optimizer/main.py
     ```

   - Results, best models, and Optuna study details are saved under `saved_models/xgboost`.

5. **Score and evaluate**
   - Generate predictions with the console app:

     ```bash
     python src/models/predict_testset/main.py
     ```

   - Review bankroll simulations and calibration plots with:

     ```bash
     python src/models/model_evaluation/unified_evaluator.py
     ```

   - Outputs are written to the `outputs` directory for comparison across runs.

## Notebooks

Notebooks prefixed with `scrape_ufc_stats_` document exploratory scraping and feature checks. They complement the production modules by capturing prototypes and manual validations.

## Copyright and License

Copyright (c) 2024 William Qin Shen. All rights reserved.

This software and associated documentation files (Fight Forecaster AI) are the exclusive property of William Qin Shen. The Software is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties.

No part of this Software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright owner, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law.

For permission requests, please contact the copyright owner at:

williambillqinshen@gmail.com

Unauthorized use, reproduction, or distribution of this Software, or any portion of it, may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under law.

## Contact

William Qin Shen  
[williambillqinshen@gmail.com](mailto:williambillqinshen@gmail.com)
