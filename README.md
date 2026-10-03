# nit_warangal_nrif_rank_predictor
Analysis of NIT Warangal's NIRF rankings (2016-2025) with ML models to predict institute rank.
# NIT Warangal NIRF Ranking Analysis

I wanted to see how NIT Warangal has done in the NIRF engineering rankings over the years, and whether its rank can be predicted from the NIRF parameter scores. This notebook does both: it plots the trends from 2016 to 2025, then trains a few regression models on the 2025 data and compares them.

## Data

The data is a CSV of NIRF engineering rankings, one row per institute, with columns for each year:

- `RPC` - Research and Professional Practice
- `GO` - Graduation Outcomes
- `OI` - Outreach and Inclusivity
- `PERCEPTION` - Perception
- `TLR` - Teaching, Learning and Resources
- `Score` and `Rank` - overall score and rank

Each column is suffixed with the year, e.g. `RPC(2025)`. The file is called `NIRF Ranking 2017.csv` but it has data up to 2025. It isn't uploaded here, so keep it in the same folder as the notebook before running.

## What the notebook does

1. Loads the CSV and picks out NIT Warangal.
2. Plots its RPC, GO, OI, Perception, overall score and rank from 2016 to 2025.
3. Uses RPC, GO, OI and Perception (2025) to predict `Rank(2025)`, with an 80/20 train-test split.
4. Compares Linear Regression, Decision Tree, Random Forest and SVR.
5. Predicts a rank for NIT Warangal using the best model.

For the decision tree I tried depths from 2 to 10. A full-depth tree got a training R² of 1.0 but only about 0.65 on the test set, so I limited it to depth 3. SVR was tuned with GridSearchCV (5-fold) after scaling the features.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| SVR | 8.19 | 11.64 | 0.831 |
| Random Forest | 9.38 | 11.69 | 0.829 |
| Linear Regression | 9.18 | 12.97 | 0.790 |
| Decision Tree (depth 3) | 11.94 | 15.67 | 0.694 |

SVR did best, with C=1000, gamma=0.01 and epsilon=0.5. Its predicted rank for NIT Warangal is about 70, compared with the actual 2025 rank of 82.

## Limitations

- The dataset is small, so the numbers change quite a bit with a different split.
- The "2026" prediction just feeds the 2025 scores into a model trained on 2025 ranks. It shows what rank those scores usually get, not what will actually happen next year.
- TLR isn't used as a feature.

## Running it

You need Python 3 with pandas, numpy, matplotlib and scikit-learn:

```
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook NIT_W.ipynb
```

It also runs on Google Colab if you upload the notebook and the CSV.
