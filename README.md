# Expected Weighted On-Base Average (xwOBA) Prediction

A machine learning approach to modeling Expected Weighted On-Base Average (xwOBA) using MLB Statcast data from the 2023 and 2024 seasons.

## Overview

This project develops a custom xwOBA prediction model using machine learning algorithms to estimate baseball player performance based on batted ball characteristics and player metrics. The model is trained on 2023 MLB data and validated against 2024 season performance, with comparisons to Baseball Savant's official xwOBA metric.

## Dataset

The analysis utilizes MLB Statcast data retrieved via the `pybaseball` library:
- **Training Data**: 2023 season (183,773 plate appearances)
- **Testing Data**: 2024 season (181,996 plate appearances)
- **Data Source**: Baseball Savant Statcast database

### Features

The model incorporates the following batted ball and player characteristics:
- **Launch Speed**: Exit velocity of batted ball (mph)
- **Launch Angle**: Vertical angle of batted ball trajectory (degrees)
- **Bat Speed**: Speed of bat at contact (mph)
- **Swing Length**: Length of bat path through hitting zone (feet)
- **Sprint Speed**: Player's maximum sprint speed (ft/sec) _(v2 model only)_

### Target Variable

The model predicts **Total Bases (TB)** as a classification problem with 5 classes:
- 0: Out
- 1: Single
- 2: Double
- 3: Triple
- 4: Home Run

These predictions are then converted to xwOBA using MLB's official wOBA weights.

## Methodology

### Data Processing

1. Filter for balls hit into play (`description == 'hit_into_play'`)
2. Remove excluded events (sacrifice bunts, catcher interference, truncated PAs)
3. Handle missing values and merge sprint speed data
4. Split 2023 data into 90/10 train/validation sets

### Models Evaluated

Four machine learning algorithms were trained and compared:

| Model | Training AUC | Validation AUC | Notes |
|-------|-------------|----------------|-------|
| **Random Forest** | 0.905 | 0.877 | Selected for final model |
| XGBoost | 0.926 | 0.885 | Highest performance |
| LightGBM | 0.926 | 0.883 | Similar to XGBoost |
| SVM | 0.859 | 0.849 | Lower performance |

**Random Forest** was selected as the final model for its balance of performance and interpretability.

### Model Configuration

```python
RandomForestClassifier(
    max_depth=8,
    n_estimators=200,
    random_state=1126
)
```

## Results

### Model Performance (v1 - 4 features)

Correlation with actual 2024 wOBA (455 qualified batters):
- **Custom xwOBA**: R² = 0.66
- **Baseball Savant xwOBA**: R² = 0.74

The custom model shows moderate correlation with actual performance, though it tends to predict higher wOBA values compared to the Savant system.

### Model Performance (v2 - 5 features including sprint speed)

After adding sprint speed as a feature, the model diverged further from both actual wOBA and Savant's xwOBA, suggesting sprint speed may introduce noise for this particular modeling approach.

## Key Findings

1. **Model Characteristics**: The custom xwOBA model is more correlated with Savant's xwOBA than with actual wOBA, indicating it captures similar expected value patterns
2. **Distribution Shift**: The model tends to predict higher wOBA values on average compared to the Savant system
3. **Feature Impact**: Adding sprint speed as a feature decreased model alignment with both actual and expected outcomes

## Project Structure

```
├── data_fetch.ipynb          # Data retrieval from Baseball Savant
├── woba_models.ipynb         # Initial 4-feature model development
├── woba_models_v2.ipynb      # Enhanced 5-feature model with sprint speed
├── data/
│   ├── 2023_hit_data.csv
│   ├── 2024_hit_data.csv
│   ├── 2023_excepted_data.csv
│   ├── 2024_excepted_data.csv
│   ├── 2023_sprint_speed_data.csv
│   └── 2024_sprint_speed_data.csv
└── README.md
```

## Technologies

- **Python 3.x**
- **Data Collection**: pybaseball
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Usage

### 1. Data Collection

Run [data_fetch.ipynb](data_fetch.ipynb) to download the latest Statcast data:
```python
from pybaseball import statcast, statcast_batter_expected_stats, statcast_sprint_speed

# Fetch Statcast data
data = statcast('2024-03-28', '2024-09-30')
```

### 2. Model Training

Open [woba_models.ipynb](woba_models.ipynb) or [woba_models_v2.ipynb](woba_models_v2.ipynb) to:
- Perform exploratory data analysis
- Train and evaluate models
- Generate predictions and comparisons

## Future Improvements

- Incorporate additional features (pitch type, count, fielder positioning)
- Explore deep learning architectures
- Implement time-series cross-validation
- Analyze feature importance and interpretability
- Calibrate probability predictions to reduce systematic bias

## References

- [Baseball Savant](https://baseballsavant.mlb.com/)
- [FanGraphs wOBA Guide](https://library.fangraphs.com/offense/woba/)
- [pybaseball Documentation](https://github.com/jldbc/pybaseball)