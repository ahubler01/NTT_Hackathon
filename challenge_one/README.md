# NTT Data Hackathon - Challenge 1

## Team: 404: Team Not Found

### Overview

This challenge focused on developing a **multiforecasting model** to predict the hourly energy supply from different power plants (solar and wind) between **June 1, 2024, and June 29, 2024**. Given that the dataset contained information from about 300 plants, we implemented **ensemble-based models** rather than forecasting individual curves manually.

### Dataset

- **df_omie_labelled.csv**: Contains historical records of energy offers per plant from February 29, 2024, to June 1, 2024.
- **filtered_categories.csv**: Maps each plant code to its generation technology (Wind, Solar).
- **unit_list.csv**: Additional metadata for generation units.

### Evaluation Metrics

The forecasting models were assessed using:

- **Mean Absolute Percentage Error (MAPE)**
- **Mean Absolute Error (MAE)**
- Given that MAPE is undefined for zero values, handling zero-production periods was critical in our approach.

---

## Approach

### 1. Feature Engineering

- **Lag Feature for Energy**: Since energy production follows strong periodic patterns, we implemented a **28-day shift** for each plant to capture seasonal trends:
  ```python
  df['lag_Energia'] = df.groupby('Codigo')['Energia'].shift(24 * 28)
  ```
- **Winsorization**: Extreme values were clipped to mitigate the impact of outliers, significantly improving model robustness.
- **Temporal Features**: Extracted `hour`, `day of the week`, and `month` to capture cyclical trends.

### 2. Modeling Approaches

We benchmarked multiple models, including:

- **LSTM**: Captured long-term dependencies but required significant training time.
- **TimeFM (Pre-trained by Microsoft)**: Provided promising initial results but had limitations in handling missing values.
- **Gradient Boosting Models (XGBoost, LightGBM)**: Performed best in terms of interpretability and speed.
- **Prophet**: Considered for its ability to model seasonality, but struggled with granular hour-level predictions.

### 3. Handling Zero Values in Target

Due to the presence of zero values in energy production (causing infinite MAPE), we implemented:

- **Asymmetric Loss Functions**: Penalized overestimation and underestimation differently for XGBoost and LightGBM models.
- **Production Ceilings & Floors**: Since renewable energy (solar/wind) has predictable production limits, we applied logical constraints:

  ```python
  # Set energy to 0 for solar plants at night
  solar_time_mask = (forecast_inverted['Categoria'] == 'Solar') & ((forecast_inverted['hour'] >= 1) & (forecast_inverted['hour'] < 6))
  forecast_inverted.loc[solar_time_mask, 'Energia'] = 0

  # Set energy to 0 for wind plants at night
  wind_time_mask = (forecast_inverted['Categoria'] == 'Eolica') & ((forecast_inverted['hour'] >= 0) & (forecast_inverted['hour'] < 6))
  forecast_inverted.loc[wind_time_mask, 'Energia'] = 0
  ```

### 4. Final Model

We opted for an **ensemble of XGBoost and LightGBM**, leveraging their fast training speed and ability to capture complex dependencies:

- **XGBoost & LightGBM Ensemble**: Both models were tuned with hyperparameter optimization to maximize forecasting accuracy.
- **Stacking Strategy**: Combined predictions from different models to improve generalization.

---

## Results & Insights

- **Winsorizing** significantly improved model stability by reducing the impact of extreme values.
- **Feature engineering (especially lag features) was critical** to capturing energy consumption patterns.
- **Handling zero values** via production ceilings and floors helped mitigate MAPE-related issues.
- **XGBoost and LightGBM ensemble** outperformed deep learning-based methods in both accuracy and computational efficiency.

---

## Submission Files

1. **Challenge1_404TeamNotFound.ipynb** - Notebook with full implementation.
2. **Challenge1_404TeamNotFound.csv** - Forecast results in the required format: `Codigo`, `fechaHora`, `Energia`.

---

## Conclusion

This challenge required balancing multiple forecasting techniques, handling missing values, and mitigating infinite errors caused by zero values. Our **ensemble-based approach with XGBoost and LightGBM, combined with effective feature engineering**, provided a competitive solution for predicting renewable energy supply.
