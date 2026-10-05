# Housing Prices 

A business intelligence analysis of California housing districts: data modeling, exploratory analysis, linear regression to predict median house value, logistic regression to flag premium districts, and an A/B test of the coastal vs. inland price premium. Built for INF-330: Business Intelligence.
Lecturer: Sokkhey Phauk
Group 4: Suos Sovanrith, Chheav Kimheng, Pov Visal, Chhin Sokhom Sidhisakdeh, Sam Chankroesna

**Notebook:** `BusinessIntelligenceFinalProject.ipynb`
**Repo:** `https://github.com/Sidhisakdeh/Housing_prices`

## Business problem

The housing market is economically stratified. Investors and planners need data-driven tools to identify high-value properties and pricing drivers before committing capital. The project is framed around four KPIs:

| KPI                              | Target                  |
|----------------------------------|-------------------------|
| Median house value prediction    | Within ±$60K (RMSE)     |
| High-value classification recall | ≥ 80%                   |
| Model R²                         | ≥ 0.60                  |
| Coastal premium (A/B test)       | Statistically significant |

## Tech stack

- Python 3 (Jupyter / Google Colab)
- pandas + NumPy (cleaning and feature preparation)
- Matplotlib + Seaborn (visualization)
- scikit-learn (linear/polynomial/logistic regression, one-hot encoding, metrics)
- SciPy (Welch's t-test, Chi-Square test)

## Dataset

[California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) on Kaggle: one row per census block group, based on the 1990 California census.

| Property          | Value                                                              |
|-------------------|--------------------------------------------------------------------|
| Raw rows          | 20,640 (10 columns)                                                |
| After cleaning    | 20,433 (207 null `total_bedrooms` rows dropped)                    |
| Regression set    | 19,475 (rows with the $500,001 price cap removed)                  |
| Target            | `median_house_value`                                               |
| Features          | location, housing age, rooms, bedrooms, population, households, `median_income`, `ocean_proximity` |

The dataset isn't included in this repo. Download it from the link above and keep the file named `housing.csv`.

## Getting started

**Google Colab**

1. Open `BusinessIntelligenceFinalProject.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `housing.csv` to the session.
3. Run the cells top to bottom.

**Locally**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook BusinessIntelligenceFinalProject.ipynb
```

If `housing.csv` isn't in the same folder as the notebook, adjust the path in the load cell.

## How the notebook works

**I. Data architecture & modeling** — a single-fact-table star schema (location, housing, and proximity dimensions around a `FACT_HousingPrices` table) and a Power Query-style ETL: load the CSV, drop null `total_bedrooms` rows, cast integer columns, encode `ocean_proximity` as a category, and handle the capped house values.

**II. Exploratory data analysis** — feature distributions, a geographic scatter of prices (Bay Area and LA coast stand out), income vs. house value (Pearson r = 0.643 without capped rows), and house value by ocean proximity.

**III. Linear regression** — three models on an 80/20 split (`random_state=42`) of the cap-removed data:

| Model                          | Features                    |
|--------------------------------|-----------------------------|
| SLR — simple linear regression | `median_income` only        |
| PLR — polynomial (degree 2)    | `median_income`, income²    |
| MLR — multiple linear regression | all features + one-hot `ocean_proximity` |

**IV. Logistic regression** — classifies districts at the price cap (`median_house_value == 500001`) using `class_weight='balanced'` and a stratified split. A Chi-Square test first checks that `ocean_proximity` is associated with high-value status.

**V. A/B test** — Welch's t-test (one-tailed) of coastal vs. inland house values at α = 0.05. Coastal = NEAR BAY, <1H OCEAN, NEAR OCEAN; inland = INLAND.

**VI. Business dashboard** — a four-panel summary: KPI scorecard, model R² comparison, A/B test means, and MLR actual vs. predicted.

## Results

### Regression (test set, n = 3,895)

| Model | R²     | RMSE      |
|-------|--------|-----------|
| SLR   | 0.4303 | $74,280   |
| PLR   | 0.4313 | $74,215   |
| MLR   | 0.6173 | $60,881   |

MLR (adjusted R² = 0.6161) explains about 62% of house value variance. Adding an income² term barely helps; the gain comes from location and the other features. Top MLR coefficients: `ocean_proximity_ISLAND` (+$165K), `median_income` (+$38.5K per unit), `ocean_proximity_INLAND` (−$38.5K), `longitude`, and `latitude`.

### Classification (test set, n = 4,087)

Logistic regression accuracy is 88.16%.

| Class                  | Precision | Recall | F1   | Support |
|------------------------|-----------|--------|------|---------|
| Normal (< $500K)       | 0.99      | 0.88   | 0.93 | 3,895   |
| High-value (≥ $500K)   | 0.27      | 0.88   | 0.41 | 192     |

### A/B test: coastal vs. inland

| Metric            | Coastal   | Inland    |
|-------------------|-----------|-----------|
| Sample size       | 13,001    | 6,469     |
| Mean house value  | $226,762  | $123,331  |
| Median house value | $211,300 | $108,300  |

Coastal homes average $103,430 more (83.9% higher). Welch's t = 89.69 with p < 0.001, so H₀ is rejected: the coastal premium is statistically significant.

### KPI scorecard

| KPI                   | Target       | Achieved        | Status                  |
|-----------------------|--------------|-----------------|-------------------------|
| Prediction error      | ≤ $60K RMSE  | $60,881 (MLR)   | Narrowly missed         |
| High-value recall     | ≥ 80%        | 88%             | Exceeded                |
| Model R²              | ≥ 0.60       | 0.617           | Met                     |
| Coastal premium A/B   | Significant  | p < 0.001       | Confirmed               |

## Key findings

1. **Income is the strongest single predictor** — but on its own it explains only about 43% of price variance.
2. **Location matters as much as income** — adding location and proximity features lifts R² from 0.43 to 0.62.
3. **The coastal premium is real** — coastal districts average roughly 84% more than inland ones, and the difference is overwhelmingly significant.
4. **High-value districts are easy to catch but noisy to flag** — recall is 88%, but precision is only 27%, so most flagged districts aren't actually at the cap.
5. **Recommendation** — price models, commission structures, and investment portfolios should distinguish coastal from inland properties explicitly.

## Known limitations

- **Price cap** — house values are capped at $500,001 (4.69% of cleaned rows), so "high-value" in the classifier really means "at the cap," not a true price threshold. Capped rows are removed for regression, so the regression models say nothing about the premium end of the market.
- **RMSE target narrowly missed** — MLR's $60,881 is just above the $60K goal, even though the notebook's summary table marks it as met.
- **Low precision on high-value flagging** — because of the 4.69% class share and balanced class weights, only 27% of flagged districts are truly high-value. This suits "don't miss any" screening, not final decisions.
- **Linear models** — regression assumptions (linearity, homoscedasticity, residual normality) aren't formally checked, and features like `total_rooms` and `population` are heavily right-skewed.
- **ISLAND category is tiny** — it has a handful of districts, so its large coefficient shouldn't be over-interpreted.
- **Old, aggregated data** — 1990 census block groups, not individual properties, so results don't reflect today's California market.
- **Coastal grouping** — "coastal" bundles <1H OCEAN, NEAR OCEAN, and NEAR BAY, which differ in price; the test doesn't control for income or other confounders.
- **Notebook text vs. outputs** — a few figures in the notebook's markdown (for example the 4.19% cap share and the A/B table values of $258K vs. $126K, t ≈ 58.6) don't match the actual outputs. This README uses the actual outputs.
