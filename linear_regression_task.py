# ============================================================
# LINEAR REGRESSION TASK
# ============================================================

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ============================================================
# PART 1 - FIRST LINEAR REGRESSION MODEL
# ============================================================

# Load tips dataset
tips = sns.load_dataset("tips")

print("=" * 70)
print("PART 1 - FIRST LINEAR REGRESSION MODEL")
print("=" * 70)

print("\nFirst 5 rows:")
print(tips.head())

print("\nDataset shape:")
print(tips.shape)

print("\nColumns:")
print(tips.columns.tolist())


# Prepare X and y
X = tips[["total_bill"]]
y = tips["tip"]


# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Print first 10 predictions vs actual values
comparison = pd.DataFrame({
    "Actual Tip": y_test.values[:10],
    "Predicted Tip": y_pred[:10]
})

print("\nFirst 10 predictions vs actual values:")
print(comparison)


# Calculate R²
r2 = model.score(X_test, y_test)

print("\nR² Score:")
print(r2)
# ============================================================
# PART 2 - INTERPRETING MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - INTERPRETING MODEL COEFFICIENTS")
print("=" * 70)


# Print coefficient and intercept
print("\nModel Coefficient (Slope):")
print(model.coef_[0])

print("\nModel Intercept:")
print(model.intercept_)


# Business interpretation
slope = model.coef_[0]

print("\nBusiness Interpretation:")
print(
    f"Every $1 increase in the total bill is associated with "
    f"an expected increase of ${slope:.3f} in the tip."
)


# ------------------------------------------------------------
# Second Model: Predict tip using table size
# ------------------------------------------------------------

X_size = tips[["size"]]
y_size = tips["tip"]


# Split data into training and testing sets
X_train_size, X_test_size, y_train_size, y_test_size = train_test_split(
    X_size,
    y_size,
    test_size=0.20,
    random_state=42
)


# Create and train second model
model_size = LinearRegression()
model_size.fit(X_train_size, y_train_size)


# Calculate predictions
y_pred_size = model_size.predict(X_test_size)


# Calculate R²
r2_size = model_size.score(X_test_size, y_test_size)


print("\nSecond Model - Using Table Size:")
print("Coefficient:", model_size.coef_[0])
print("Intercept:", model_size.intercept_)
print("R² Score:", r2_size)


# ------------------------------------------------------------
# Compare the two models
# ------------------------------------------------------------

print("\nModel Comparison:")
print(f"R² using total_bill: {r2:.4f}")
print(f"R² using size:       {r2_size:.4f}")

if r2 > r2_size:
    print("\nConclusion:")
    print(
        "Total bill is a stronger single predictor of tip than "
        "table size because it has the higher R²."
    )
else:
    print("\nConclusion:")
    print(
        "Table size is a stronger single predictor of tip than "
        "total bill because it has the higher R²."
    )


# ------------------------------------------------------------
# Business Question
# ------------------------------------------------------------

print("\nBusiness Question:")
print(
    "Are larger tables guaranteed to leave higher tips?"
)

print(
    f"Based on the R² of {r2_size:.4f}, table size alone does not "
    "provide a sufficiently strong basis for predicting tip values. "
    "A larger table may be associated with higher tips, but the "
    "relationship is not strong enough to guarantee higher tips."
)
# ============================================================
# PART 3 - REGRESSION FOR SIMPLE TREND FORECASTING
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - REGRESSION FOR SIMPLE TREND FORECASTING")
print("=" * 70)


# Load flights dataset
flights = sns.load_dataset("flights")


# Create monthly date column
flights["date"] = pd.to_datetime(
    flights["year"].astype(str) + "-" +
    flights["month"].astype(str) + "-01"
)


# Sort by date
flights = flights.sort_values("date")


# Create sequential month number
flights["month_number"] = np.arange(1, len(flights) + 1)


print("\nFirst 5 rows of monthly data:")
print(flights[["date", "passengers", "month_number"]].head())


# ------------------------------------------------------------
# Train Linear Regression model
# ------------------------------------------------------------

X_time = flights[["month_number"]]
y_time = flights["passengers"]


trend_model = LinearRegression()
trend_model.fit(X_time, y_time)


# R² of trend model
trend_r2 = trend_model.score(X_time, y_time)


print("\nTrend Model Coefficient:")
print(trend_model.coef_[0])

print("\nTrend Model Intercept:")
print(trend_model.intercept_)

print("\nTrend Model R²:")
print(trend_r2)


# ------------------------------------------------------------
# Predict the existing data
# ------------------------------------------------------------

flights["trend_prediction"] = trend_model.predict(X_time)


# ------------------------------------------------------------
# Forecast next 6 months
# ------------------------------------------------------------

last_month_number = flights["month_number"].max()

future_month_numbers = np.arange(
    last_month_number + 1,
    last_month_number + 7
)


future_predictions = trend_model.predict(
    future_month_numbers.reshape(-1, 1)
)


# Create future dates
last_date = flights["date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)


future_forecast = pd.DataFrame({
    "date": future_dates,
    "month_number": future_month_numbers,
    "predicted_passengers": future_predictions
})


print("\n6-Month Forecast:")
print(future_forecast)


# ------------------------------------------------------------
# Plot original data + trend line + future forecast
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    flights["date"],
    flights["passengers"],
    label="Actual Passengers"
)

plt.plot(
    flights["date"],
    flights["trend_prediction"],
    label="Regression Trend Line"
)

plt.plot(
    future_forecast["date"],
    future_forecast["predicted_passengers"],
    marker="o",
    linestyle="--",
    label="6-Month Forecast"
)

plt.title("Passenger Trend and 6-Month Linear Regression Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# PART 4 - INTEGRATED BUSINESS PROJECT
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - INTEGRATED BUSINESS PROJECT")
print("=" * 70)


# Business Question:
# Does a higher total bill help predict a higher tip?


# Build a simple regression model
X_business = tips[["total_bill"]]
y_business = tips["tip"]


business_model = LinearRegression()
business_model.fit(X_business, y_business)


# Model coefficient and intercept
business_slope = business_model.coef_[0]
business_intercept = business_model.intercept_


# R²
business_r2 = business_model.score(X_business, y_business)


print("\nBusiness Question:")
print("Does a higher total bill help predict a higher tip?")


print("\nRegression Equation:")
print(
    f"Tip = {business_slope:.4f} × Total Bill + "
    f"{business_intercept:.4f}"
)


print("\nCoefficient:")
print(f"{business_slope:.4f}")


print("\nIntercept:")
print(f"{business_intercept:.4f}")


print("\nR²:")
print(f"{business_r2:.4f}")


# Business interpretation
print("\nBusiness Interpretation:")
print(
    f"For every $1 increase in the total bill, the predicted tip "
    f"increases by approximately ${business_slope:.3f}."
)


print(
    f"The model explains approximately {business_r2 * 100:.1f}% "
    "of the variation in tip values."
)


# Decision-making interpretation
print("\nDecision-Making Interpretation:")

if business_r2 >= 0.50:
    print(
        "The relationship is meaningful enough to be useful as one "
        "business indicator, but it should not be used alone for "
        "individual tip predictions."
    )
else:
    print(
        "The relationship is relatively weak, so total bill should "
        "not be used alone for individual tip predictions."
    )


# ------------------------------------------------------------
# Final Two-Paragraph Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL BUSINESS REPORT")
print("=" * 70)


report_paragraph_1 = (
    f"The analysis examined whether total bill value can be used "
    f"to predict tip amount in the restaurant dataset. A simple "
    f"Linear Regression model was built using total_bill as the "
    f"independent variable and tip as the dependent variable. "
    f"The model produced a coefficient of {business_slope:.4f}, "
    f"meaning that a $1 increase in the total bill is associated "
    f"with an expected increase of approximately ${business_slope:.3f} "
    f"in the tip. The model achieved an R² of {business_r2:.4f}, "
    f"indicating that total bill explains approximately "
    f"{business_r2 * 100:.1f}% of the variation in tip values."
)

report_paragraph_2 = (
    "From a business perspective, total bill is a useful indicator "
    "for understanding and estimating tipping behavior, but it "
    "should not be treated as the only factor affecting tips. "
    "The comparison with table size showed that total bill had a "
    "much stronger relationship with tip than table size alone. "
    "Therefore, restaurant management could use bill value as one "
    "input when analyzing tipping patterns, while considering other "
    "factors such as table size, day, time, and customer behavior "
    "for a more complete analysis."
)


print("\nParagraph 1:")
print(report_paragraph_1)

print("\nParagraph 2:")
print(report_paragraph_2)