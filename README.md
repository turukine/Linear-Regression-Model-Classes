Project Goal
Master supervised learning with linear models, regularization techniques, and quality metrics for regression problems.

Core Concepts
Regression Problems
Predicting continuous values using linear relationships:

Linear Regression: $f(x) = \mathbf{w}^T \mathbf{x} + b$

Loss Function: Mean Squared Error (MSE)

Optimization: Gradient Descent

Key Challenges
Overfitting: Model learns noise instead of patterns

Underfitting: Model fails to capture data patterns

Bias-Variance Tradeoff: Balancing model complexity

Regularization Techniques
Ridge Regression (L2): $R(\theta) = |\theta|_2^2$

Lasso Regression (L1): $R(\theta) = |\theta|_1$

Elastic Net: Combines L1 and L2 regularization

Quality Metrics
MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

R² (R-squared coefficient)

MAPE (Mean Absolute Percentage Error)

Practical Tasks
1. Theoretical Questions
Derive analytical solution for linear regression

Explain L1/L2 regularization effects

Describe feature selection with L1

Handle nonlinear dependencies

2. Data Preparation
python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
Steps:

Load and preprocess rental listing data

Process "Features" column into binary indicators

Create feature set with 22 dimensions

3. Model Implementation
Custom implementations:

Linear Regression with SGD

Ridge Regression

Lasso Regression

Elastic Net

Compare with sklearn implementations

4. Feature Engineering
Implement MinMaxScaler and StandardScaler

Train models with normalized features

Create polynomial features (degree 10)

5. Model Evaluation
Calculate MAE, RMSE, R² metrics

Compare custom vs sklearn implementations

Identify best performing model

Analyze model stability

6. Advanced Techniques
Target variable transformation (log scaling)

Outlier handling

Batch/mini-batch training

Analytical solution implementation

Key Deliverables
Jupyter Notebook with all implementations

Comparison tables for model metrics

Analysis of regularization effects

Identification of optimal model configuration
