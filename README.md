# ST211: Linear and Logistic Regression — Workshop Compendium

This repository contains supplementary workshop materials for **ST211** at the London School of Economics. The workshops are designed to accompany the lecture series. Here, I am providing some additional resources to my workshop students to build statistical intuition through computation.

## What This Is

ST211 teaches linear and logistic regression with a strong applied component. 

## Contents

| Workshop | Topic | Key concepts |
|----------|-------|--------------|
| [Workshop 1](workshop_01.md) | Introduction to R | Vectors, arrays, indexing, vectorized arithmetic, floating-point comparison, simulating from the Normal distribution, Q-Q plots, one-sample z-test |
| [Workshop 2](workshop_02.md) | Data Frames and Hypothesis Testing | Loading and inspecting data, subsetting, aggregation, `ggplot2` scatterplots and boxplots, one-sample and two-sample t-tests, statistical power |
| [Workshop 3](workshop_03.md) | Simple Linear Regression | OLS estimation, interpreting coefficients and R², residuals, the four assumptions (linearity, independence, constant variance, normality), diagnostic plots, a gallery of assumption violations using built-in datasets |
| [Workshop 4](workshop_04.md) | Multiple Linear Regression | Ceteris paribus interpretation, categorical predictors and dummy variables, interactions, the model matrix, $F$-statistic, adjusted $R^2$, ANOVA decomposition |
| [Workshop 5](workshop_05.md) | Outliers, Leverage, and Influence | Standardized residuals, the hat matrix, Cook's distance, DFFITS, Anscombe's quartet, manufacturing influential points, the decision framework for handling outliers |
| [Workshop 6](workshop_06.md) | Theory & Geometry Supplement | Set operations and probability axioms, historical origins and application of *ceteris paribus*, omitted variable bias, geometric intuition for interaction terms|
| [Workshop 7](workshop_07.md) | Transforms and Non-Linear Relationships | Centering, standardizing, log transforms on outcomes and predictors, the four model types (level-level, log-level, level-log, log-log), interpreting percentage changes, quadratic terms, diagnostic before-and-after comparisons |
| [Workshop 8](workshop_08.md) | Prediction and Cross-Validation | Log model interpretation review, the four model types, `predict()`, confidence vs. prediction intervals, in-sample vs. out-of-sample, the cat analogy for CV, MSPE, split ratios, model comparison via cross-validation, the bias-variance trade-off |
| [Workshop 9](workshop_09.md) | Introduction to Logistic Regression | Why linear regression fails for binary outcomes, probability–odds–log-odds pipeline, the Kentucky Derby odds primer, the logistic (sigmoid) function, `glm()` syntax, odds ratios, classification tables, sensitivity and specificity, the base rate trap, class imbalance and threshold selection |
| [Workshop 10](workshop_10.md) | Multiple Logistic Regression | Multiple predictors in logistic models, deviance and model comparison, predicted probabilities, the S-curve and non-linearity, risk ratios vs. odds ratios, average predictive comparisons (APC), classification tables |
| [Workshop 11](workshop_11.md) | Poisson Regression | GLMs (linear, logistic, Poisson), count data and the log link, rate ratios via exponentiation, polynomial terms for non-linear predictors, prediction of expected counts, offsets for modeling rates, overdispersion and quasipoisson, parallel interpretation across all three GLM types |

## Requirements

- **R** (≥ 4.0)
- **Packages:** `ggplot2`, `arm`, `gridExtra`, `dplyr`

Install any missing packages with:

```r
install.packages(c("ggplot2", "arm", "gridExtra", "dplyr"))
```

Some workshops use course-specific datasets distributed through the ST211 Moodle page. Where possible, exercises also use datasets built into R so that students can practice without additional downloads.

### Supplementary Datasets

This repository also includes curated datasets for hands-on regression practice:

- **[🏍️ Motorcycle Specs 2020](moto_data/README.md)** — 656 motorcycles from 15 major brands with engine specs, dimensions, and a custom binary outcome (`trevors_fav`) for logistic regression. Hopefully you find this helpful for both linear and logistic regression. See the [dataset README](moto_data/moto_README.md) for full documentation.
