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

Further workshops will be added as the course progresses.

## Requirements

- **R** (≥ 4.0)
- **Packages:** `ggplot2`, `arm`, `gridExtra`, `dplyr`

Install any missing packages with:

```r
install.packages(c("ggplot2", "arm", "gridExtra", "dplyr"))
```

Some workshops use course-specific datasets distributed through the ST211 Moodle page. Where possible, exercises also use datasets built into R so that students can practice without additional downloads.
