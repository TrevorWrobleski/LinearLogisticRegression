# Workshop 4: Multiple Linear Regression, Categorical Predictors, and Interactions

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop

---

## 1. Setup

```r
library(ggplot2)
library(arm)
library(gridExtra)
library(car)

data(mtcars)
mtcars$am_factor  <- factor(mtcars$am, labels = c("Auto", "Manual"))
mtcars$cyl_factor <- factor(mtcars$cyl)
```

We use `mtcars` throughout this workshop. The key variables:

| Variable | Type | Description |
|----------|------|-------------|
| `mpg` | Continuous | Fuel efficiency (miles per gallon) — our outcome |
| `wt` | Continuous | Weight (thousands of lbs) |
| `am` | Binary | Transmission: 0 = automatic, 1 = manual |
| `cyl` | Categorical | Number of cylinders: 4, 6, or 8 |

The `factor()` calls are essential. Without them, R treats `am` as a number
and `cyl` as a number. Regression on a numeric `cyl` assumes a *linear* effect
(going from 4 to 6 cylinders has the same effect as going from 6 to 8). Factor
encoding makes no such assumption — each level gets its own coefficient.

---

## 2. From Simple to Multiple Regression

### 2.1 The model

In simple linear regression we had one predictor. Multiple linear regression
(MLR) extends this to $k$ predictors:

$$
Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \cdots + \beta_k X_{ki} + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)
$$

The estimation method is the same — OLS minimizes the sum of squared
residuals — but the interpretation of each coefficient changes in an important way.

### 2.2 The ceteris paribus interpretation

In simple regression, $\beta_1$ captures the *total* association between $X$
and $Y$. In MLR, $\beta_j$ captures the association between $X_j$ and $Y$
**holding all other predictors constant**. This is sometimes called the
*ceteris paribus* ("all else equal") interpretation.

Consider a model of youth unemployment ($Y$) as a function of GDP ($X_1$) and
inequality as measured by the Gini coefficient ($X_2$):

$$
\hat{Y} = 11.51 - 1.65 \cdot \text{GDP} + 0.52 \cdot \text{GINI}
$$

The coefficient $-1.65$ means: comparing two countries with the *same*
inequality, the one with GDP one unit higher is expected to have youth
unemployment 1.65 percentage points lower. It does not mean that increasing GDP
by one unit *causes* a 1.65-point drop — that requires a causal argument. 

Note also that the intercept (11.51) is the predicted unemployment when both
GDP and GINI equal zero. In most applications this is outside the range of the
data and has no substantive meaning. The intercept is **often** an anchor for
the plane, not a quantity you can interpret literally.

---

## 3. Categorical Predictors: Parallel Lines

### 3.1 Adding a binary predictor

Let's reconsider modeling fuel efficiency, this time as a function of weight and transmission type:

$$
\text{mpg}_i = \beta_0 + \beta_1 \cdot \text{wt}_i + \beta_2 \cdot \mathbb{1}[\text{Manual}]_i + \varepsilon_i
$$

where $\mathbb{1}[\text{Manual}]$ is an indicator variable equal to 1 for
manual transmission and 0 for automatic.

```r
mod1 <- lm(mpg ~ wt + am_factor, data = mtcars)
display(mod1)
```

**Interpreting the output.** Suppose the estimates are:

| Coefficient | Estimate | SE |
|---|---|---|
| (Intercept) | 37.32 | 3.05 |
| wt | −5.35 | 0.79 |
| am\_factorManual | −0.02 | 1.55 |

- $\hat{\beta}_0 = 37.32$: predicted mpg for an automatic car weighing 0
  lbs. (Meaningless in practice, but it anchors the line.)
- $\hat{\beta}_1 = -5.35$: for every additional 1,000 lbs of weight, mpg
  decreases by 5.35, **holding transmission constant**.
- $\hat{\beta}_2 = -0.02$: manual cars get 0.02 fewer mpg than automatics
  **of the same weight**. This is tiny and not statistically significant
  (the coefficient is much smaller than $2 \times \text{SE}$ - a quick estimate for significance).

### 3.2 What parallel lines mean

This model forces the slope of weight to be identical for both transmission
types. The only difference between groups is the intercept — the lines are
shifted vertically but run in parallel.

We can derive the two group equations from the single model:

- **Automatic** ($\mathbb{1}[\text{Manual}] = 0$): $\hat{y} = 37.32 - 5.35 \cdot \text{wt}$
- **Manual** ($\mathbb{1}[\text{Manual}] = 1$): $\hat{y} = (37.32 - 0.02) - 5.35 \cdot \text{wt} = 37.30 - 5.35 \cdot \text{wt}$

```r
ggplot(mtcars, aes(x = wt, y = mpg, color = am_factor)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_abline(intercept = coef(mod1)[1], slope = coef(mod1)[2],
              color = "steelblue", linewidth = 0.8, linetype = "dashed") +
  geom_abline(intercept = coef(mod1)[1] + coef(mod1)[3], slope = coef(mod1)[2],
              color = "firebrick", linewidth = 0.8, linetype = "dashed") +
  scale_color_manual(values = c("Auto" = "steelblue", "Manual" = "firebrick")) +
  labs(title = "Model 1: Additive Model (Parallel Lines)",
       subtitle = "Same slope for both groups; only the intercept differs",
       x = "Weight (1000 lbs)", y = "Miles per Gallon", color = "Transmission") +
  theme_minimal(base_size = 13)
```

### 3.3 Multi-level categorical predictors

When a categorical variable has more than two levels, R creates $k - 1$ dummy
variables, where $k$ is the number of levels. One level serves as the
**baseline** (which we'll typically refer to as the "reference"); 
the coefficients for the remaining levels represent
differences from that baseline.

```r
mod3 <- lm(mpg ~ cyl_factor, data = mtcars)
display(mod3)
```

If `cyl_factor` has levels 4, 6, 8, R chooses 4 as the baseline (the first
level). The model is:

$$
\text{mpg}_i = \beta_0 + \beta_2 \cdot \mathbb{1}[\text{6 cyl}]_i + \beta_3 \cdot \mathbb{1}[\text{8 cyl}]_i + \varepsilon_i
$$

This represents three groups:

- **4-cylinder** ($\mathbb{1}[6] = 0, \mathbb{1}[8] = 0$): $\hat{y} = \beta_0$
- **6-cylinder** ($\mathbb{1}[6] = 1, \mathbb{1}[8] = 0$): $\hat{y} = \beta_0 + \beta_2$
- **8-cylinder** ($\mathbb{1}[6] = 0, \mathbb{1}[8] = 1$): $\hat{y} = \beta_0 + \beta_3$

The intercept is the mean mpg for the baseline group (4-cylinder). Each
coefficient is a *difference from the baseline*, not an absolute mean.

### 3.4 Changing the reference level

The choice of baseline is arbitrary and affects the coefficients but not the
model's predictions. You can change it with `relevel()`:

```r
mtcars$cyl_factor <- relevel(mtcars$cyl_factor, ref = "8")
mod3_relevel <- lm(mpg ~ cyl_factor, data = mtcars)
display(mod3_relevel)
```

Now 8-cylinder is the baseline. The intercept becomes the mean mpg for
8-cylinder cars, and the coefficients for 4 and 6 are positive (since they
are more fuel-efficient than 8-cylinder cars).

```r
ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = cyl_factor)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  scale_fill_manual(values = c("8" = "firebrick", "4" = "steelblue", "6" = "gray60")) +
  labs(x = "Cylinders", y = "Miles per Gallon",
       title = "MPG by Cylinder Count",
       subtitle = "Each boxplot center ≈ an intercept or intercept + coefficient") +
  theme_minimal(base_size = 13)
```

---

## 4. Interactions: Non-Parallel Lines

### 4.1 The problem with parallel lines

The additive model in Section 3 assumes that weight affects mpg the same way
regardless of transmission type. But is this plausible? A light manual car
might be very efficient, while a heavy manual car might lose efficiency faster
than a heavy automatic. If so, the *slopes* should differ.

### 4.2 The interaction model

We allow the slope of weight to depend on transmission by adding an
interaction term:

$$
\text{mpg}_i = \beta_0 + \beta_1 \cdot \text{wt}_i + \beta_2 \cdot \mathbb{1}[\text{Manual}]_i + \beta_3 \cdot (\text{wt}_i \times \mathbb{1}[\text{Manual}]_i) + \varepsilon_i
$$

```r
mod2 <- lm(mpg ~ wt * am_factor, data = mtcars)
display(mod2)
```

### 4.3 Deriving the group equations

Set the indicator to 0 or 1 and simplify:

- **Automatic** ($\mathbb{1}[\text{Manual}] = 0$):

$$
\hat{y} = \beta_0 + \beta_1 \cdot \text{wt}
$$

- **Manual** ($\mathbb{1}[\text{Manual}] = 1$):

$$
\hat{y} = (\beta_0 + \beta_2) + (\beta_1 + \beta_3) \cdot \text{wt}
$$

The interaction coefficient $\beta_3$ is the *difference in slopes* between
manual and automatic. If $\beta_3 < 0$, manual cars lose mpg faster per unit
of weight than automatics. If $\beta_3 = 0$, the lines are parallel and the
additive model was sufficient.

### 4.4 The calculus perspective

For any model $E[Y \mid X] = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \cdot X_2)$, the marginal effect of $X_1$ is obtained by partial differentiation:

$$
\frac{\partial\, E[Y]}{\partial X_1} = \beta_1 + \beta_3 X_2
$$

When $X_2$ is binary, this switches between two values: $\beta_1$ (for the
baseline group) and $\beta_1 + \beta_3$ (for the other group). When $X_2$ is
continuous, the marginal effect of $X_1$ changes *continuously* as $X_2$
varies. 

### 4.5 Visualization

```r
ggplot(mtcars, aes(x = wt, y = mpg, color = am_factor)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_manual(values = c("Auto" = "steelblue", "Manual" = "firebrick")) +
  labs(title = "Model 2: Interaction Model (Non-Parallel Lines)",
       subtitle = "The slopes differ — manual cars are more sensitive to weight",
       x = "Weight (1000 lbs)", y = "Miles per Gallon", color = "Transmission") +
  theme_minimal(base_size = 13)
```

When you pass `color = am_factor` and use `geom_smooth(method = "lm")`,
`ggplot2` automatically fits separate regression lines for each group. This is
visually equivalent to the interaction model.

### 4.6 Comparing the two models side by side

```r
p_add <- ggplot(mtcars, aes(x = wt, y = mpg, color = am_factor)) +
  geom_point(size = 2, alpha = 0.6) +
  geom_abline(intercept = coef(mod1)[1], slope = coef(mod1)[2],
              color = "steelblue", linewidth = 0.7) +
  geom_abline(intercept = coef(mod1)[1] + coef(mod1)[3], slope = coef(mod1)[2],
              color = "firebrick", linewidth = 0.7) +
  scale_color_manual(values = c("Auto" = "steelblue", "Manual" = "firebrick")) +
  labs(title = "Additive (Parallel)", x = "Weight", y = "mpg") +
  theme_minimal(base_size = 11) + theme(legend.position = "none")

p_int <- ggplot(mtcars, aes(x = wt, y = mpg, color = am_factor)) +
  geom_point(size = 2, alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.7) +
  scale_color_manual(values = c("Auto" = "steelblue", "Manual" = "firebrick")) +
  labs(title = "Interaction (Non-Parallel)", x = "Weight", y = "mpg") +
  theme_minimal(base_size = 11) + theme(legend.position = "none")

grid.arrange(p_add, p_int, ncol = 2)
```

---

## 5. The Model Matrix: What R Actually Does

Understanding the design matrix helps clarify the `*` and `:` operators and
explains why coefficients are interpreted as differences.

```r
X <- model.matrix(~ wt * am_factor, data = mtcars)
head(X)
```

The output has four columns:

| Column | Contents | Role |
|---|---|---|
| `(Intercept)` | All 1s | Estimates $\beta_0$ |
| `wt` | Continuous weight values | Estimates $\beta_1$ |
| `am_factorManual` | 0 for Auto, 1 for Manual | Estimates $\beta_2$ |
| `wt:am_factorManual` | `wt` $\times$ `am_factorManual` | Estimates $\beta_3$ |

For an automatic car, the last two columns are both zero. For a manual car
with weight 2.62, column 3 is 1 and column 4 is 2.62. The interaction column
is nothing more than the element-wise product of the weight column and the
dummy column. The OLS estimate $\hat{\beta}_3$ tells you how much the slope of
weight changes when the dummy switches from 0 to 1.

The formula shorthand `wt * am_factor` expands to `wt + am_factor +
wt:am_factor`. If you only want the interaction without the main effects (which
is almost never advisable), you would write `wt:am_factor`.

---

## 6. Model Diagnostics: $F$-Statistic and Adjusted $R^2$

### 6.1 The problem with $R^2$

Recall that $R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$
measures the proportion of variance explained. There's a problem though: adding
*any* predictor to the model can only decrease $\text{SS}_{\text{res}}$ (or
leave it unchanged), so $R^2$ can only increase. You could add a column of
random noise and $R^2$ would tick upward. This makes raw $R^2$ less useful when
comparing models with different numbers of predictors.

### 6.2 Adjusted $R^2$

The adjusted $R^2$ normalizes the sums of squares by their degrees of freedom,
introducing a penalty for model complexity:

$$
R^2_{\text{adj}} = 1 - \frac{\text{SS}_{\text{res}} / (n - p - 1)}{\text{SS}_{\text{tot}} / (n - 1)} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}
$$

The factor $(n - 1) / (n - p - 1)$ is always greater than 1, so $R^2_{\text{adj}} \leq R^2$. If a new predictor does not reduce $\text{SS}_{\text{res}}$ enough to offset the loss of a degree of freedom, $R^2_{\text{adj}}$ *decreases*. This makes it a valid tool for model comparison.

```r
summary(mod1)$adj.r.squared
summary(mod2)$adj.r.squared
```

### 6.3 The ANOVA decomposition

The total variation in $Y$ decomposes into explained and unexplained parts:

$$
\underbrace{\sum_{i=1}^n (y_i - \bar{y})^2}_{\text{SS}_{\text{tot}}} = \underbrace{\sum_{i=1}^n (\hat{y}_i - \bar{y})^2}_{\text{SS}_{\text{reg}}} + \underbrace{\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{SS}_{\text{res}}}
$$

Dividing each sum of squares by its degrees of freedom gives the **mean
squares**:

$$
\text{MS}_{\text{reg}} = \frac{\text{SS}_{\text{reg}}}{p}, \qquad \text{MS}_{\text{res}} = \frac{\text{SS}_{\text{res}}}{n - p - 1}
$$

### 6.4 The $F$-statistic

The $F$-statistic is the ratio of explained variance to unexplained variance:

$$
F = \frac{\text{MS}_{\text{reg}}}{\text{MS}_{\text{res}}}
$$

Under the null hypothesis $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$
(the model has no predictive power at all), $F$ follows an
$F$-distribution with $(p,\; n - p - 1)$ degrees of freedom. A large $F$
means the regression plane fits the data substantially better than a flat
surface at $\bar{y}$.

The individual $t$-tests in `summary()` test whether *each* coefficient is
zero. The $F$-test is the *global* test: at least one coefficient is non-zero.
A significant $F$-test with no individually significant $t$-tests can occur
when predictors are correlated (a preview of multicollinearity, which we cover
next week).

### 6.5 Manual computation of the $F$-statistic

We verify R's output by computing the ANOVA decomposition from scratch:

```r
fit <- mod2  # the interaction model

y     <- mtcars$mpg
y_hat <- fitted(fit)
y_bar <- mean(y)

# Sums of squares
SS_tot <- sum((y - y_bar)^2)
SS_reg <- sum((y_hat - y_bar)^2)
SS_res <- sum((y - y_hat)^2)

# Verify the decomposition
SS_tot
SS_reg + SS_res  # should match SS_tot

# Degrees of freedom
n <- nrow(mtcars)
p <- length(coef(fit)) - 1  # number of predictors (excluding intercept)
df_reg <- p
df_res <- n - p - 1

# Mean squares
MS_reg <- SS_reg / df_reg
MS_res <- SS_res / df_res

# F-statistic
F_manual <- MS_reg / MS_res
F_manual
summary(fit)$fstatistic[1]  # compare to R's output
```

Both values should be identical. Working through this once removes the mystery
from the `summary()` output.

### 6.6 Visualizing the $F$-test

```r
f_obs <- summary(fit)$fstatistic[1]
df1   <- summary(fit)$fstatistic[2]
df2   <- summary(fit)$fstatistic[3]

f_grid <- seq(0, f_obs * 1.5, length.out = 500)
f_dens <- df(f_grid, df1 = df1, df2 = df2)
f_df   <- data.frame(f = f_grid, density = f_dens)

f_crit <- qf(0.95, df1 = df1, df2 = df2)

ggplot(f_df, aes(x = f, y = density)) +
  geom_line(linewidth = 0.7) +
  geom_area(data = subset(f_df, f >= f_crit),
            fill = "firebrick", alpha = 0.3) +
  geom_vline(xintercept = f_obs, color = "steelblue",
             linewidth = 0.8, linetype = "dashed") +
  annotate("text", x = f_obs + 1, y = max(f_dens) * 0.5,
           label = paste0("F = ", round(f_obs, 1)),
           color = "steelblue", hjust = 0, size = 4.2) +
  annotate("text", x = f_crit + 0.5, y = max(f_dens) * 0.25,
           label = "Rejection region", color = "firebrick",
           hjust = 0, size = 3.8) +
  labs(title = "F-Test: Is the Model Significant?",
       subtitle = bquote(H[0]*": all "*beta[j] == 0*"    df = ("*.(df1)*", "*.(df2)*")"),
       x = "F", y = "Density") +
  theme_minimal(base_size = 13)
```

---

## 7. Putting It All Together: A Complete Analysis

We now run a full analysis — model specification, coefficient interpretation,
comparison of additive and interaction models, and diagnostics — in a single
coherent workflow.

```r
# Reset the factor baseline
mtcars$cyl_factor <- factor(mtcars$cyl)

# Model A: additive (weight + cylinders)
modA <- lm(mpg ~ wt + cyl_factor, data = mtcars)

# Model B: interaction (weight * cylinders)
modB <- lm(mpg ~ wt * cyl_factor, data = mtcars)

display(modA)
display(modB)
```

### 7.1 Compare adjusted $R^2$

```r
cat("Model A (additive)   adj R² =", round(summary(modA)$adj.r.squared, 4), "\n")
cat("Model B (interaction) adj R² =", round(summary(modB)$adj.r.squared, 4), "\n")
```

If the interaction model's adjusted $R^2$ is not meaningfully larger, the added
complexity is not justified.

### 7.2 Visualize both models

```r
p_a <- ggplot(mtcars, aes(x = wt, y = mpg, color = cyl_factor)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.7,
              formula = y ~ x,
              data = transform(mtcars, mpg = fitted(modA))) +
  labs(title = "Additive: wt + cyl", x = "Weight", y = "mpg") +
  scale_color_manual(values = c("4" = "steelblue", "6" = "gray50", "8" = "firebrick")) +
  theme_minimal(base_size = 11) + theme(legend.position = "none")

p_b <- ggplot(mtcars, aes(x = wt, y = mpg, color = cyl_factor)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.7) +
  labs(title = "Interaction: wt * cyl", x = "Weight", y = "mpg") +
  scale_color_manual(values = c("4" = "steelblue", "6" = "gray50", "8" = "firebrick")) +
  theme_minimal(base_size = 11) + theme(legend.position = "bottom", legend.title = element_blank())

grid.arrange(p_a, p_b, ncol = 2)
```

### 7.3 Residual diagnostics

```r
mtcars$fitted_B <- fitted(modB)
mtcars$resid_B  <- residuals(modB)

p_rvf <- ggplot(mtcars, aes(x = fitted_B, y = resid_B)) +
  geom_point(size = 2, alpha = 0.7, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_smooth(se = FALSE, color = "firebrick", linewidth = 0.7, method = "loess") +
  labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs. Fitted") +
  theme_minimal(base_size = 12)

p_qq <- ggplot(data.frame(r = rstandard(modB)), aes(sample = r)) +
  stat_qq(size = 2, alpha = 0.7, color = "steelblue") +
  stat_qq_line(color = "firebrick", linewidth = 0.8) +
  labs(x = "Theoretical Quantiles", y = "Standardized Residuals",
       title = "Normal Q-Q Plot") +
  theme_minimal(base_size = 12)

grid.arrange(p_rvf, p_qq, ncol = 2)
```

---

## 8. Exercises

1. **Ceteris paribus interpretation.** Fit `lm(mpg ~ wt + hp, data = mtcars)`.
   Write out the fitted equation. A car weighs 3,000 lbs and has 150
   horsepower. What mpg does the model predict? Now consider a second car
   that weighs 3,500 lbs but also has 150 hp. What is the predicted
   difference in mpg? Verify that it equals $500 \times \hat{\beta}_{\text{wt}} / 1000$.

2. **Dummy variable mechanics.** Fit `lm(mpg ~ cyl_factor, data = mtcars)`.
   Compute the mean mpg for each cylinder group using `tapply()`. Verify
   that the intercept equals the mean of the baseline group and each
   coefficient equals the difference from the baseline.

3. **Reference level.** Using the model from Exercise 2, re-fit with 6
   cylinders as the baseline (`relevel(cyl_factor, ref = "6")`). Do the
   predicted values change? Do the coefficients change? Explain why one
   changes and the other does not.

4. **Interaction interpretation.** Using `mod2` (the `wt * am_factor`
   interaction model), compute the predicted mpg for: (a) a 2,500 lb
   automatic, (b) a 2,500 lb manual, (c) a 4,000 lb automatic, (d) a 4,000
   lb manual. Write out the arithmetic for each. Which group is more
   sensitive to weight? How does the interaction coefficient tell you this
   directly?

5. **Model comparison.** Fit three models predicting `mpg`:
   (a) `wt` only,
   (b) `wt + cyl_factor`,
   (c) `wt * cyl_factor`.
   Report the $R^2$ and adjusted $R^2$ for each. Does raw $R^2$ always
   increase? Does adjusted $R^2$? Which model do you prefer and why?

6. **Manual $F$-test.** For model (c) in Exercise 5, compute the ANOVA
   decomposition by hand (SS\_tot, SS\_reg, SS\_res, degrees of freedom,
   mean squares, $F$-statistic). Verify that your $F$-statistic matches the
   one reported by `summary()`. What is the $p$-value? Is the model as a
   whole significant?

7. **Diagnostics.** Produce residuals-vs-fitted and Q-Q plots for each of the
   three models in Exercise 5. Which model has the best-behaved residuals?
   Are there any observations that consistently appear as outliers across
   models? (Identify them by row name using `which.max(abs(residuals(...)))`.)

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 3–4.
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*.
  3rd ed. Sage. Ch. 5–6.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer.
