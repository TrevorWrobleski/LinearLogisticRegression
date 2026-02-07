# Workshop 3: Simple Linear Regression — Estimation, Residuals, and Assumptions

**ST211 — Linear and Logistic Regression**
Supplemental Material for the workshop

---

## 1. Setup and Data

We work with a dataset of student study habits and grades. Each row is a
student; the variables include `Grade` (exam score), `Study.Skills` (a
self-reported numeric score), and `Interesting` (a binary indicator: 1 if the
student finds the subject interesting, 0 otherwise).

```r
library(arm)
library(ggplot2)
library(gridExtra)

Study_habits <- read.csv("Stats_study_habits_slr.csv", header = TRUE)
head(Study_habits)
summary(Study_habits)
```

As we discussed in our last workshop, always inspect before modeling. 
`summary()` gives you the range and central
tendency of each variable, flags missing values, and lets you catch obvious
problems (e.g., a grade of 999 or a negative study skills score).

Even though it seems basic, it's important to know the range and tendency of your data. 
Imagine if you are trying to estimate the range of airplanes based on their wingspan. 
If your data includes variables for range and wingspan but only includes 30 planes like a Cessna 172, Piper Cherokee, Cirrisu SR22, etc.
with spingspans of about 36 ft, let's say ranging from 32-39 ft, is it reasonable to make estimates about 
commercial twin-engines jets liek the CRJ-900 or a 787 with wingspans of say, 148 ft? 

Probabaly not.

---

## 2. Regression with a Binary Predictor

### 2.1 Visual inspection

Before fitting anything, compare the two groups visually:

```r
ggplot(Study_habits, aes(x = factor(Interesting), y = Grade, fill = factor(Interesting))) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "firebrick")) +
  labs(x = "Finds Subject Interesting", y = "Grade",
       title = "Grades by Interest in the Subject") +
  theme_minimal(base_size = 13)
```

The boxplot tells you whether the medians differ, whether the spreads are
comparable, and whether there are outliers. It is the visual analog of the
two-sample comparison you studied in Workshop 2.

### 2.2 Fitting the model

```r
grade.interesting.lm <- lm(Grade ~ Interesting, data = Study_habits)
display(grade.interesting.lm)
summary(grade.interesting.lm)
```

The estimated model is:

$$
\widehat{\text{Grade}}_i = \hat{\beta}_0 + \hat{\beta}_1 \cdot \text{Interesting}_i
$$

**Interpreting each coefficient:**

- $\hat{\beta}_0$ (the intercept): the estimated mean grade for students with
  `Interesting = 0` — those who do *not* find the subject interesting. This is
  the **baseline group**.
- $\hat{\beta}_1$ (the slope): the estimated *difference* in mean grade between
  students who find the subject interesting (`Interesting = 1`) and those who
  do not. If $\hat{\beta}_1 > 0$, interested students score higher on average.

When the predictor is binary, the regression slope is algebraically identical to
the difference in group means: $\hat{\beta}_1 = \bar{y}_1 - \bar{y}_0$. You
can verify this:

```r
# Verify: slope = difference in means
mean(Study_habits$Grade[Study_habits$Interesting == 1]) -
  mean(Study_habits$Grade[Study_habits$Interesting == 0])
```

This is why regression with a single binary predictor is equivalent to a
two-sample $t$-test.

---

## 3. Regression with a Continuous Predictor

### 3.1 Scatterplot

```r
ggplot(Study_habits, aes(x = Study.Skills, y = Grade)) +
  geom_point(size = 2, alpha = 0.7, color = "steelblue") +
  labs(x = "Study Skills Score", y = "Grade",
       title = "Grade vs. Study Skills") +
  theme_minimal(base_size = 13)
```

Look for three things: (1) direction — is the relationship positive or
negative? (2) form — does it look approximately linear, or is there curvature?
(3) strength — how tightly do the points cluster around an imaginary line?

### 3.2 Fitting the model

The simple linear regression model is:

$$
\text{Grade}_i = \beta_0 + \beta_1 \cdot \text{Study.Skills}_i + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)
$$

We estimate $\beta_0$ and $\beta_1$ by **ordinary least squares (OLS)** — the
method that minimizes the sum of squared residuals:

$$
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} \left( y_i - \beta_0 - \beta_1 x_i \right)^2
$$

```r
grade.lm <- lm(Grade ~ Study.Skills, data = Study_habits)
display(grade.lm)
```

Suppose the output is:

```
lm(formula = Grade ~ Study.Skills, data = Study_habits)
            coef.est coef.se
(Intercept) 30.77     4.21
Study.Skills 1.49     0.15
---
n = 133, k = 2
residual sd = 15.34, R-Squared = 0.20
```

The fitted equation is:

$$
\widehat{\text{Grade}}_i = 30.77 + 1.49 \cdot \text{Study.Skills}_i
$$

### 3.3 Interpreting every number in the output

**The intercept ($\hat{\beta}_0 = 30.77$).** The predicted grade for a student
with `Study.Skills = 0`. Whether this is meaningful depends on whether
`Study.Skills = 0` is within the range of the data. If the minimum observed
study skills score is, say, 10, then the intercept is an extrapolation — it
anchors the line mathematically but does not describe any real student.

**The slope ($\hat{\beta}_1 = 1.49$).** For every one-unit increase in study
skills, the model predicts an increase of 1.49 points in grade, *on average
and holding everything else constant*. The "holding everything else constant"
clause is trivial here (we have one predictor), but it will matter
in multiple regression.

**Standard errors (4.21 and 0.15).** These quantify the uncertainty in the
coefficient estimates. The ratio $\hat{\beta}/\text{SE}(\hat{\beta})$ is the
$t$-statistic. For the slope: $1.49 / 0.15 \approx 9.93$. The quick rule of
thumb: if the absolute value of the coefficient is more than twice its standard
error, it is statistically significant at approximately the 5% level.

**Residual standard deviation ($\hat{\sigma} = 15.34$).** This estimates the
standard deviation of the errors $\varepsilon_i$. It tells you how much
individual grades deviate from the regression line, in the original units of $y$
(points). Smaller is better. Compare it to the range and standard deviation of
`Grade` to judge whether the model is usefully precise.

**$R^2 = 0.20$.** The proportion of the total variance in `Grade` that is
explained by the linear relationship with `Study.Skills`. Formally:

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

An $R^2$ of 0.20 means 20% of the variation in grades is accounted for by study
skills; the remaining 80% is due to other factors (intelligence, sleep, luck,
measurement error, etc.). $R^2$ ranges from 0 (the model explains nothing) to 1
(perfect fit).

### 3.4 Visualizing the fit

```r
ggplot(Study_habits, aes(x = Study.Skills, y = Grade)) +
  geom_point(size = 2, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "firebrick",
              fill = "firebrick", alpha = 0.15, linewidth = 0.9) +
  labs(x = "Study Skills Score", y = "Grade",
       title = "Grade ~ Study Skills: Fitted Regression Line",
       subtitle = "Shaded band = 95% confidence interval for the mean response") +
  theme_minimal(base_size = 13)
```

The gray band is the **confidence interval for the conditional mean** — it
tells you where the *average* grade lies for a given study skills value, not
where an *individual* student's grade lies. The latter would require a
**prediction interval**, which is wider.

---

## 4. What Are Residuals?

The **residual** for observation $i$ is the difference between what we observed
and what the model predicted:

$$
e_i = y_i - \hat{y}_i = y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)
$$

Residuals are the model's mistakes. They are the empirical stand-ins for the
true (unobservable) errors $\varepsilon_i$.

```r
Study_habits$fitted <- fitted(grade.lm)
Study_habits$resid  <- residuals(grade.lm)
```

### 4.1 Visualizing residuals

```r
ggplot(Study_habits, aes(x = Study.Skills, y = Grade)) +
  geom_segment(aes(xend = Study.Skills, yend = fitted),
               color = "firebrick", alpha = 0.5, linewidth = 0.7) +
  geom_point(size = 2.5, color = "steelblue") +
  geom_line(aes(y = fitted), color = "firebrick", linewidth = 0.9) +
  labs(x = "Study Skills Score", y = "Grade",
       title = "Residuals Are Vertical Distances from the Line",
       subtitle = "OLS minimizes the sum of the squared red segments") +
  theme_minimal(base_size = 13)
```

Each red segment is one residual. OLS chooses the line that minimizes the sum
of the *squares* of these segments. We square for three reasons: (1) positive
and negative residuals would otherwise cancel; (2) squaring penalizes large
errors more heavily than small ones; (3) the squared loss is differentiable
everywhere, which makes the calculus clean.

---

## 5. The Four Assumptions and Why They Matter

The model $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ comes with assumptions
about $\varepsilon_i$.

### The assumptions

| # | Assumption | Formal statement | What it means |
|---|---|---|---|
| 1 | **Linearity** | $E[\varepsilon_i \mid x_i] = 0$ | The true relationship is linear in $x$ |
| 2 | **Independence** | $\varepsilon_i \perp \varepsilon_j$ for $i \neq j$ | One observation's error tells you nothing about another's |
| 3 | **Constant variance** | $\text{Var}(\varepsilon_i) = \sigma^2$ for all $i$ | The spread of the errors does not depend on $x$ |
| 4 | **Normality** | $\varepsilon_i \sim N(0, \sigma^2)$ | The errors follow a Normal distribution |

### Which assumptions matter for what?


**For estimation (OLS point estimates).** OLS gives you the best linear
unbiased estimates of $\beta_0$ and $\beta_1$ under assumptions 1–3 alone (the
Gauss-Markov theorem). You do **not** need normality to get good estimates.

**For inference (standard errors, $t$-tests, confidence intervals, $p$-values).**
You need all four assumptions. The $t$-statistics and $p$-values reported by
`summary()` are derived under the assumption that the errors are Normal. If
normality fails but the sample is large, the Central Limit Theorem makes the
inference approximately valid anyway. If the sample is small and the errors are
heavily skewed or have thick tails, the reported $p$-values may be unreliable.

**For prediction.** The accuracy of point predictions depends mainly on
assumptions 1 and 3. The accuracy of *prediction intervals* depends on all
four.

Linearity and constant variance are always important. 
Normality matters primarily for small-sample inference.

---

## 6. Diagnostic Plots

We check the assumptions by examining residuals. There are three standard
diagnostic plots.

### 6.1 Residuals vs. Fitted Values

It checks linearity and constant variance.

```r
p_rvf <- ggplot(Study_habits, aes(x = fitted, y = resid)) +
  geom_point(size = 2, alpha = 0.7, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "firebrick") +
  geom_smooth(se = FALSE, color = "firebrick", linewidth = 0.7, method = "loess") +
  labs(x = "Fitted Values", y = "Residuals",
       title = "Residuals vs. Fitted Values",
       subtitle = "Look for patterns: curvature (non-linearity) or funnel (non-constant variance)") +
  theme_minimal(base_size = 13)
p_rvf
```

**What you want to see:** a random horizontal cloud centered at zero, with
roughly constant vertical spread. The loess smoother (red curve) should hug
the dashed line.

**What violations look like:** curvature in the smoother indicates
non-linearity; a funnel shape (residuals spreading out or narrowing as fitted
values increase) indicates heteroscedasticity.

### 6.2 Normal Q-Q Plot

```r
p_qq <- ggplot(Study_habits, aes(sample = rstandard(grade.lm))) +
  stat_qq(size = 2, alpha = 0.7, color = "steelblue") +
  stat_qq_line(color = "firebrick", linewidth = 0.8) +
  labs(x = "Theoretical Quantiles", y = "Standardized Residuals",
       title = "Normal Q-Q Plot",
       subtitle = "Points on the line = normality; S-curves or heavy tails = non-normality") +
  theme_minimal(base_size = 13)
p_qq
```

### 6.3 Histogram of Standardized Residuals

```r
p_hist <- ggplot(data.frame(r = rstandard(grade.lm)), aes(x = r)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 15, fill = "steelblue", color = "white", alpha = 0.7) +
  stat_function(fun = dnorm, color = "firebrick", linewidth = 0.8) +
  labs(x = "Standardized Residuals", y = "Density",
       title = "Histogram of Standardized Residuals",
       subtitle = "Overlay: standard Normal density") +
  theme_minimal(base_size = 13)
p_hist
```

### 6.4 All three together

```r
grid.arrange(p_rvf, p_qq, p_hist, layout_matrix = rbind(c(1, 1), c(2, 3)))
```

### 6.5 Arranging multiple base plots (for reference)

You will sometimes see diagnostic plots produced with base R's `plot()`:

```r
par(mfrow = c(2, 2))
plot(grade.lm, which = c(1, 2))
hist(rstandard(grade.lm), main = "Standardized Residuals",
     xlab = "Standardized Residuals", freq = FALSE)
dev.off()
```

`par(mfrow = c(2, 2))` sets up a 2×2 grid. `dev.off()` resets the graphics
device to its default (one plot per panel). 

---

## 7. What Bad Residuals Look Like

Theory is easier to internalize when you can see what violations look like in
practice. We use datasets built into R so that no downloads are required.

### 7.1 Non-linearity

The relationship between engine displacement and fuel efficiency in `mtcars`
is not linear — it flattens out at high displacement.

```r
data(mtcars)
m_nonlin <- lm(mpg ~ disp, data = mtcars)

mtcars$fitted_nl  <- fitted(m_nonlin)
mtcars$resid_nl   <- residuals(m_nonlin)

p_data_nl <- ggplot(mtcars, aes(x = disp, y = mpg)) +
  geom_point(size = 2.5, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick", linewidth = 0.8) +
  labs(x = "Displacement (cu. in.)", y = "Miles per Gallon",
       title = "mpg ~ displacement: Note the Curvature") +
  theme_minimal(base_size = 13)

p_resid_nl <- ggplot(mtcars, aes(x = fitted_nl, y = resid_nl)) +
  geom_point(size = 2.5, alpha = 0.7, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_smooth(se = FALSE, color = "firebrick", linewidth = 0.7, method = "loess") +
  labs(x = "Fitted Values", y = "Residuals",
       title = "Residuals vs. Fitted: Clear Curvature",
       subtitle = "The loess curve bends — linearity is violated") +
  theme_minimal(base_size = 13)

grid.arrange(p_data_nl, p_resid_nl, ncol = 2)
```

The residuals show a U-shaped pattern: the model over-predicts in the middle
of the range and under-predicts at both ends. This is the signature of a
non-linear relationship being forced through a straight line.

### 7.2 Heteroscedasticity (non-constant variance)

Population and area data from `state.x77` exhibit classic heteroscedasticity —
larger states have more variable populations.

```r
state_df <- data.frame(
  area = state.x77[, "Area"],
  pop  = state.x77[, "Population"]
)
m_hetero <- lm(pop ~ area, data = state_df)

state_df$fitted_h <- fitted(m_hetero)
state_df$resid_h  <- residuals(m_hetero)

p_data_h <- ggplot(state_df, aes(x = area, y = pop)) +
  geom_point(size = 2.5, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick", linewidth = 0.8) +
  labs(x = "Land Area (sq. miles)", y = "Population (thousands)",
       title = "Population ~ Area: Spread Increases with Area") +
  theme_minimal(base_size = 13)

p_resid_h <- ggplot(state_df, aes(x = fitted_h, y = resid_h)) +
  geom_point(size = 2.5, alpha = 0.7, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_smooth(se = FALSE, color = "firebrick", linewidth = 0.7, method = "loess") +
  labs(x = "Fitted Values", y = "Residuals",
       title = "Residuals vs. Fitted: Funnel Shape",
       subtitle = "Variance increases with fitted values — heteroscedasticity") +
  theme_minimal(base_size = 13)

grid.arrange(p_data_h, p_resid_h, ncol = 2)
```

The funnel shape (residuals fanning out to the right) tells you the constant-
variance assumption is violated. The OLS estimates are still unbiased, but the
standard errors — and therefore the $p$-values and confidence intervals — are
wrong. This is a case where estimation is fine but inference is compromised.

### 7.3 Non-normality (heavy tails)

We simulate data from a $t$-distribution with 3 degrees of freedom to produce
heavy-tailed errors, then compare the Q-Q plot to a well-behaved case.

```r
set.seed(211)
n <- 100
x_sim <- runif(n, 0, 10)

# Good residuals: Normal errors
y_good <- 2 + 3 * x_sim + rnorm(n, 0, 2)
m_good <- lm(y_good ~ x_sim)

# Bad residuals: heavy-tailed errors (t with 3 df, scaled)
y_heavy <- 2 + 3 * x_sim + rt(n, df = 3) * 2
m_heavy <- lm(y_heavy ~ x_sim)

qq_good <- ggplot(data.frame(r = rstandard(m_good)), aes(sample = r)) +
  stat_qq(color = "steelblue", size = 2, alpha = 0.7) +
  stat_qq_line(color = "firebrick", linewidth = 0.8) +
  labs(title = "Q-Q Plot: Normal Errors",
       subtitle = "Points follow the line closely",
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal(base_size = 12)

qq_heavy <- ggplot(data.frame(r = rstandard(m_heavy)), aes(sample = r)) +
  stat_qq(color = "steelblue", size = 2, alpha = 0.7) +
  stat_qq_line(color = "firebrick", linewidth = 0.8) +
  labs(title = "Q-Q Plot: Heavy-Tailed Errors",
       subtitle = "Tails deviate sharply from the line",
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal(base_size = 12)

grid.arrange(qq_good, qq_heavy, ncol = 2)
```

In the left panel, the points track the line — normality holds. In the right
panel, the tails curve away from the line (an S-shape). This tells you the
error distribution has heavier tails than the Normal. The coefficient estimates
are still unbiased, but if $n$ is small, the $p$-values from `summary()` may
be misleading.

### 7.4 Well-behaved residuals

For contrast, here is `mtcars` regressing `mpg` on `wt` — a case where the
assumptions are reasonably satisfied.

```r
m_ok <- lm(mpg ~ wt, data = mtcars)
mtcars$fitted_ok <- fitted(m_ok)
mtcars$resid_ok  <- residuals(m_ok)

p_rvf_ok <- ggplot(mtcars, aes(x = fitted_ok, y = resid_ok)) +
  geom_point(size = 2.5, alpha = 0.7, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_smooth(se = FALSE, color = "firebrick", linewidth = 0.7, method = "loess") +
  labs(x = "Fitted Values", y = "Residuals",
       title = "Residuals vs. Fitted: Reasonably Well-Behaved") +
  theme_minimal(base_size = 12)

p_qq_ok <- ggplot(data.frame(r = rstandard(m_ok)), aes(sample = r)) +
  stat_qq(color = "steelblue", size = 2, alpha = 0.7) +
  stat_qq_line(color = "firebrick", linewidth = 0.8) +
  labs(title = "Q-Q Plot: Approximately Normal",
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal(base_size = 12)

grid.arrange(p_rvf_ok, p_qq_ok, ncol = 2)
```

No strong curvature, no funnel, no dramatic departures from the Q-Q line.
In practice, real data will never be perfectly clean — the question is whether
the violations are severe enough to undermine your conclusions.

---

## 8. Making Predictions

Once you have a fitted model, we can make predictions based on our estimates:

$$
\widehat{\text{Grade}} = \hat{\beta}_0 + \hat{\beta}_1 \cdot \text{Study.Skills}
$$

For a student with `Study.Skills = 25`:

```r
new_student <- data.frame(Study.Skills = 25)
predict(grade.lm, newdata = new_student)

# Manual check:
coef(grade.lm)[1] + coef(grade.lm)[2] * 25
```

Both should return the same number. It's good to verify programmatic predictions
with the manual calculation at least once.

---

## 9. Summary: Reading `display()` Output

When you see output from `display()` or `summary()`, here is exactly what to
report:

| Output field | Symbol | Interpretation |
|---|---|---|
| `(Intercept) coef.est` | $\hat{\beta}_0$ | Predicted $y$ when all predictors are 0 |
| `x coef.est` | $\hat{\beta}_1$ | Change in predicted $y$ per 1-unit increase in $x$ |
| `coef.se` | $\text{SE}(\hat{\beta})$ | Uncertainty in the estimate; $\lvert\hat{\beta}\rvert > 2 \cdot \text{SE}$ suggests significance |
| `residual sd` | $\hat{\sigma}$ | Typical size of prediction errors (in units of $y$) |
| `R-Squared` | $R^2$ | Proportion of variance in $y$ explained by the model |
| `n` | $n$ | Number of observations |
| `k` | $k$ | Number of estimated parameters (intercept + slopes) |

---

## 10. Exercises

1. **Binary predictor.** Fit `Grade ~ Interesting`. Write out the fitted
   equation. Compute the mean grade separately for each group using `tapply()`
   or `dplyr`. Verify that the intercept equals the mean of the baseline group
   and the slope equals the difference in means.

2. **Interpretation drill.** Using `grade.lm` (Grade ~ Study.Skills), answer:
   (a) What grade does the model predict for a student with `Study.Skills = 0`?
   Is this prediction meaningful?
   (b) Two students differ by 5 points in study skills. What is the predicted
   difference in their grades?
   (c) If the residual standard deviation is 8.72, roughly what range contains
   about 95% of individual prediction errors? (*Hint:* $\pm 2\hat{\sigma}$.)

3. **Diagnostic practice.** Fit `lm(mpg ~ hp, data = mtcars)`. Produce the
   three diagnostic plots (residuals vs. fitted, Q-Q, histogram). Which
   assumptions appear satisfied? Which appear questionable? Be specific.

4. **Non-linearity fix.** Using `mtcars`, fit `lm(mpg ~ disp)` and confirm
   the non-linear pattern from Section 7.1. Now fit `lm(mpg ~ log(disp))`.
   Produce the residuals-vs-fitted plot for the log model. Does the curvature
   improve? What does this tell you about transformations?

5. **Simulation.** Generate 150 observations from the model
   $y = 5 + 2x + \varepsilon$ where $x \sim \text{Uniform}(0, 10)$ and
   $\varepsilon \sim N(0, 3)$. Fit the regression `lm(y ~ x)`. Compare the
   estimated coefficients to the true values. Produce all three diagnostic
   plots — they should look clean. Then repeat with $\varepsilon \sim t_3$
   (scaled by 3, using `rt(n, df = 3) * 3`) and observe the difference.

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 3.
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*.
  3rd ed. Sage. Ch. 6 (Diagnostics).
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer.
