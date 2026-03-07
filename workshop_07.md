# Workshop 7: Transforms — Centering, Standardizing, and Log Models

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop
---

## 1. Why do we need transformations? 


| Problem | Tool | What it does |
|---|---|---|
| Uninterpretable intercept | **Centering** | Shifts the zero-point of each predictor to its mean |
| Incomparable coefficients | **Standardizing** | Puts all predictors on a common scale (standard deviations) |
| Violated assumptions / non-linearity | **Log and power transforms** | Converts multiplicative relationships to additive ones |

---

## 2. Setup

```r
library(ggplot2)
library(arm)
library(gridExtra)
```

We define two utility functions that we will use throughout:

```r
center <- function(v) v - mean(v, na.rm = TRUE)

standardize <- function(v) (v - mean(v, na.rm = TRUE)) / sd(v, na.rm = TRUE)
```

---

## 3. Centering

### 3.1 What centering does, algebraically

Given a predictor $x$, define the centered version:

$$
\tilde{x}_i = x_i - \bar{x}
$$

Substitute into the regression $y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i$:

$$
y_i = \beta_0 + \beta_1(\tilde{x}_{1i} + \bar{x}_1) + \beta_2(\tilde{x}_{2i} + \bar{x}_2) + \varepsilon_i
$$

$$
= \underbrace{(\beta_0 + \beta_1 \bar{x}_1 + \beta_2 \bar{x}_2)}_{\beta_0^*} + \beta_1 \tilde{x}_{1i} + \beta_2 \tilde{x}_{2i} + \varepsilon_i
$$

The new intercept is:

$$
\beta_0^* = \hat{y}\big|_{x_1 = \bar{x}_1,\; x_2 = \bar{x}_2}
$$

That is: the predicted outcome for an observation at the **mean of every
predictor**. The slopes $\beta_1$ and $\beta_2$, $R^2$, and $p$-values do not change.

### 3.2 Worked example

Consider a model of systolic blood pressure:

$$
\widehat{\text{BP}} = 100.5 - 0.08 \cdot \text{Dose} + 0.45 \cdot \text{Age}
$$

The intercept (100.5) is the predicted BP when Dose = 0 and Age = 0 —
meaningless. Suppose the sample means are $\overline{\text{Dose}} = 100$ mg
and $\overline{\text{Age}} = 50$ years. The centered intercept is:

$$
\beta_0^* = 100.5 - 0.08(100) + 0.45(50) = 100.5 - 8 + 22.5 = 115.0
$$

Interpretation: for a patient taking the average dose (100 mg) at the average
age (50 years), the predicted systolic BP is 115 mmHg. This is a clinically
meaningful statement. The slopes remain $-0.08$ and $+0.45$ — shifting the
ruler does not change how fast things move.

### 3.3 Centering in R with `mtcars`

We illustrate centering with our well loved mtcars dataset to
predict fuel efficiency from weight and horsepower.

```r
data(mtcars)

# Raw model
raw.lm <- lm(mpg ~ wt + hp, data = mtcars)
display(raw.lm)
```

The intercept is the predicted mpg for a car with zero weight and zero
horsepower — hard to find. Now if we center:

```r
mtcars$wt_c <- center(mtcars$wt)
mtcars$hp_c <- center(mtcars$hp)

cent.lm <- lm(mpg ~ wt_c + hp_c, data = mtcars)
display(cent.lm)
```

we can compare the two models:

```r
cat("Raw intercept:     ", round(coef(raw.lm)[1], 2), "\n")
cat("Centered intercept:", round(coef(cent.lm)[1], 2), "\n")
cat("Mean mpg:          ", round(mean(mtcars$mpg), 2), "\n")

cat("\nRaw slope (wt):     ", round(coef(raw.lm)[2], 4), "\n")
cat("Centered slope (wt):", round(coef(cent.lm)[2], 4), "\n")
```

The centered intercept equals the mean of `mpg` (since we have no
interactions and the centered predictors have mean zero). The slopes are
identical. Nothing about the model's predictive ability changed — we simply
moved the reference point to somewhere meaningful.

### 3.4 When to center

Center continuous predictors whenever zero is not a meaningful value in the
data. 

Centering is unnecessary when zero has a natural
interpretation: number of cigarettes smoked per day (zero = non-smoker),
number of prior hospitalizations (zero = never hospitalized), or an indicator
variable (0 or 1).

---

## 4. Standardizing

### 4.1 The problem with raw coefficients

If we return to the blood pressure model:

$$
\widehat{\text{BP}} = 100.5 - 0.08 \cdot \text{Dose} + 0.12 \cdot \text{Weight}
$$

Weight has a larger coefficient (0.12 vs 0.08). Does weight matter more? No.
Dose ranges from 10 to 200 mg — the full range produces a change of
$0.08 \times 190 = 15.2$ mmHg. Weight ranges from 55 to 120 kg — the full
range produces $0.12 \times 65 = 7.8$ mmHg. Dose has roughly double the
practical impact, despite the smaller coefficient.

Raw coefficients cannot be compared across predictors because they are on
different scales. A one-milligram change in dose and a one-kilogram change in
weight are not comparable quantities.

### 4.2 The fix: divide by the standard deviation

Standardizing extends centering by also dividing by the standard deviation:

$$
z_i = \frac{x_i - \bar{x}}{s_x}
$$

Now $z$ is unitless, with mean 0 and standard deviation 1. A one-unit change
in $z$ corresponds to a **one-standard-deviation change** in the original
predictor. The relationship between raw and standardized coefficients is:

$$
\beta_j^{\text{std}} = \beta_j^{\text{raw}} \times s_{x_j}
$$

Suppose $s_{\text{Dose}} = 55$ mg and $s_{\text{Weight}} = 18$ kg:

$$
\beta_{\text{Dose}}^{\text{std}} = (-0.08) \times 55 = -4.4
$$

$$
\beta_{\text{Weight}}^{\text{std}} = (+0.12) \times 18 = +2.16
$$

Now we can compare: a one-SD increase in dose is associated with a 4.4 mmHg
decrease in BP, while a one-SD increase in weight is associated with a 2.16
mmHg increase. Dose has roughly twice the impact. The raw coefficients were
misleading because dose has far more variability than weight.

### 4.3 Standardizing in R with `mtcars`

```r
mtcars$wt_z <- standardize(mtcars$wt)
mtcars$hp_z <- standardize(mtcars$hp)

std.lm <- lm(mpg ~ wt_z + hp_z, data = mtcars)
display(std.lm)
```

```r
# Verify the relationship: std coef = raw coef * sd(x)
cat("Raw wt coef * sd(wt):  ", round(coef(raw.lm)["wt"] * sd(mtcars$wt), 4), "\n")
cat("Standardized wt coef:  ", round(coef(std.lm)["wt_z"], 4), "\n")

cat("\nRaw hp coef * sd(hp):  ", round(coef(raw.lm)["hp"] * sd(mtcars$hp), 4), "\n")
cat("Standardized hp coef:  ", round(coef(std.lm)["hp_z"], 4), "\n")
```

Which predictor has the larger standardized coefficient? That is the stronger
predictor, measured in the only fair currency: standard deviations.

### 4.4 What does not change

Standardizing the predictors does **not** change $R^2$, the residual standard
deviation, $p$-values, or the $F$-statistic. 

### 4.5 Do not standardize categorical predictors

A one-standard-deviation change in a 0/1 variable has no real-world
interpretation. Gender does not come in fractions of a standard deviation.
Leave dummy variables on their natural (0/1) scale. Standardize only
continuous predictors.

### 4.6 Comparing centering and standardizing side by side

```r
comparison <- data.frame(
  Quantity = c("Intercept", "Slope (wt)", "Slope (hp)", "R-squared"),
  Raw      = c(round(coef(raw.lm)[1], 2), round(coef(raw.lm)[2], 4),
               round(coef(raw.lm)[3], 4), round(summary(raw.lm)$r.squared, 4)),
  Centered = c(round(coef(cent.lm)[1], 2), round(coef(cent.lm)[2], 4),
               round(coef(cent.lm)[3], 4), round(summary(cent.lm)$r.squared, 4)),
  Standardized = c(round(coef(std.lm)[1], 2), round(coef(std.lm)[2], 4),
                    round(coef(std.lm)[3], 4), round(summary(std.lm)$r.squared, 4))
)
print(comparison, row.names = FALSE)
```

The intercept changes with centering and standardizing. The slopes change only
with standardizing. $R^2$ is identical across all three.

---

## 5. Log Transforms

Log transforms address a different problem than centering or standardizing by 
changing the *structure* of the model, not just the scale.

### 5.1 Additive vs. multiplicative effects

Consider two people getting a raise:

- Alice earns \$20,000 and gets a \$2,000 raise.
- Bob earns \$100,000 and gets a \$2,000 raise.

In absolute terms, they received the same raise. In relative terms, Alice got
a 10% raise and Bob got a 2% raise. Most economic and biological processes
are multiplicative: a 10% raise, a 5% infection rate, a 3% annual growth
rate. Standard linear regression models additive effects. If the true process
is multiplicative, the model is wrong.

The log transform converts multiplicative processes into additive ones,
because $\log(a \times b) = \log(a) + \log(b)$. 

I strongly recommend you review your log laws - they will be very helpful
as you progress in statistics... and they'll make calculations much easier.

### 5.2 Seeing the problem: heteroscedasticity in earnings data

We use `mtcars` to demonstrate the pattern. The relationship between weight
and the *dollar value* of a car (approximated from its characteristics)
exhibits the classic right-skew and heteroscedasticity pattern that earnings
data shows.

We construct a synthetic but realistic example:

```r
set.seed(211)
n <- 500
age <- round(runif(n, 22, 65))
experience <- age - 22
education <- sample(c("HS", "BA", "MA"), n, replace = TRUE, prob = c(0.4, 0.4, 0.2))

# Multiplicative DGP: earnings depend on experience and education
log_earn <- 9.5 + 0.03 * experience + 
            ifelse(education == "BA", 0.3, ifelse(education == "MA", 0.55, 0)) +
            rnorm(n, 0, 0.4)
earnings <- exp(log_earn)

earn_df <- data.frame(earnings, age, experience, education = factor(education))
```

Fit a linear model on the raw (untransformed) earnings:

```r
earn_raw.lm <- lm(earnings ~ experience + education, data = earn_df)
```

```r
earn_df$fitted_raw <- fitted(earn_raw.lm)
earn_df$resid_raw  <- residuals(earn_raw.lm)

p_rvf_raw <- ggplot(earn_df, aes(x = fitted_raw, y = resid_raw)) +
  geom_point(size = 1, alpha = 0.4, colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_smooth(se = FALSE, colour = "firebrick", linewidth = 0.7, method = "loess") +
  labs(title = "Raw Model: Residuals vs. Fitted",
       subtitle = "Classic funnel — variance increases with fitted values",
       x = "Fitted Values ($)", y = "Residuals ($)") +
  theme_minimal(base_size = 12)

p_qq_raw <- ggplot(data.frame(r = rstandard(earn_raw.lm)), aes(sample = r)) +
  stat_qq(size = 1, alpha = 0.4, colour = "steelblue") +
  stat_qq_line(colour = "firebrick", linewidth = 0.8) +
  labs(title = "Raw Model: Q-Q Plot",
       subtitle = "Bowed shape — right-skewed residuals",
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal(base_size = 12)

grid.arrange(p_rvf_raw, p_qq_raw, ncol = 2)
```

The funnel in the residuals-vs-fitted plot and the bowed Q-Q plot have the
same root cause: earnings are strictly positive, right-skewed, and generated
by a multiplicative process. High earners have large absolute residuals; low
earners have small ones. The constant-variance assumption is violated.

### 5.3 The fix: log the outcome

If the true data-generating process is multiplicative:

$$
y_i = e^{\beta_0 + \beta_1 x_i + \varepsilon_i}
$$

then taking the natural log of both sides:

$$
\log(y_i) = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

This is an ordinary linear regression with $\log(y)$ as the outcome. All our
familiar tools — OLS, residual plots, $t$-tests — work exactly as before.

```r
earn_log.lm <- lm(log(earnings) ~ experience + education, data = earn_df)
display(earn_log.lm)
```

```r
earn_df$fitted_log <- fitted(earn_log.lm)
earn_df$resid_log  <- residuals(earn_log.lm)

p_rvf_log <- ggplot(earn_df, aes(x = fitted_log, y = resid_log)) +
  geom_point(size = 1, alpha = 0.4, colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_smooth(se = FALSE, colour = "firebrick", linewidth = 0.7, method = "loess") +
  labs(title = "Log Model: Residuals vs. Fitted",
       subtitle = "Funnel is gone — constant variance restored",
       x = "Fitted Values (log $)", y = "Residuals (log $)") +
  theme_minimal(base_size = 12)

p_qq_log <- ggplot(data.frame(r = rstandard(earn_log.lm)), aes(sample = r)) +
  stat_qq(size = 1, alpha = 0.4, colour = "steelblue") +
  stat_qq_line(colour = "firebrick", linewidth = 0.8) +
  labs(title = "Log Model: Q-Q Plot",
       subtitle = "Points track the line — normality restored",
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal(base_size = 12)

grid.arrange(p_rvf_log, p_qq_log, ncol = 2)
```

The funnel is improved. The Q-Q plot straightens. This is because on the log
scale, multiplicative errors become additive:
$\log(y \cdot \varepsilon) = \log(y) + \log(\varepsilon)$. The error
$\log(\varepsilon)$ has constant variance regardless of the level of $y$.

### 5.4 Interpreting coefficients in a log model


Consider two observations differing by one unit in $x$:

$$
\log(y_{\text{new}}) - \log(y_{\text{old}}) = \beta_1
$$

Since $\log(a) - \log(b) = \log(a/b)$:

$$
\log\!\left(\frac{y_{\text{new}}}{y_{\text{old}}}\right) = \beta_1
$$

Exponentiate:

$$
\frac{y_{\text{new}}}{y_{\text{old}}} = e^{\beta_1}
$$

A one-unit increase in $x$ **multiplies** $y$ by $e^{\beta_1}$. The
percentage change in $y$ is $(e^{\beta_1} - 1) \times 100\%$.

**The small-$\beta$ approximation.** When $|\beta_1| < 0.20$ or so,
$e^{\beta_1} \approx 1 + \beta_1$, so $\beta_1$ is approximately the
proportional change. For example, $\beta_1 = 0.07$ means approximately a 7%
increase. For larger coefficients, always compute $e^{\beta_1}$ exactly.

### 5.5 Interpreting the simulated model

```r
display(earn_log.lm)
```

Read the output:

- **Experience coefficient** ($\approx 0.03$): each additional year of
  experience is associated with a $0.03 \times 100 = 3\%$ increase in
  earnings, holding education constant. (Exact: $e^{0.03} = 1.0305$, so
  3.05%. The approximation is excellent.)
- **Education (BA) coefficient** ($\approx 0.30$): someone with a BA earns
  $e^{0.30} = 1.350$ times what a high-school graduate earns — a 35%
  premium. Note that 0.30 is large enough that the naive approximation
  ("30%") is noticeably off. Always exponentiate for $|\beta| > 0.10$.
- **Education (MA) coefficient** ($\approx 0.55$): $e^{0.55} = 1.733$, a
  73% premium over high school.

### 5.6 Categorical predictors in a log model

For a dummy variable $D$ (0 or 1) in a log model:

$$
D = 0: \quad \log(\hat{y}) = \hat{\beta}_0 + \hat{\beta}_1 x
$$
$$
D = 1: \quad \log(\hat{y}) = (\hat{\beta}_0 + \hat{\beta}_2) + \hat{\beta}_1 x
$$

The ratio of predicted outcomes:

$$
\frac{\hat{y}_{D=1}}{\hat{y}_{D=0}} = e^{\hat{\beta}_2}
$$

The coefficient $\hat{\beta}_2$ yields a **multiplier**, not a difference.
This is more informative than an additive interpretation because it scales
naturally. Saying "a BA holder earns 35% more" is meaningful regardless of
whether the baseline is \$30,000 or \$80,000. Saying "a BA holder earns
\$15,000 more" is misleading if it applies uniformly across the income
distribution.

---

## 6. Log Transforms on Predictors: Diminishing Returns

Sometimes the non-linearity is in the predictor, not the outcome. The
relationship between engine displacement and horsepower in `mtcars` is a
shows that doubling displacement does not double power.

### 6.1 Three models, one dataset

```r
data(mtcars)

# Level-level (linear)
lm_lin    <- lm(mpg ~ wt, data = mtcars)

# Log-level (log outcome, raw predictor)
lm_loglin <- lm(log(mpg) ~ wt, data = mtcars)

# Log-log (log both)
lm_loglog <- lm(log(mpg) ~ log(wt), data = mtcars)

cat("Level-level R²:", round(summary(lm_lin)$r.squared, 3), "\n")
cat("Log-level R²:  ", round(summary(lm_loglin)$r.squared, 3), "\n")
cat("Log-log R²:    ", round(summary(lm_loglog)$r.squared, 3), "\n")
```

```r
p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Level–Level", x = "Weight (1000 lbs)", y = "MPG") +
  theme_minimal(base_size = 11)

p2 <- ggplot(mtcars, aes(x = wt, y = log(mpg))) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Log–Level", x = "Weight (1000 lbs)", y = "log(MPG)") +
  theme_minimal(base_size = 11)

p3 <- ggplot(mtcars, aes(x = log(wt), y = log(mpg))) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Log–Log", x = "log(Weight)", y = "log(MPG)") +
  theme_minimal(base_size = 11)

grid.arrange(p1, p2, p3, ncol = 3)
```

The log-log plot is the straightest. This tells you the relationship follows
a **power law**: $\text{MPG} \propto \text{wt}^{\beta}$.

### 6.2 The four model types

This table is the single most important reference for interpreting
transformed regressions. Memorize it.

| Model | Equation | Meaning of $\hat{\beta}_1$ |
|---|---|---|
| Level–level | $y = \beta_0 + \beta_1 x$ | 1 unit $\uparrow x$ → $\beta_1$ units $\Delta y$ |
| Log–level | $\log y = \beta_0 + \beta_1 x$ | 1 unit $\uparrow x$ → $\approx \beta_1 \times 100\%$ $\Delta y$ |
| Level–log | $y = \beta_0 + \beta_1 \log x$ | 1% $\uparrow x$ → $\beta_1 / 100$ units $\Delta y$ |
| Log–log | $\log y = \beta_0 + \beta_1 \log x$ | 1% $\uparrow x$ → $\beta_1$% $\Delta y$ (elasticity) |

The log-level interpretation uses the small-$\beta$ approximation. The
log-log coefficient is an **elasticity** — one of the most important
quantities in economics. If you model $\log(\text{quantity demanded})$ as a
function of $\log(\text{price})$, the slope is the price elasticity of
demand.

### 6.3 Verifying with `mtcars`

```r
display(lm_loglog)
```

If the log-log slope is approximately $-1.2$, this means: a 1% increase in
weight is associated with a 1.2% decrease in fuel efficiency. Doubling
weight ($+100\%$) would be associated with roughly a $1.2 \times 100 = 120\%$
decrease — though at that scale the linear approximation to the log breaks
down and you should compute the exact prediction.

---

## 7. Quadratic Terms: An Alternative to the Log

When a relationship shows curvature — earnings rising steeply early in a
career and flattening later — a quadratic term is another option:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon
$$

The marginal effect of $x$ is now $\beta_1 + 2\beta_2 x$ — it depends on
where you are on the $x$-axis. If $\beta_2 < 0$, the curve peaks at
$x^* = -\beta_1 / (2\beta_2)$ and declines thereafter.

```r
# Quadratic model for mpg ~ wt
lm_quad <- lm(mpg ~ wt + I(wt^2), data = mtcars)
display(lm_quad)

# Visualize the quadratic fit
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2),
              se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Quadratic Fit: mpg ~ wt + wt²",
       x = "Weight (1000 lbs)", y = "MPG") +
  theme_minimal(base_size = 13)
```

### 7.1 Quadratic vs. log: trade-offs

The quadratic model often fits marginally better by $R^2$, but it has two
disadvantages. First, interpretation is harder — the effect of weight changes
at every point along the curve. Second, quadratics extrapolate poorly: outside
the data range, the parabola turns around and predicts absurd values (e.g.,
mpg increasing again for extremely heavy cars). The log model captures
diminishing returns naturally without these pathologies.

In practice: use the quadratic for prediction within the data range; use the
log for communication and interpretation.

---

## 8. Comparing Raw and Log Models: Diagnostic Panels

We bring the full diagnostic comparison together in one place. Using the
simulated earnings data from Section 5:

```r
# Side-by-side: raw vs log
par_raw <- data.frame(
  fitted = fitted(earn_raw.lm),
  resid  = residuals(earn_raw.lm),
  std_r  = rstandard(earn_raw.lm)
)
par_log <- data.frame(
  fitted = fitted(earn_log.lm),
  resid  = residuals(earn_log.lm),
  std_r  = rstandard(earn_log.lm)
)

p1 <- ggplot(par_raw, aes(x = fitted, y = resid)) +
  geom_point(size = 0.8, alpha = 0.3, colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE, colour = "firebrick", linewidth = 0.6, method = "loess") +
  labs(title = "Raw: Residuals vs Fitted", x = "Fitted", y = "Residuals") +
  theme_minimal(base_size = 10)

p2 <- ggplot(par_log, aes(x = fitted, y = resid)) +
  geom_point(size = 0.8, alpha = 0.3, colour = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE, colour = "firebrick", linewidth = 0.6, method = "loess") +
  labs(title = "Log: Residuals vs Fitted", x = "Fitted", y = "Residuals") +
  theme_minimal(base_size = 10)

p3 <- ggplot(par_raw, aes(sample = std_r)) +
  stat_qq(size = 0.8, alpha = 0.3, colour = "steelblue") +
  stat_qq_line(colour = "firebrick") +
  labs(title = "Raw: Q-Q Plot") + theme_minimal(base_size = 10)

p4 <- ggplot(par_log, aes(sample = std_r)) +
  stat_qq(size = 0.8, alpha = 0.3, colour = "steelblue") +
  stat_qq_line(colour = "firebrick") +
  labs(title = "Log: Q-Q Plot") + theme_minimal(base_size = 10)

grid.arrange(p1, p2, p3, p4, ncol = 2)
```

The improvement should be dramatic: funnel gone, Q-Q line restored. This
before-and-after comparison is how you justify a log transform — not by
citing $R^2$ alone, but by showing that the residual assumptions are met.

---

## 9. When to Log: A Decision Checklist

**Log the outcome when:**

- The outcome is strictly positive (income, price, population, concentration)
- The outcome is right-skewed
- The residuals show a fan or funnel shape
- Effects should be proportional ("a 5% increase") rather than absolute ("a \$500 increase")

**Log a predictor when:**

- The predictor shows diminishing returns
- You want an elasticity interpretation
- The predictor is right-skewed and you want to compress the scale

**Do NOT log when:**

- The variable takes zero or negative values ($\log(0) = -\infty$)
- The relationship is genuinely linear (do not fix what is not broken)
- You cannot justify it

---

## 10. Summary: Three Valid Reasons to Transform

Every transformation you apply should be justified by at least one of these
reasons:

1. **Improve interpretation.** Centering makes the intercept meaningful.
   Standardizing makes coefficients comparable.
2. **Improve model fit and diagnostics.** Log transforms fix
   heteroscedasticity and non-normality. Power transforms straighten
   curves. The evidence is in the residual plots — show them before and
   after.
3. **Theoretical motivation.** Log returns in finance. Elasticities in
   economics. Weber's law in psychophysics. Power laws in physics and
   engineering. If the theory says the process is multiplicative, the log
   transform is not optional — it is the correct model.

---

## 11. Exercises

1. **Centering by hand.** Using `mtcars`, fit `lm(mpg ~ disp + hp)`.
   Record the intercept. Now center `disp` and `hp`, refit, and record the
   new intercept. Verify algebraically that the new intercept equals
   $\hat{\beta}_0 + \hat{\beta}_1 \overline{\text{disp}} + \hat{\beta}_2 \overline{\text{hp}}$
   from the raw model. Confirm that the slopes are unchanged.

2. **Standardizing and comparing.** Fit `lm(mpg ~ disp + hp + wt, data = mtcars)`.
   Which predictor has the largest raw coefficient? Standardize all three
   predictors and refit. Which has the largest standardized coefficient?
   Explain why the ranking changed (or did not change).

3. **The four model types.** Using `mtcars`, fit all four models for the
   relationship between `hp` and `mpg`: level-level, log-level, level-log,
   and log-log. For each, state the interpretation of the `hp` coefficient
   in plain English. Report $R^2$ for each. Which model has the best
   residuals-vs-fitted plot?

4. **Diagnosing and fixing heteroscedasticity.** Generate data:
   `set.seed(42); x <- runif(300, 1, 50); y <- exp(1.5 + 0.04*x + rnorm(300, 0, 0.5))`.
   Fit `lm(y ~ x)` and produce the residuals-vs-fitted plot. Diagnose the
   problem. Now fit `lm(log(y) ~ x)` and produce the same plot. Did the
   transform fix the problem? Report the coefficient of `x` in the log
   model and interpret it as a percentage change.

5. **Quadratic vs. log.** Using `mtcars`, fit `lm(mpg ~ wt + I(wt^2))` and
   `lm(log(mpg) ~ log(wt))`. Both capture curvature. For the quadratic
   model, compute the marginal effect of weight at $\text{wt} = 2$ and at
   $\text{wt} = 5$. For the log-log model, state the elasticity. Which
   model would you trust for a prediction at $\text{wt} = 7$ (far outside
   the data)? Why?

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 4.
- Wooldridge, J.M. (2019). *Introductory Econometrics*. 7th ed. Cengage.
  Ch. 6 (Log models).
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*.
  3rd ed. Sage. Ch. 4 (Transformations).
