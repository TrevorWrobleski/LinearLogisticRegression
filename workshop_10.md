# Workshop 10: Multiple Logistic Regression, Risk Ratios, and Average Predictive Comparisons

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop

---

## 1. Logistic regression is linear on the log-odds scale, but non-linear on the probability scale.

In linear regression, a one-unit increase in $x$ changes $y$ by
$\beta_1$, regardless of where you start. In logistic regression, a one-unit
increase in $x$ always changes the *log-odds* by $\beta_1$ — but the
corresponding change in *probability* depends on where you are on the
S-curve. Predicted probabilities, risk ratios, average predictive comparisons 
are strategies for dealing with non-linearity.

---

## 2. Setup

```r
library(ggplot2)
library(arm)
library(gridExtra)
```

We need the inverse-logit function throughout. It converts log-odds back to
probabilities:

```r
# arm::invlogit() does this, but for clarity:
invlogit <- function(x) exp(x) / (1 + exp(x))
```

We also define a classification table function:

```r
ct <- function(model, data, outcome, threshold = 0.5) {
  probs <- predict(model, newdata = data, type = "response")
  pred_class <- ifelse(probs >= threshold, 1, 0)
  actual <- data[[outcome]]
  tab <- table(Predicted = pred_class, Actual = actual)
  cat("Classification Table:\n")
  print(tab)
  cat("\n")
  if (nrow(tab) == 2 && ncol(tab) == 2) {
    cat("% correct (actual = 0):", round(tab[1,1] / sum(tab[,1]) * 100, 1), "\n")
    cat("% correct (actual = 1):", round(tab[2,2] / sum(tab[,2]) * 100, 1), "\n")
    cat("Overall accuracy:      ", round(sum(diag(tab)) / sum(tab) * 100, 1), "\n")
  }
}
```

---

## 3. Multiple Predictors in Logistic Regression

### 3.1 The model

Adding predictors to a logistic regression looks like adding them to
a linear regression byt on the log-odds scale:

$$
\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k
$$

Each $\beta_j$ is the change in log-odds for a one-unit increase in $X_j$,
holding all other predictors constant. The odds ratio $e^{\beta_j}$ is the
factor by which the odds are multiplied. The *ceteris paribus* reasoning
carries over.

### 3.2 Why we center the predictors

The intercept $\beta_0$ is the log-odds of the outcome when every predictor
equals zero. If zero is outside the observed range — and it usually is (no
one has zero age, zero blood pressure, zero income) — the intercept describes
a biologically or economically impossible person. Centering each predictor at
its mean redefines "zero" to mean "average." The intercept becomes the
log-odds for a person at the mean of every predictor — typically a more real, interpretable
quantity. (The slopes do not change.)

```r
center <- function(v) v - mean(v, na.rm = TRUE)
```

### 3.3 Demonstration: ICU survival data

We use a simulated ICU dataset to illustrate. The outcome
is whether a patient died (1) or survived (0). Predictors include age,
systolic blood pressure (SBP), and heart rate (HR).

```r
set.seed(211)
n <- 200
age <- round(runif(n, 20, 90))
sbp <- round(rnorm(n, 120, 25))
hr  <- round(rnorm(n, 80, 15))

# True model: older patients and those with extreme vitals are more likely to die
logit_p <- -4.5 + 0.04 * age - 0.015 * sbp + 0.02 * hr
p_die <- invlogit(logit_p)
died <- rbinom(n, 1, p_die)

icu <- data.frame(died, age, sbp, hr)
icu$age_c <- center(icu$age)
icu$sbp_c <- center(icu$sbp)
icu$hr_c  <- center(icu$hr)

cat("Death rate:", round(mean(died), 3), "\n")
```

### 3.4 Fitting and interpreting the model

```r
icu_glm <- glm(died ~ age_c + sbp_c + hr_c, data = icu, family = binomial)
display(icu_glm)
```

Read the output the same way as linear regression: check each coefficient
against twice its standard error. If $|\hat{\beta}_j| > 2 \times \text{SE}$,
the predictor is significant at approximately the 5% level.

Interpret the coefficients:

```r
coefs <- coef(icu_glm)
cat("Odds ratios:\n")
cat("  Age (per year):      ", round(exp(coefs["age_c"]), 3), "\n")
cat("  SBP (per mmHg):      ", round(exp(coefs["sbp_c"]), 3), "\n")
cat("  HR  (per bpm):       ", round(exp(coefs["hr_c"]), 3), "\n")
```

If the age odds ratio is 1.04, each additional year of age multiplies the
odds of death by 1.04 — a 4% increase in odds per year, holding blood
pressure and heart rate constant.

### 3.5 Deviance: the logistic analogue of $R^2$

In linear regression, we assessed model fit with $R^2$. In logistic
regression, we use **deviance**. The null deviance measures how poorly a
model with no predictors fits. The residual deviance measures how poorly our
model fits. The difference — the deviance reduction — tells us how much the
predictors improved the fit.

To test whether this improvement is statistically significant, compare the
deviance reduction to a $\chi^2$ distribution with $k$ degrees of freedom
(one for each predictor):

```r
null_dev <- icu_glm$null.deviance
resid_dev <- icu_glm$deviance
dev_drop <- null_dev - resid_dev
k <- length(coefs) - 1  # number of predictors

cat("Null deviance:    ", round(null_dev, 1), "\n")
cat("Residual deviance:", round(resid_dev, 1), "\n")
cat("Deviance drop:    ", round(dev_drop, 1), "\n")
cat("Chi-sq critical (df =", k, "):", round(qchisq(0.95, k), 2), "\n")
cat("Model significant:", dev_drop > qchisq(0.95, k), "\n")
```

A useful heuristic: deviance is like negative $R^2$ — lower is better. The
null deviance is the worst you would do; the residual deviance is what
remains after your predictors explain what they can. The bigger the drop, the
better the model.

### 3.6 Comparing models: do we need all predictors?

```r
# Full model: age + sbp + hr
# Reduced model: age only
icu_reduced <- glm(died ~ age_c, data = icu, family = binomial)

cat("Full model deviance:   ", round(icu_glm$deviance, 1), "\n")
cat("Reduced model deviance:", round(icu_reduced$deviance, 1), "\n")
cat("Difference:            ", round(icu_reduced$deviance - icu_glm$deviance, 1), "\n")
```

If the difference is large relative to $\chi^2_{0.95, 2}$ (the number of
dropped predictors), the full model is significantly better. If the
difference is small, the extra predictors are not worth keeping.

### 3.7 Classification tables

```r
ct(icu_glm, icu, "died")
ct(icu_reduced, icu, "died")
```

Compare the two tables. The full model should do better at identifying deaths
(higher sensitivity) without sacrificing much accuracy on survivors
(specificity). Always compare classification accuracy to the **base rate**:
if 80% of patients survive, a model that predicts "survived" for everyone
achieves 80% accuracy while being completely useless. The model must beat the
base rate to be worth anything.

---

## 4. Predicting Probabilities and Risk Ratios

```r
# Use the ICU model. Compare the effect of a 10-year age increase
# at two different starting points.

b <- coef(icu_glm)
mean_sbp <- mean(icu$sbp_c)  # 0 (centered)
mean_hr  <- mean(icu$hr_c)   # 0 (centered)

# Person A: age 30 (young, low risk)
age_A_lo <- 30 - mean(icu$age)
age_A_hi <- 40 - mean(icu$age)

logit_A_lo <- b[1] + b[2] * age_A_lo + b[3] * mean_sbp + b[4] * mean_hr
logit_A_hi <- b[1] + b[2] * age_A_hi + b[3] * mean_sbp + b[4] * mean_hr

p_A_lo <- invlogit(logit_A_lo)
p_A_hi <- invlogit(logit_A_hi)

# Person B: age 65 (older, higher risk)
age_B_lo <- 65 - mean(icu$age)
age_B_hi <- 75 - mean(icu$age)

logit_B_lo <- b[1] + b[2] * age_B_lo + b[3] * mean_sbp + b[4] * mean_hr
logit_B_hi <- b[1] + b[2] * age_B_hi + b[3] * mean_sbp + b[4] * mean_hr

p_B_lo <- invlogit(logit_B_lo)
p_B_hi <- invlogit(logit_B_hi)

cat("Young patient (age 30 → 40):\n")
cat("  Logit change: ", round(logit_A_hi - logit_A_lo, 3), "\n")
cat("  Prob change:  ", round(p_A_hi - p_A_lo, 3),
    " (", round(p_A_lo, 3), "→", round(p_A_hi, 3), ")\n\n")

cat("Older patient (age 65 → 75):\n")
cat("  Logit change: ", round(logit_B_hi - logit_B_lo, 3), "\n")
cat("  Prob change:  ", round(p_B_hi - p_B_lo, 3),
    " (", round(p_B_lo, 3), "→", round(p_B_hi, 3), ")\n")
```

The logit changes are identical — the model is linear on the log-odds scale.
But the probability changes differ, sometimes dramatically. The change is
largest when the starting probability is near 0.5 (the steep middle of the
S-curve) and smallest near 0 or 1 (the flat tails).

### 4.2 Visualizing the S-curve

```r
age_grid <- seq(20, 90, by = 1)
age_grid_c <- age_grid - mean(icu$age)
logit_grid <- b[1] + b[2] * age_grid_c
prob_grid  <- invlogit(logit_grid)

scurve_df <- data.frame(age = age_grid, probability = prob_grid)

ggplot(scurve_df, aes(x = age, y = probability)) +
  geom_line(linewidth = 0.9, colour = "steelblue") +
  geom_segment(aes(x = 30, xend = 40, y = p_A_lo, yend = p_A_lo),
               linetype = "dashed", colour = "firebrick") +
  geom_segment(aes(x = 40, xend = 40, y = p_A_lo, yend = p_A_hi),
               colour = "firebrick", linewidth = 0.7) +
  geom_segment(aes(x = 65, xend = 75, y = p_B_lo, yend = p_B_lo),
               linetype = "dashed", colour = "darkgreen") +
  geom_segment(aes(x = 75, xend = 75, y = p_B_lo, yend = p_B_hi),
               colour = "darkgreen", linewidth = 0.7) +
  annotate("text", x = 42, y = (p_A_lo + p_A_hi) / 2,
           label = paste0("Δp = ", round(p_A_hi - p_A_lo, 3)),
           colour = "firebrick", size = 3.5, hjust = 0) +
  annotate("text", x = 77, y = (p_B_lo + p_B_hi) / 2,
           label = paste0("Δp = ", round(p_B_hi - p_B_lo, 3)),
           colour = "darkgreen", size = 3.5, hjust = 0) +
  labs(title = "Same Coefficient, Different Probability Changes",
       subtitle = "Red: age 30→40 (flat part of curve). Green: age 65→75 (steeper part).",
       x = "Age", y = "P(Death)") +
  theme_minimal(base_size = 13)
```

The vertical red and green segments show the probability change for the same
10-year age increase at two different starting points. Where the curve is
steep, the same horizontal shift produces a larger vertical change.

### 4.3 Risk ratios

A risk ratio is the ratio of two probabilities:

$$
\text{RR} = \frac{P(Y = 1 \mid X = x_{\text{high}})}{P(Y = 1 \mid X = x_{\text{low}})}
$$

```r
rr_young <- p_A_hi / p_A_lo
rr_older <- p_B_hi / p_B_lo

cat("Risk ratio (age 30→40):", round(rr_young, 2), "\n")
cat("Risk ratio (age 65→75):", round(rr_older, 2), "\n")
```

The risk ratio for young patients will typically be larger — going from a 5%
to a 10% risk is a doubling (RR = 2.0), while going from 40% to 50% is only
a 25% relative increase (RR = 1.25). Same coefficient, same absolute age
change, very different relative change.

A risk ratio in isolation can be misleading. An "80% increase in risk" sounds
alarming, but 80% of a 3% baseline is only 5.4%. It's best to present risk ratios
alongside the actual probabilities.

### 4.4 Risk ratios are not odds ratios

This distinction matters and students frequently confuse the two. The odds
ratio ($e^{\hat{\beta}}$) is a ratio of *odds*. The risk ratio is a ratio of
*probabilities*. They are approximately equal when the event is rare (the
"rare disease assumption"), but diverge as the baseline probability increases.
For a baseline near 0.5, the odds ratio overstates the relative risk
considerably.

---

## 5. Average Predictive Comparisons (APC)

### 5.1 The problem APC solves

We have just seen that the probability change from a 10-year age increase
depends on the starting age and on the values of every other predictor. If
someone asks "what is the effect of age?", it
depends on which patient you are talking about.

The Average Predictive Comparison gives a single number by computing
the effect for *every person in the dataset* and averaging.

### 5.2 The algorithm

For a predictor of interest $X_j$, choose a "low" value $x_j^{\text{lo}}$
and a "high" value $x_j^{\text{hi}}$ (typically the observed range). Then:

1. For **every observation** $i$ in the dataset, compute the predicted
   probability at $x_j^{\text{hi}}$ and at $x_j^{\text{lo}}$, using
   observation $i$'s actual values for all other predictors.
2. Take the difference: $\Delta_i = \hat{p}_i^{\text{hi}} - \hat{p}_i^{\text{lo}}$.
3. Average: $\text{APC}_j = \frac{1}{n} \sum_{i=1}^n \Delta_i$.

The APC accounts for the fact that different people sit at different places
on the S-curve. Instead of picking one "representative" person, it computes
the effect for everyone and averages.

### 5.3 APC in R

```r
# APC for a 20-year age increase (age 35 → 55) in the ICU model
b <- coef(icu_glm)
lo_age <- 35 - mean(icu$age)
hi_age <- 55 - mean(icu$age)

delta_age <- invlogit(b[1] + b[2] * hi_age + b[3] * icu$sbp_c + b[4] * icu$hr_c) -
             invlogit(b[1] + b[2] * lo_age + b[3] * icu$sbp_c + b[4] * icu$hr_c)

apc_age <- mean(delta_age)
cat("APC for age (35 → 55):", round(apc_age, 3), "\n")
```

Read each piece: `b[2] * hi_age` plugs in the "high" age. `b[3] * icu$sbp_c`
uses each patient's actual centered SBP — this is a vector of $n$ values,
not a single mean. The subtraction gives $n$ individual $\Delta_i$ values.
`mean()` averages them.

### 5.4 APC for all predictors

```r
# APC for SBP: from 5th percentile to 95th percentile
lo_sbp <- quantile(icu$sbp_c, 0.05)
hi_sbp <- quantile(icu$sbp_c, 0.95)

delta_sbp <- invlogit(b[1] + b[2] * icu$age_c + b[3] * hi_sbp + b[4] * icu$hr_c) -
             invlogit(b[1] + b[2] * icu$age_c + b[3] * lo_sbp + b[4] * icu$hr_c)
apc_sbp <- mean(delta_sbp)

# APC for HR: from 5th to 95th percentile
lo_hr <- quantile(icu$hr_c, 0.05)
hi_hr <- quantile(icu$hr_c, 0.95)

delta_hr <- invlogit(b[1] + b[2] * icu$age_c + b[3] * icu$sbp_c + b[4] * hi_hr) -
            invlogit(b[1] + b[2] * icu$age_c + b[3] * icu$sbp_c + b[4] * lo_hr)
apc_hr <- mean(delta_hr)

cat("APC summary:\n")
cat("  Age (35 → 55):          ", round(apc_age, 3), "\n")
cat("  SBP (5th → 95th pct):   ", round(apc_sbp, 3), "\n")
cat("  HR  (5th → 95th pct):   ", round(apc_hr, 3), "\n")
```

### 5.5 Visualizing the distribution of individual effects

The APC is a mean, but the individual $\Delta_i$ values have a distribution.
Examining this distribution shows how much the effect varies across patients.

```r
delta_df <- data.frame(
  delta = c(delta_age, delta_sbp, delta_hr),
  predictor = rep(c("Age (35→55)", "SBP (5th→95th)", "HR (5th→95th)"),
                  each = nrow(icu))
)

ggplot(delta_df, aes(x = delta, fill = predictor)) +
  geom_histogram(bins = 30, alpha = 0.7, colour = "white") +
  geom_vline(data = data.frame(
    predictor = c("Age (35→55)", "SBP (5th→95th)", "HR (5th→95th)"),
    apc = c(apc_age, apc_sbp, apc_hr)),
    aes(xintercept = apc), linetype = "dashed", linewidth = 0.7) +
  facet_wrap(~ predictor, scales = "free_x") +
  scale_fill_manual(values = c("steelblue", "firebrick", "darkgreen")) +
  labs(title = "Distribution of Individual Predictive Differences",
       subtitle = "Dashed line = APC (mean of the distribution)",
       x = "Change in Predicted Probability", y = "Count") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
```

If the distribution is tight, the APC is a good summary — the effect is
roughly the same for everyone. If the distribution is spread out, the APC
is still valid as a mean, but it conceals heterogeneity. Some
patients experience a large probability change and others almost none,
depending on where they sit on the S-curve.

### 5.6 Why APC is better than plugging in means

Setting all other predictors to their means and
computing the probability change gives a number that applies only to a
patient with *exactly average* vitals. If a patient has unusually high heart
rate, the number is different.

The APC uses every patient's actual predictor values, computes the effect for
each, and averages. It reflects the *population*, not one hypothetical
individual. 

---

## 6. A Complete Example: `mtcars` Logistic Regression

We create a binary outcome from `mtcars` — whether a car achieves "high"
fuel efficiency (mpg $>$ 20) — and run the full workflow: fitting,
interpreting, predicting, risk ratios, and APC. No external data required.

```r
data(mtcars)
mtcars$high_mpg <- ifelse(mtcars$mpg > 20, 1, 0)
mtcars$wt_c <- center(mtcars$wt)
mtcars$hp_c <- center(mtcars$hp)

cat("High mpg rate:", round(mean(mtcars$high_mpg), 3), "\n")
```

### 6.1 Fit the model

```r
car_glm <- glm(high_mpg ~ wt_c + hp_c + factor(am), data = mtcars, family = binomial)
display(car_glm)
```

Interpret: the weight coefficient (negative) means heavier cars are less
likely to achieve high mpg. The transmission coefficient tells you the odds
ratio for manual vs. automatic, holding weight and horsepower constant.

```r
cat("Odds ratios:\n")
cat("  Weight (per 1000 lbs):", round(exp(coef(car_glm)["wt_c"]), 3), "\n")
cat("  HP (per unit):        ", round(exp(coef(car_glm)["hp_c"]), 3), "\n")
cat("  Manual vs Auto:       ", round(exp(coef(car_glm)["factor(am)1"]), 3), "\n")
```

### 6.2 Deviance check

```r
dev_drop <- car_glm$null.deviance - car_glm$deviance
cat("Deviance drop:", round(dev_drop, 1), "\n")
cat("Critical value (df = 3):", round(qchisq(0.95, 3), 2), "\n")
cat("Model significant:", dev_drop > qchisq(0.95, 3), "\n")
```

### 6.3 Predicted probabilities and risk ratios

```r
b <- coef(car_glm)

# Light car (2000 lbs) vs heavy car (4000 lbs), average hp, automatic
wt_light <- 2.0 - mean(mtcars$wt)
wt_heavy <- 4.0 - mean(mtcars$wt)
hp_avg   <- 0  # centered

p_light <- invlogit(b[1] + b[2] * wt_light + b[3] * hp_avg + b[4] * 0)
p_heavy <- invlogit(b[1] + b[2] * wt_heavy + b[3] * hp_avg + b[4] * 0)

cat("P(high mpg | 2000 lbs):", round(p_light, 3), "\n")
cat("P(high mpg | 4000 lbs):", round(p_heavy, 3), "\n")
cat("Risk ratio:            ", round(p_light / p_heavy, 2), "\n")
```

### 6.4 APC for weight

```r
lo_wt <- min(mtcars$wt_c)
hi_wt <- max(mtcars$wt_c)

delta_wt <- invlogit(b[1] + b[2] * lo_wt + b[3] * mtcars$hp_c + b[4] * mtcars$am) -
            invlogit(b[1] + b[2] * hi_wt + b[3] * mtcars$hp_c + b[4] * mtcars$am)

apc_wt <- mean(delta_wt)
cat("APC for weight (lightest → heaviest):", round(apc_wt, 3), "\n")
cat("Interpretation: on average, going from the lightest to the heaviest car\n")
cat("  decreases the probability of high mpg by", round(abs(apc_wt) * 100, 1),
    "percentage points.\n")
```

### 6.5 Classification table

```r
ct(car_glm, mtcars, "high_mpg")
```

Compare the overall accuracy to the base rate. If 44% of cars have high mpg,
a model that predicts "low mpg" for everyone gets 56% accuracy. The model
must substantially exceed this to be useful.

---

## 7. Key Formulas

| Concept | Formula |
|---|---|
| Logistic model | $\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k$ |
| Predicted probability | $p = \text{invlogit}(\eta) = \frac{e^{\eta}}{1 + e^{\eta}}$ |
| Odds ratio (1-unit change) | $\text{OR} = e^{\beta_j}$ |
| Odds ratio ($\Delta$-unit change) | $\text{OR} = e^{\Delta \cdot \beta_j}$ |
| Risk ratio | $\text{RR} = P(Y=1 \mid X = x_{\text{hi}}) \;/\; P(Y=1 \mid X = x_{\text{lo}})$ |
| APC for predictor $j$ | $\text{APC}_j = \frac{1}{n}\sum_{i=1}^n \left[\text{invlogit}(\hat{\eta}_i^{\text{hi}}) - \text{invlogit}(\hat{\eta}_i^{\text{lo}})\right]$ |
| Deviance test | Compare (null deviance $-$ residual deviance) to $\chi^2_{0.95, k}$ |
| Sensitivity | True positives / (true positives + false negatives) |
| Specificity | True negatives / (true negatives + false positives) |

---

## 8. Exercises

1. **Interpretation drill.** Using `car_glm` from Section 6, the weight
   coefficient is some negative number $\hat{\beta}_{\text{wt}}$. (a) State
   the odds ratio. (b) Compute the predicted probability of high mpg for a
   3,000 lb automatic car with average horsepower. (c) Repeat for a 3,000 lb
   manual car. (d) What is the risk ratio for manual vs. automatic at this
   weight?

2. **S-curve visualization.** Using `car_glm`, produce the S-curve of
   predicted probability vs. weight for automatic cars at average horsepower
   (analogous to Section 4.2). Mark the probability at 2,000 lbs and at
   4,000 lbs. Draw the vertical segments showing the probability change for
   a 500 lb increase at each starting point. Confirm that the change is
   larger where the curve is steeper.

3. **Deviance comparison.** Fit two models: (a) `high_mpg ~ wt_c` and
   (b) `high_mpg ~ wt_c + hp_c + factor(am)`. Compare their deviances. Is
   the full model significantly better? Produce classification tables for
   both. Which model better identifies the minority class?

4. **APC computation.** Compute the APC for horsepower in `car_glm`, using
   the 10th and 90th percentiles of `hp_c` as the low and high values.
   Plot the histogram of individual $\Delta_i$ values. Is the distribution
   tight or spread out? What does this tell you about the heterogeneity of
   the horsepower effect across cars?

5. **Risk ratio vs. odds ratio.** Using `car_glm`, compute both the risk
   ratio and the odds ratio for a 1,000 lb increase in weight, starting at
   (a) 2,000 lbs and (b) 4,000 lbs. At which starting point do the two
   quantities differ more? Explain why, in terms of the baseline
   probability.

6. **Full analysis.** Create a binary variable in `mtcars`: `fast <- ifelse(mtcars$qsec < median(mtcars$qsec), 1, 0)` (below-median quarter-mile
   time = "fast"). Fit a logistic regression predicting `fast` from `hp`,
   `wt`, and `factor(cyl)`. Center the continuous predictors. Report the
   odds ratios. Produce the classification table and compare to the base
   rate. Compute the APC for horsepower over its observed range. Write a
   one-paragraph interpretation of the results.

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 5–6.
- Agresti, A. (2013). *Categorical Data Analysis*. 3rd ed. Wiley. Ch. 5.
- Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X. (2013). *Applied Logistic
  Regression*. 3rd ed. Wiley. Ch. 1–4.
