# Workshop 8: Prediction and Cross-Validation

**ST211 — Linear and Logistic Regression** 
Supplemental Material for the workshop

---

## 1. Review: Interpreting Log Models

Before we can evaluate predictions, we need to be certain about what our
models are saying. If the outcome is logged, predictions come out on the log
scale and must be back-transformed. If you cannot interpret the coefficients,
you cannot assess whether the predictions are reasonable.

### 1.1 The four model types on `mtcars`

We fit all four combinations of logging (or not) the outcome and predictor,
using the same data so the differences are concrete.

```r
library(ggplot2)
library(arm)
library(gridExtra)

data(mtcars)

lm_ll <- lm(mpg ~ wt, data = mtcars)          # level-level
lm_Ll <- lm(log(mpg) ~ wt, data = mtcars)     # log-level
lm_lL <- lm(mpg ~ log(wt), data = mtcars)     # level-log
lm_LL <- lm(log(mpg) ~ log(wt), data = mtcars) # log-log
```

```r
p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Level–Level: y ~ x", x = "Weight (1000 lbs)", y = "MPG") +
  theme_minimal(base_size = 11)

p2 <- ggplot(mtcars, aes(x = wt, y = log(mpg))) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Log–Level: log(y) ~ x", x = "Weight (1000 lbs)", y = "log(MPG)") +
  theme_minimal(base_size = 11)

p3 <- ggplot(mtcars, aes(x = log(wt), y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Level–Log: y ~ log(x)", x = "log(Weight)", y = "MPG") +
  theme_minimal(base_size = 11)

p4 <- ggplot(mtcars, aes(x = log(wt), y = log(mpg))) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  labs(title = "Log–Log: log(y) ~ log(x)", x = "log(Weight)", y = "log(MPG)") +
  theme_minimal(base_size = 11)

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

### 1.2 Interpretation reference

| Model | R syntax | $\hat{\beta}_1$ means... | Prediction needs... |
|---|---|---|---|
| Level–level | `y ~ x` | 1 unit $\uparrow x$ → $\hat{\beta}_1$ units $\Delta y$ | Nothing extra |
| Log–level | `log(y) ~ x` | 1 unit $\uparrow x$ → $\times e^{\hat{\beta}_1}$ on $y$ | `exp(predict(...))` |
| Level–log | `y ~ log(x)` | 1% $\uparrow x$ → $\hat{\beta}_1 / 100$ units $\Delta y$ | Nothing extra |
| Log–log | `log(y) ~ log(x)` | 1% $\uparrow x$ → $\hat{\beta}_1$% $\Delta y$ (elasticity) | `exp(predict(...))` |

### 1.3 Why we exponentiate: the log-level derivation, line by line

Let's say we have fitted the model:

$$
\log(y_i) = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

We want to know what happens to $y$ when $x$ increases by one unit. Consider
two observations — one at $x$ and one at $x + 1$ — with the same error term
for simplicity:

$$
\log(y_{\text{new}}) = \beta_0 + \beta_1(x + 1)
$$

$$
\log(y_{\text{old}}) = \beta_0 + \beta_1 x
$$

**Step 1. Subtract.** Take the second equation from the first:

$$
\log(y_{\text{new}}) - \log(y_{\text{old}}) = \beta_1(x + 1) - \beta_1 x = \beta_1
$$

Everything cancels except $\beta_1$. The intercept drops out. The value of
$x$ drops out. The effect of a one-unit change in $x$ is the same regardless 
of where you start on the $x$-axis. That is, $\beta_1$ does not depend on $x$.

**Step 2. Apply the log quotient rule.** Recall that
$\log(a) - \log(b) = \log(a / b)$. Apply it to the left-hand side:

$$
\log\!\left(\frac{y_{\text{new}}}{y_{\text{old}}}\right) = \beta_1
$$

The left-hand side is now the log of a *ratio*. 

**Step 3. Exponentiate both sides.** To undo the log, apply $e^{(\cdot)}$
to both sides. Since $e^{\log(z)} = z$:

$$
\frac{y_{\text{new}}}{y_{\text{old}}} = e^{\beta_1}
$$

A one-unit increase in $x$ **multiplies** $y$ by
$e^{\beta_1}$.

**Step 4. Convert to a percentage change.** If the ratio is $e^{\beta_1}$,
then the percentage change is:

$$
\%\Delta y = \left(e^{\beta_1} - 1\right) \times 100\%
$$

We subtract 1 because a multiplier of 1.0 means "no change." Anything above
1 is a percentage increase; anything below 1 is a percentage decrease.

**Concrete example.** In the `mtcars` log-level model, $\hat{\beta}_1 = -0.25$.
Walk through each step with the number:

| Step | Expression | Value |
|---|---|---|
| The coefficient | $\hat{\beta}_1$ | $-0.25$ |
| Exponentiate | $e^{-0.25}$ | $0.7788$ |
| Interpret as multiplier | $y$ is multiplied by... | $0.7788$ (i.e., 77.9% of original) |
| Convert to % change | $(0.7788 - 1) \times 100$ | $-22.1\%$ |

Each additional 1,000 lbs of weight multiplies fuel efficiency by 0.779,
which is a 22.1% decrease. This percentage applies equally whether you start
at 30 mpg (lose 6.6) or at 15 mpg (lose 3.3). That is the power of the
multiplicative interpretation — the proportional effect is constant, even
though the absolute effect is not.

**Why the small-$\beta$ shortcut works (and when it doesn't).** The Taylor
expansion of $e^{\beta}$ around zero is:

$$
e^{\beta} = 1 + \beta + \frac{\beta^2}{2} + \cdots
$$

When $|\beta|$ is small, the higher-order terms are negligible, so
$e^{\beta} \approx 1 + \beta$, and the percentage change is approximately
$\beta \times 100\%$. At $\beta = 0.05$, the exact answer is 5.13% and the
approximation gives 5% — close enough. At $\beta = 0.25$, the exact answer
is 28.4% and the approximation gives 25% — a 3-percentage-point error that
matters. The rule of thumb: use the shortcut for quick intuition, but always
compute $e^{\hat{\beta}}$ for anything you report.


### 1.4 Back-transforming predictions

If we want the predicted mpg (not log-mpg) for a 3,000 lb car:

```r
log_pred <- predict(lm_Ll, data.frame(wt = 3.0))
cat("Predicted log(mpg):", round(log_pred, 3), "\n")
cat("Predicted mpg:     ", round(exp(log_pred), 1), "\n")
```

Whenever you predict from a log model, the raw output is on the log scale.
You must exponentiate to recover the original units. This applies to
prediction intervals as well — exponentiate both bounds:

```r
pi <- predict(lm_Ll, data.frame(wt = 3.0), interval = "prediction")
cat("On log scale:", round(pi, 3), "\n")
cat("On mpg scale:", round(exp(pi), 1), "\n")
```

The exponentiated interval is asymmetric around the point prediction — wider
on the upper side — which reflects the right-skewed nature of the original
variable.

### 1.5 Categorical predictors in a log model

For a dummy variable $D$ (0/1) in a log model:

$$
\frac{\hat{y}_{D=1}}{\hat{y}_{D=0}} = e^{\hat{\beta}_2}
$$

```r
lm_am <- lm(log(mpg) ~ wt + factor(am), data = mtcars)
display(lm_am)
```

If the coefficient of `factor(am)1` is 0.15, then manual cars get
$e^{0.15} = 1.162$ times the mpg of automatics at the same weight — a 16.2%
advantage. This scales naturally: 1.6 extra mpg on a 10-mpg truck, 4.9 extra
mpg on a 30-mpg sedan. The percentage interpretation is always more
informative than an absolute difference.

---

## 2. Prediction: The Mechanics

### 2.1 The `predict()` function

`predict()` takes a fitted model and a data frame of new predictor values.
Adding `interval = "prediction"` gives a predictive interval that accounts
for both estimation uncertainty and irreducible error.

```r
# Using the level-level model
pred_10 <- predict(lm_ll, mtcars[10, ], interval = "prediction")
pred_10
```

The point estimate is the model's best guess. The interval tells you where a
*new individual observation* at those predictor values is likely to fall.

### 2.2 Confidence interval vs. prediction interval

**Confidence interval** (`interval = "confidence"`):

$$
\hat{y} \pm t_{\alpha/2} \cdot \text{SE}(\hat{y})
$$

This captures uncertainty about the *mean response* at a given $x$. It
answers: "Where is the true regression line?"

**Prediction interval** (`interval = "prediction"`):

$$
\hat{y} \pm t_{\alpha/2} \cdot \sqrt{\text{SE}(\hat{y})^2 + \hat{\sigma}^2}
$$

This captures uncertainty for a *new individual observation*. It answers:
"Where will the next data point actually fall?"

The prediction interval is wider because it includes the residual
variance $\hat{\sigma}^2$. Even if you knew the true regression line
perfectly, individual observations scatter around it.

```r
ci <- predict(lm_ll, mtcars[10, ], interval = "confidence")
pi <- predict(lm_ll, mtcars[10, ], interval = "prediction")
cat("Confidence interval width:", round(ci[3] - ci[2], 2), "\n")
cat("Prediction interval width:", round(pi[3] - pi[2], 2), "\n")
```

---

## 3. In-Sample vs. Out-of-Sample Prediction

### 3.1 The obvious case: beyond the data range

If weight in `mtcars` ranges from 1.5 to 5.4 thousand lbs, predicting mpg
for an 8,000-lb vehicle is extrapolation. The model has no observations in
that region and the prediction is unreliable.

### 3.2 The subtle case: within range but out of sample

Consider two predictors. A point can have each predictor within its observed
range, yet the *combination* may not exist in the data. This is extrapolation
in the joint predictor space, even though it looks like interpolation in each
margin.

```r
ggplot(mtcars, aes(x = wt, y = hp)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  annotate("point", x = 3.2, y = 150, colour = "darkgreen",
           size = 4, shape = 17) +
  annotate("text", x = 3.4, y = 155, label = "In-sample",
           colour = "darkgreen", size = 3.5, hjust = 0) +
  annotate("point", x = 1.8, y = 300, colour = "firebrick",
           size = 4, shape = 17) +
  annotate("text", x = 2.0, y = 305, label = "Out-of-sample\n(both values in range!)",
           colour = "firebrick", size = 3, hjust = 0) +
  labs(title = "Joint Predictor Space: In-Sample vs. Out-of-Sample",
       x = "Weight (1000 lbs)", y = "Horsepower") +
  theme_minimal(base_size = 13)
```

A light car (1,800 lbs) with 300 hp — both values are within their
individual ranges, but no such car exists in the data. The model has never
seen this combination and the prediction at that point is extrapolation.

This connects directly to the leverage concept from Workshop 5: a point that
is extreme in the joint predictor space has high leverage. An out-of-sample
prediction point would have *higher* leverage than any observed point.

---

## 4. The Cat Problem: Why Cross-Validation Matters

Before we touch the mechanics, let's consider an example.

You are teaching a child to identify cats. You have 10 cat breeds in another
room, one specimen of each: a Scottish Fold, a Maine Coon, an orange tabby,
a Siamese, a Bengal, a Ragdoll, a British Shorthair, a Persian, an Abyssinian,
and a Sphynx. The child has never seen any of them. You will teach the child
on some breeds and then test on the rest.

**The 90/10 split.** You show the child 9 breeds and hold back 1 for
testing. The child studies those 9 cats develops a mental
model: cats have fur, (mostly) pointed ears, whiskers, tails, slit pupils. Then you
bring out the Sphynx — hairless, wrinkled, bat-eared. The child stares at it
and says, "That is not a cat." The model fails the test completely.

What went wrong? The child learned the *training data* extremely well. Nine
breeds provided a detailed picture of what a cat looks like. But the model
overfit to the specific features of those 9 breeds (especially the fur), and
the single test observation happened to be the one breed that violates that
feature. With only 1 test case, the assessment is noisy and misleading because of one
unlucky holdout.

**The 50/50 split.** You show the child 5 breeds and test on the other 5.
The child has less training data, so its mental model is coarser — maybe just
"four legs, a tail, pointed ears, and smaller than a dog." But when tested on
the remaining 5 breeds, including the Sphynx, the child gets 4 out of 5
right (the coarser model is more robust). The evaluation is based on 5 test
cases instead of 1, so it gives a more reliable picture of how well the child
generalizes.

**The trade-off.** More training data builds a richer model but leaves fewer
cases for evaluation, making the assessment noisy. More test data gives a
reliable evaluation but starves the model of learning material. The standard
compromise — 70/30 or 80/20 — balances both. And the real solution is to run
the exercise multiple times with different splits: sometimes the Sphynx is in
training, sometimes it is in testing. Over many iterations, you get a better
picture of the model's ability to generalize.

This is exactly what cross-validation does with regression models. The
"breeds" are observations. The "mental model" is the fitted regression. The
test is prediction on held-out data. MSPE — the average squared prediction
error on the test set — is the report card.

---

## 5. Cross-Validation: Implementation

### 5.1 A reusable CV function

```r
run_cv <- function(data, formula, outcome, split = 0.8, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(data)
  idx <- sample(1:n, floor(split * n))
  train_lm <- lm(formula, data = data[idx, ])
  preds    <- predict(train_lm, newdata = data[-idx, ])
  actuals  <- data[-idx, outcome]
  list(
    predicted = preds,
    actual    = actuals,
    error     = actuals - preds,
    mse_in    = mean(residuals(train_lm)^2),
    mspe_out  = mean((actuals - preds)^2)
  )
}
```

### 5.2 Single split on `mtcars`

We predict mpg from weight, horsepower, and transmission using an 80/20
split.

```r
set.seed(211)
n <- nrow(mtcars)
train_idx <- sample(1:n, floor(0.8 * n))
train <- mtcars[train_idx, ]
test  <- mtcars[-train_idx, ]

cat("Training set:", nrow(train), "cars\n")
cat("Test set:    ", nrow(test), "cars\n")

train_lm <- lm(mpg ~ wt + hp + factor(am), data = train)
test_pred <- predict(train_lm, newdata = test)
test_actual <- test$mpg
test_error  <- test_actual - test_pred
```

### 5.3 Visualizing the results

```r
cv_df <- data.frame(predicted = test_pred, actual = test_actual,
                    error = test_error)

p_pa <- ggplot(cv_df, aes(x = predicted, y = actual)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              colour = "firebrick") +
  labs(title = "Predicted vs. Actual",
       subtitle = "Dashed line = perfect prediction",
       x = "Predicted MPG", y = "Actual MPG") +
  theme_minimal(base_size = 12)

p_pe <- ggplot(cv_df, aes(x = predicted, y = error)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "firebrick") +
  geom_hline(yintercept = c(-2 * sd(test_error), 2 * sd(test_error)),
             linetype = "dotted", colour = "grey50") +
  labs(title = "Predicted vs. Error",
       subtitle = "Dotted lines = +/- 2 SD of errors",
       x = "Predicted MPG", y = "Error (Actual - Predicted)") +
  theme_minimal(base_size = 12)

grid.arrange(p_pa, p_pe, ncol = 2)
```

In the predicted-vs-actual plot, points on the dashed diagonal are perfect
predictions. Systematic departures from the diagonal indicate bias. In the
predicted-vs-error plot, random scatter around zero indicates a well-calibrated
model; a trend or funnel indicates a problem the model has not captured.

---

## 6. Mean Squared Predictive Error (MSPE)

### 6.1 Definition

The MSPE summarizes cross-validation performance in a single number:

$$
\text{MSPE} = \frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} (y_i - \hat{y}_i)^2
$$

This is the out-of-sample analogue of the in-sample MSE. If the model
generalizes well, the MSPE should be close to (but slightly larger than) the
in-sample MSE. If the MSPE is *much* larger, the model is overfit — it
learned noise in the training data that does not carry over to new
observations.

```r
mse_in   <- mean(residuals(train_lm)^2)
mspe_out <- mean(test_error^2)

cat("In-sample MSE: ", round(mse_in, 2), "\n")
cat("Out-sample MSPE:", round(mspe_out, 2), "\n")
cat("Ratio (MSPE/MSE):", round(mspe_out / mse_in, 2), "\n")
```

A ratio near 1 means limited overfitting. A ratio of 2 or more means the
model is substantially worse on new data than on its training data.

---

## 7. Multiple Iterations and Split Ratios

### 7.1 Why one split is not enough

A single random split can be lucky or unlucky. If the test set happens to
contain the easiest-to-predict observations, the MSPE looks artificially
good. If it contains the hardest, the MSPE looks artificially bad. We need
multiple iterations with different random splits.

```r
mspe_results <- sapply(1:10, function(i) {
  cv <- run_cv(mtcars, mpg ~ wt + hp + factor(am), "mpg",
               split = 0.8, seed = i * 100)
  cv$mspe_out
})

cat("MSPE across 10 iterations:\n")
cat(round(mspe_results, 2), "\n")
cat("Mean MSPE:", round(mean(mspe_results), 2), "\n")
cat("SD of MSPE:", round(sd(mspe_results), 2), "\n")
```

If the standard deviation of MSPE across iterations is large relative to the
mean, the model's predictive quality is unstable — it depends heavily on which
observations end up in the test set. This is a warning sign, especially
common with small datasets.

### 7.2 Comparing split ratios

```r
splits <- c(0.50, 0.70, 0.80, 0.90)
split_results <- data.frame()

for (s in splits) {
  mspes <- sapply(1:20, function(i) {
    cv <- run_cv(mtcars, mpg ~ wt + hp + factor(am), "mpg",
                 split = s, seed = i * 50)
    cv$mspe_out
  })
  split_results <- rbind(split_results,
    data.frame(split = paste0(s * 100, "/", (1 - s) * 100),
               mspe = mspes))
}

ggplot(split_results, aes(x = split, y = mspe)) +
  geom_boxplot(fill = "steelblue", alpha = 0.5) +
  labs(title = "MSPE by Split Ratio (20 iterations each)",
       subtitle = "mtcars: mpg ~ wt + hp + am",
       x = "Train/Test Split", y = "MSPE") +
  theme_minimal(base_size = 13)
```

What you should observe:

- **50/50:** the MSPE distribution is relatively tight (reliable assessment)
  but the median is higher (the model is trained on too little data).
- **90/10:** the median MSPE may be lower (better model) but the distribution
  is wide (the assessment is noisy because only 3 cars are in the test set).
- **70/30 and 80/20:** a reasonable compromise.

| Split | Advantage | Disadvantage |
|---|---|---|
| 50/50 | Large test set, reliable MSPE estimate | Model trained on too little data |
| 70/30 | Good compromise | Standard choice |
| 80/20 | Good model estimation | Smaller test set |
| 90/10 | Best model quality | Very small test set, noisy MSPE |

Think back to the cat analogy: the 90/10 split is showing the child 9 breeds
and testing on 1. The child's knowledge is deep, but if that 1 test breed
happens to be the Sphynx, you conclude the child knows nothing about cats.
That conclusion is wrong — it is an artifact of a tiny, unrepresentative test
set.

---

## 8. Using MSPE for Model Comparison

Cross-validation's real power is comparing models. In-sample $R^2$ always
favors the more complex model. MSPE does not — it penalizes complexity that
does not generalize.

### 8.1 Full vs. reduced model on `mtcars`

```r
# Compare three models across 20 CV iterations
model_comparison <- data.frame()

for (i in 1:20) {
  set.seed(i * 77)
  idx <- sample(1:n, floor(0.75 * n))
  tr <- mtcars[idx, ]
  te <- mtcars[-idx, ]

  m_simple <- lm(mpg ~ wt, data = tr)
  m_medium <- lm(mpg ~ wt + hp, data = tr)
  m_full   <- lm(mpg ~ wt + hp + factor(am) + factor(cyl) + disp, data = tr)

  model_comparison <- rbind(model_comparison, data.frame(
    iter = i,
    Simple = mean((te$mpg - predict(m_simple, te))^2),
    Medium = mean((te$mpg - predict(m_medium, te))^2),
    Full   = mean((te$mpg - predict(m_full, te))^2)
  ))
}

# Summary
comp_long <- tidyr::pivot_longer(model_comparison, -iter,
                                  names_to = "Model", values_to = "MSPE")

ggplot(comp_long, aes(x = Model, y = MSPE, fill = Model)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("Simple" = "steelblue",
                                "Medium" = "grey60",
                                "Full" = "firebrick")) +
  labs(title = "Cross-Validated MSPE: Three Models",
       subtitle = "20 iterations of 75/25 splits on mtcars",
       x = NULL, y = "MSPE") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
```

```r
cat("Mean MSPE:\n")
cat("  Simple (wt only):         ", round(mean(model_comparison$Simple), 2), "\n")
cat("  Medium (wt + hp):         ", round(mean(model_comparison$Medium), 2), "\n")
cat("  Full (wt + hp + am + ...): ", round(mean(model_comparison$Full), 2), "\n")
```

If the Full model has a lower mean MSPE than the Medium model, the additional
predictors carry real information that generalizes. If the Full model has a
*higher* or similar mean MSPE but with more variability, the extra
predictors are overfitting — they capture noise in the training set that
does not replicate.

### 8.2 The bias-variance trade-off

A simple model is consistently mediocre: low variance across splits, but
high bias (it misses real patterns). A complex model is sometimes brilliant
and sometimes terrible: low bias, but high variance (it fits noise as if it
were signal). Cross-validation reveals where the balance lies.

$$
\text{MSPE} \approx \underbrace{\text{Bias}^2}\_{\text{model too simple}} + \underbrace{\text{Variance}}\_{\text{model too complex}} + \underbrace{\sigma^2}\_{\text{irreducible noise}}
$$

You cannot reduce the irreducible noise $\sigma^2$ — it is a property of the
data, not the model. The goal is to find the model that minimizes the sum of
bias$^2$ and variance.

---

## 9. Cross-Validating a Log Model

When the model uses `log(y)`, predictions emerge on the log scale. The MSPE
must be computed on the *original* scale to be meaningful — you are
evaluating how well the model predicts actual mpg, not log-mpg.

```r
log_comparison <- data.frame()

for (i in 1:20) {
  set.seed(i * 77)
  idx <- sample(1:n, floor(0.75 * n))
  tr <- mtcars[idx, ]
  te <- mtcars[-idx, ]

  m_raw  <- lm(mpg ~ wt + hp, data = tr)
  m_log  <- lm(log(mpg) ~ wt + hp, data = tr)

  pred_raw <- predict(m_raw, te)
  pred_log <- exp(predict(m_log, te))  # back-transform!

  log_comparison <- rbind(log_comparison, data.frame(
    iter = i,
    Raw_model = mean((te$mpg - pred_raw)^2),
    Log_model = mean((te$mpg - pred_log)^2)
  ))
}

cat("Mean MSPE:\n")
cat("  Raw model: ", round(mean(log_comparison$Raw_model), 2), "\n")
cat("  Log model: ", round(mean(log_comparison$Log_model), 2), "\n")
```

Note the `exp()` wrapper on the log model's predictions. Without it, you
would be computing prediction errors on the log scale, which is not
comparable to errors on the original scale.

---

## 10. Simulated Example: Seeing Overfitting Directly

To make overfitting visible, we generate data from a simple true model and
then fit both an appropriate model and an absurdly complex one.

```r
set.seed(211)
n_sim <- 50
x1 <- rnorm(n_sim, 10, 3)
y_sim <- 5 + 2 * x1 + rnorm(n_sim, 0, 4)
sim_df <- data.frame(y = y_sim, x1 = x1)

# Add 10 columns of pure noise
for (j in 1:10) {
  sim_df[[paste0("noise_", j)]] <- rnorm(n_sim)
}

# Cross-validate: true model vs noise-stuffed model
overfit_results <- data.frame()
for (i in 1:30) {
  set.seed(i * 13)
  idx <- sample(1:n_sim, floor(0.75 * n_sim))
  tr <- sim_df[idx, ]
  te <- sim_df[-idx, ]

  m_true  <- lm(y ~ x1, data = tr)
  m_noise <- lm(y ~ ., data = tr)

  overfit_results <- rbind(overfit_results, data.frame(
    iter = i,
    True_model  = mean((te$y - predict(m_true, te))^2),
    Noise_model = mean((te$y - predict(m_noise, te))^2)
  ))
}

cat("Mean MSPE:\n")
cat("  True model (y ~ x1):     ", round(mean(overfit_results$True_model), 2), "\n")
cat("  Noise model (y ~ x1 + junk):", round(mean(overfit_results$Noise_model), 2), "\n")
```

```r
of_long <- tidyr::pivot_longer(overfit_results, -iter,
                                names_to = "Model", values_to = "MSPE")

ggplot(of_long, aes(x = Model, y = MSPE, fill = Model)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("True_model" = "steelblue",
                                "Noise_model" = "firebrick")) +
  labs(title = "Overfitting Demonstrated",
       subtitle = "True model vs. model stuffed with 10 noise predictors",
       x = NULL, y = "MSPE (out-of-sample)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
```

The noise model has a lower in-sample MSE (it always will — more parameters
means a better fit to the training data) but a higher out-of-sample MSPE.
The 10 noise columns contribute nothing real, but the model wastes degrees of
freedom fitting their random fluctuations. This is overfitting made visible,
and cross-validation is what catches it.

---

## 11. Exercises

1. **Log interpretation drill.** Fit `lm(log(mpg) ~ wt + factor(cyl), data = mtcars)`. The baseline for `cyl` is 4-cylinder. (a) If the `wt`
   coefficient is $-0.22$, compute the exact percentage change in mpg per
   1,000 lbs. (b) If the `factor(cyl)8` coefficient is $-0.30$, compute
   the exact multiplier. What fraction of a 4-cylinder car's mpg does an
   8-cylinder car get, at the same weight? (c) Predict the mpg of a 3,500 lb,
   6-cylinder car. Show both the log-scale and back-transformed prediction.

2. **Confidence vs. prediction intervals.** Using `lm(mpg ~ wt, data = mtcars)`, compute both intervals for a car weighing 3,000 lbs.
   Report the widths. Explain in one sentence why the prediction interval
   is wider.

3. **Out-of-sample detection.** Plot `wt` against `disp` for `mtcars`.
   Identify a point that would be within the range of both predictors
   individually but outside the observed joint cloud. Use `hatvalues()`
   to compute what the leverage would be for a new observation at that
   point. Is it above the $2p/n$ threshold?

4. **The cat problem, quantified.** Using `mtcars` (32 observations), run
   a 90/10 cross-validation 50 times (use seeds 1 through 50). Record the
   MSPE for each iteration. Then do the same with a 70/30 split. Plot both
   distributions as boxplots. Which split produces a more stable (lower
   variance) MSPE estimate? Which produces a lower median MSPE? Relate
   your findings to the cat analogy from Section 4.

5. **Model comparison.** Cross-validate (80/20, 20 iterations) three models
   for `mpg`: (a) `wt` only, (b) `wt + hp + factor(am)`,
   (c) `wt + hp + factor(am) + factor(cyl) + disp + drat + qsec`.
   Compute the mean MSPE for each. Does model (c) outperform model (b) out
   of sample, or does it overfit? Produce the boxplot comparison from
   Section 8.1.

6. **Log model CV.** Fit both `lm(mpg ~ wt + hp)` and
   `lm(log(mpg) ~ wt + hp)` and cross-validate each (80/20, 20
   iterations). Remember to back-transform the log model's predictions
   before computing MSPE. Which model predicts better on the original
   scale? Why?

---

## References

- James, G., Witten, D., Hastie, T. & Tibshirani, R. (2021). *An
  Introduction to Statistical Learning*. 2nd ed. Springer. Ch. 5
  (Cross-Validation).
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 4, 7.
- Wooldridge, J.M. (2019). *Introductory Econometrics*. 7th ed. Cengage.
  Ch. 6 (Log models).
