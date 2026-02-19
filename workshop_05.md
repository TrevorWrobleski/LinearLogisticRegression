# Workshop 5: Outliers, Leverage, and Influence

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop

---

## 1. The Problem

Most datasets have unusual points. Some are data entry errors. Some are
genuine observations that happen to be extreme. When we load a new dataset,
we need to ask whether these points are distorting our regression and what might
explain them. A single observation can, under the right conditions, change the
sign of a coefficient, inflate or deflate $R^2$, and reverse substantive conclusions.

---

## 2. Three Concepts

### 2.1 Outliers: unusual in $y$

An **outlier** is a point whose observed response $y_i$ is far from the
model's prediction $\hat{y}_i$ — that is, it has a large residual. But raw
residuals are not directly comparable: a residual of 5 means something
different in a model with $\hat{\sigma} = 2$ than in one with
$\hat{\sigma} = 50$. We therefore **standardize**:

$$
r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
$$

where $e_i = y_i - \hat{y}_i$ is the raw residual, $\hat{\sigma}$ is the
residual standard deviation, and $h_{ii}$ is the leverage (defined below).
Under the model assumptions, standardized residuals are approximately
$N(0, 1)$.

**Threshold:** In this class, we will investigate points where $|r_i| > 3$. 
You can think about this in terms of hte probabilty that a true observation is three standard deviations 
from the mean. In this case, a standardized residual
above 3 corresponds roughly to a 1-in-300 event (under normality).

### 2.2 Leverage: unusual in $x$

**Leverage** measures how far an observation sits from the center of the
predictor space. It is defined as the $i$-th diagonal element of the **hat
matrix**:

$$
H = X(X^TX)^{-1}X^T
$$

The hat matrix earns its name because it "puts the hat on $y$": $\hat{y} = Hy$.
The diagonal entry $h_{ii}$ quantifies how much observation $i$ influences
its own fitted value purely through its position in the predictor space.

Key properties:

- $0 \leq h_{ii} \leq 1$
- $\sum_{i=1}^n h_{ii} = p$, where $p$ is the number of estimated parameters (including the intercept)
- The average leverage is $p/n$

**Threshold:** In this class, we encourage you to investigate points where $h_{ii} > 2p/n$ (twice the average).

By way of analogy, consider a see-saw. A point with high leverage sits far from the
fulcrum. It has the *potential* to tilt the line, but it has not necessarily
done so. Whether it actually tilts the line depends on whether it also
disagrees with the trend. (Give this a try if you find a see-saw to refresh yourself 
on the concept of physics or the playgrounds feels like a distant memory.)

### 2.3 Influence: the combination that matters

A point is **influential** if removing it substantially changes the fitted
regression. Influence requires both leverage (unusual position in $x$) and a
large residual (disagreement with the trend). Neither alone is sufficient.

$$
\text{Influence} \approx \text{Leverage} \times \text{Outlierness}
$$

We quantify influence with two statistics:

**Cook's Distance:**

$$
D_i = \frac{1}{p} \cdot \frac{h_{ii}}{1 - h_{ii}} \cdot r_i^2
$$

This combines leverage ($h_{ii}/(1 - h_{ii})$) with outlierness ($r_i^2$)
into a single number. **Threshold:** investigate points where $D_i > 1$.

**DFFITS:**

$$
\text{DFFITS}_i = r_i^{*} \sqrt{\frac{h_{ii}}{1 - h_{ii}}}
$$

where $r_i^{*}$ is the externally studentized residual (computed using
leave-one-out standard deviation). DFFITS measures how much the fitted value
$\hat{y}_i$ changes when observation $i$ is deleted. **Threshold:**
investigate if $|\text{DFFITS}_i| > 2\sqrt{p/n}$.

### 2.4 Summary

| Statistic | What it measures | Threshold |
|---|---|---|
| Standardized residual $r_i$ | Unusual in $y$ | $\|r_i\| > 3$ |
| Leverage $h_{ii}$ | Unusual in $x$ | $h_{ii} > 2p/n$ |
| Cook's distance $D_i$ | Overall influence | $D_i > 1$ |
| DFFITS | Influence on fitted value | $\|\text{DFFITS}_i\| > 2\sqrt{p/n}$ |

---

## 3. Setup

```r
library(ggplot2)
library(arm)
library(gridExtra)
```

We also need a utility function that computes all four diagnostic statistics
and flags observations exceeding the thresholds. Define this once and use it
throughout:

```r
show_outliers <- function(the.linear.model, topN) {
  n <- length(fitted(the.linear.model))
  p <- length(coef(the.linear.model))

  res.out   <- which(abs(rstandard(the.linear.model)) > 3)
  res.top   <- head(rev(sort(abs(rstandard(the.linear.model)))), topN)

  lev.out   <- which(lm.influence(the.linear.model)$hat > 2 * p / n)
  lev.top   <- head(rev(sort(lm.influence(the.linear.model)$hat)), topN)

  dffits.out <- which(dffits(the.linear.model) > 2 * sqrt(p / n))
  dffits.top <- head(rev(sort(dffits(the.linear.model))), topN)

  cooks.out  <- which(cooks.distance(the.linear.model) > 1)
  cooks.top  <- head(rev(sort(cooks.distance(the.linear.model))), topN)

  list(Std.res = res.out, Std.res.top = res.top,
       Leverage = lev.out, Leverage.top = lev.top,
       DFFITS = dffits.out, DFFITS.top = dffits.top,
       Cooks = cooks.out, Cooks.top = cooks.top)
}
```

The function returns a list. `$Std.res` gives the row indices exceeding the
threshold; `$Std.res.top` gives the top $N$ values sorted in descending
order. The same structure applies to leverage, DFFITS, and Cook's distance.
To find observations flagged by *multiple* criteria, use `intersect()`:

```r
# Rows flagged by BOTH leverage and DFFITS
intersect(result$Leverage, result$DFFITS)
```

---

## 4. Example 1: `mtcars` — Natural Outliers in Real Data

### 4.1 The regression

```r
data(mtcars)

mpg.lm <- lm(mpg ~ wt, data = mtcars)
display(mpg.lm)
```

Each additional 1,000 lbs of weight costs approximately 5.3 mpg. We want to
know whether any individual car is disproportionately driving this estimate.

### 4.2 Visual inspection with labels

```r
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, colour = "steelblue", linewidth = 0.8) +
  geom_text(aes(label = rownames(mtcars)), size = 2.2,
            nudge_y = 0.8, check_overlap = TRUE) +
  labs(title = "Weight vs. MPG with Car Names",
       subtitle = "Which cars might be pulling the line?",
       x = "Weight (1000 lbs)", y = "Miles per Gallon") +
  theme_minimal(base_size = 13)
```

Before running any diagnostics, look at the plot. The Lincoln Continental and
Chrysler Imperial are the heaviest cars — they sit far to the right (high
leverage). The Toyota Corolla and Fiat 128 are above the line at low weight
(large positive residuals). Are any of these *influential*?

### 4.3 Formal diagnostics

```r
mpg.stats <- show_outliers(mpg.lm, 5)

mpg.stats$Leverage.top    # cars far from average weight
mpg.stats$Std.res.top     # cars far from the regression line
mpg.stats$Cooks.top       # cars that actually move the line
```

### 4.4 Removing the heaviest car and comparing

```r
heaviest <- which.max(mtcars$wt)
cat("Heaviest car:", rownames(mtcars)[heaviest], "\n")

mpg.lm.clean <- lm(mpg ~ wt, data = mtcars[-heaviest, ])

p_base <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue") +
  geom_point(data = mtcars[heaviest, ], colour = "firebrick",
             size = 4, shape = 17) +
  geom_smooth(method = "lm", se = FALSE, colour = "steelblue", linewidth = 0.8) +
  labs(x = "Weight (1000 lbs)", y = "MPG",
       title = "Solid = all data; Dashed = without heaviest car")

p_clean <- ggplot() +
  geom_smooth(data = mtcars[-heaviest, ], aes(x = wt, y = mpg),
              method = "lm", se = FALSE, linetype = "dashed",
              colour = "firebrick", linewidth = 0.8)

p_base + p_clean$layers[] + theme_minimal(base_size = 13)
```

If the solid and dashed lines are nearly identical, the heaviest car has high
leverage but is not influential — it sits close to the regression line and
reinforces the trend rather than fighting it. If the lines differ noticeably,
the point is influential and warrants further investigation.

### 4.5 Manufacturing an influential point

To see what genuine influence looks like, we add a fictitious observation: a
vehicle weighing 6,000 lbs that gets 35 mpg. This point has extreme leverage
(far right) *and* a large residual (far above the line).

```r
mtcars.fake <- rbind(mtcars, data.frame(
  mpg = 35, cyl = 4, disp = 100, hp = 200, drat = 3.5,
  wt = 6.0, qsec = 15, vs = 0, am = 1, gear = 4, carb = 2,
  row.names = "Fake Electric Truck"))

mpg.lm.fake <- lm(mpg ~ wt, data = mtcars.fake)

p_fake <- ggplot(mtcars.fake, aes(x = wt, y = mpg)) +
  geom_point(size = 2.5, colour = "steelblue") +
  geom_point(data = mtcars.fake["Fake Electric Truck", ],
             colour = "firebrick", size = 5, shape = 18) +
  geom_smooth(method = "lm", se = FALSE, colour = "steelblue", linewidth = 0.8) +
  labs(title = "One Influential Point Flattens the Entire Line",
       x = "Weight (1000 lbs)", y = "MPG") +
  theme_minimal(base_size = 13)

p_orig <- ggplot() +
  geom_smooth(data = mtcars, aes(x = wt, y = mpg),
              method = "lm", se = FALSE, linetype = "dashed",
              colour = "grey50", linewidth = 0.8)

p_fake + p_orig$layers[]
```

```r
cat("Original slope:", round(coef(mpg.lm)[2], 3), "\n")
cat("Slope with fake point:", round(coef(mpg.lm.fake)[2], 3), "\n")
```

A single observation changed the slope substantially. That is influence: high
leverage combined with a large residual.

---

## 5. Example 2: Anscombe's Quartet

Anscombe's quartet is a set of four datasets that have nearly identical
summary statistics — same means, same variances, same correlation, same
regression slope ($\approx 0.5$), same $R^2$ ($\approx 0.67$) — but
completely different structures. It is the definitive argument for always
plotting your data.

```r
data(anscombe)

a1 <- data.frame(x = anscombe$x1, y = anscombe$y1, Set = "I: Well-behaved")
a2 <- data.frame(x = anscombe$x2, y = anscombe$y2, Set = "II: Curved")
a3 <- data.frame(x = anscombe$x3, y = anscombe$y3, Set = "III: Y-outlier")
a4 <- data.frame(x = anscombe$x4, y = anscombe$y4, Set = "IV: X-leverage")
ansc <- rbind(a1, a2, a3, a4)

ggplot(ansc, aes(x = x, y = y)) +
  geom_point(size = 2.5, colour = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick", linewidth = 0.8) +
  facet_wrap(~ Set, scales = "free") +
  labs(title = "Anscombe's Quartet: Same Regression, Different Stories",
       subtitle = "All four: slope ≈ 0.5, R² ≈ 0.67") +
  theme_minimal(base_size = 12)
```

Read each panel through the lens of today's concepts:

- **Set I:** well-behaved scatter. No outlier or leverage issues. The
  regression is appropriate.
- **Set II:** the relationship is curved. Is this an outlier problem? No, it
  is a model specification problem (non-linearity). The diagnostics from
  Workshop 3 (residuals vs. fitted) would catch this.
- **Set III:** one point has a large residual (far above the line), but it sits
  in the middle of the $x$-range. It has low leverage. Is it influential?
- **Set IV:** one point sits far to the right ($x = 19$). It has extreme
  leverage. Without it, there is no linear relationship.

### 5.1 Proving it with numbers

```r
# Set III: remove the y-outlier
lm3_full <- lm(y3 ~ x3, data = anscombe)
lm3_no   <- lm(y3 ~ x3, data = anscombe[-which.max(anscombe$y3), ])

cat("Set III slope with outlier:   ", round(coef(lm3_full)[2], 3), "\n")
cat("Set III slope without outlier:", round(coef(lm3_no)[2], 3), "\n")

# Set IV: remove the leverage point
lm4_full <- lm(y4 ~ x4, data = anscombe)
lm4_no   <- lm(y4 ~ x4, data = anscombe[-which.max(anscombe$x4), ])

cat("\nSet IV slope with leverage point:   ", round(coef(lm4_full)[2], 3), "\n")
cat("Set IV slope without leverage point:", round(coef(lm4_no)[2], 3), "\n")
```

In Set III, removing the outlier changes the slope modestly — the point has a
large residual but low leverage, so its influence is limited. In Set IV,
removing the leverage point changes the slope dramatically — that single
observation was the entire source of the apparent linear relationship.

```r
cat("\nSet III max Cook's D:", round(max(cooks.distance(lm3_full)), 3), "\n")
cat("Set IV  max Cook's D:", round(max(cooks.distance(lm4_full)), 3), "\n")
```

Cook's distance confirms the visual impression: the leverage point in Set IV
has far more influence than the $y$-outlier in Set III.

---

## 6. Example 3: The Four Scenarios

Here's another example to help you visualize these concepts.

```r
set.seed(211)
base <- data.frame(x = rnorm(40, 10, 2))
base$y <- 3 + 0.7 * base$x + rnorm(40, 0, 1.5)

make_scenario <- function(base, new_x, new_y, title, subtitle) {
  df <- rbind(base, data.frame(x = new_x, y = new_y))
  df$type <- c(rep("Original", nrow(base)), "Added")
  n <- nrow(df)

  lm_full <- lm(y ~ x, data = df)
  lm_orig <- lm(y ~ x, data = base)

  d_slope <- round(coef(lm_full)[2] - coef(lm_orig)[2], 3)
  h_val   <- round(hatvalues(lm_full)[n], 3)
  cd_val  <- round(cooks.distance(lm_full)[n], 3)

  p <- ggplot(df, aes(x = x, y = y)) +
    geom_point(aes(colour = type, size = type)) +
    geom_smooth(method = "lm", se = FALSE, colour = "steelblue", linewidth = 0.8) +
    scale_colour_manual(values = c(Added = "firebrick", Original = "grey50")) +
    scale_size_manual(values = c(Added = 4, Original = 2)) +
    labs(title = title,
         subtitle = paste0(subtitle, "\n\u0394slope = ", d_slope,
                           "  h = ", h_val, "  Cook's D = ", cd_val)) +
    theme_minimal(base_size = 11) + theme(legend.position = "none")

  p_orig <- ggplot() +
    geom_smooth(data = base, aes(x = x, y = y),
                method = "lm", se = FALSE, linetype = "dashed",
                colour = "grey70", linewidth = 0.7)
  p + p_orig$layers[]
}

xbar <- mean(base$x)
yhat_at <- function(x) 3 + 0.7 * x

pA <- make_scenario(base, xbar, yhat_at(xbar),
        "A: Low leverage, small residual",
        "Near center, on the line. Harmless.")

pB <- make_scenario(base, xbar, yhat_at(xbar) + 12,
        "B: Low leverage, LARGE residual",
        "Outlier in y, but near center of x.")

pC <- make_scenario(base, 22, yhat_at(22),
        "C: HIGH leverage, small residual",
        "Far right, but sits on the line.")

pD <- make_scenario(base, 22, yhat_at(22) - 15,
        "D: HIGH leverage, LARGE residual",
        "Far right AND far from line. INFLUENTIAL.")

grid.arrange(pA, pB, pC, pD, nrow = 2)
```

Read the four panels:

- **A (low leverage, small residual):** The added point is unremarkable.
  It sits near the center of $x$ and on the regression line. The slope,
  leverage, and Cook's distance are all negligible.
- **B (low leverage, large residual):** The point has a huge residual — it
  is far above the line. But it sits near the center of $x$, so its leverage
  is low. It inflates $\hat{\sigma}$ but barely rotates the line. This is an
  outlier that is *not* influential.
- **C (high leverage, small residual):** The point is far to the right
  (high leverage) but falls on the regression line. It *reinforces* the
  existing trend rather than fighting it. High leverage alone does not
  equal influence.
- **D (high leverage, large residual):** The point is far to the right *and*
  disagrees with the trend. The solid line rotates visibly toward it. This is
  influence: the line must compromise to accommodate a point that has both
  positional authority (leverage) and disagreement (large residual).

**influence = leverage $\times$ outlierness.**

---

## 7. Example 4: Building Your Own Influential Point

The `cars` dataset records speed and stopping distance for 50 cars from the
1920s. Try to add one point and watch what happens.

```r
data(cars)
cars.lm <- lm(dist ~ speed, data = cars)
display(cars.lm)
```

Now add a car traveling at 50 mph that stops in 10 feet — physically
implausible (but let's say LSE orders us a McMurtry Spéirling - okay, 
that's still not capable of withstanding the G-forces we're talking here, 
but let's consider it becasue it's instructive):

```r
cars.mod <- rbind(cars, data.frame(speed = 50, dist = 10))
cars.lm.mod <- lm(dist ~ speed, data = cars.mod)

p_mod <- ggplot(cars.mod, aes(x = speed, y = dist)) +
  geom_point(size = 2, colour = "steelblue") +
  geom_point(data = data.frame(speed = 50, dist = 10),
             colour = "firebrick", size = 5, shape = 18) +
  geom_smooth(method = "lm", se = FALSE, colour = "steelblue", linewidth = 0.8) +
  labs(title = "With Fake Point at (50, 10)",
       x = "Speed (mph)", y = "Stopping Distance (ft)") +
  theme_minimal(base_size = 13)

p_orig <- ggplot() +
  geom_smooth(data = cars, aes(x = speed, y = dist),
              method = "lm", se = FALSE, linetype = "dashed",
              colour = "grey50", linewidth = 0.8)

p_mod + p_orig$layers[]

cat("Original slope:", round(coef(cars.lm)[2], 2), "\n")
cat("Modified slope:", round(coef(cars.lm.mod)[2], 2), "\n")
cat("Cook's D for fake point:",
    round(cooks.distance(cars.lm.mod)[nrow(cars.mod)], 3), "\n")
```

The slope drops substantially, and Cook's distance for the added point is
well above the threshold. One implausible observation reshaped the
relationship between speed and stopping distance.

---

## 8. What to Do When You Find an Influential Point


**Step 1. Is it a mistake?**

Check for typos, impossible values (negative ages, grades above the maximum),
or recording errors. If it is clearly erroneous, remove it and document why.

**Step 2. Is it influential?**

Fit the model with and without the point. Compare coefficients. If they
barely change, the point is unusual but not distorting your results — keep it
and move on.

**Step 3. If it is influential, present both models.**

Report the regression with all observations and the regression without the
influential point(s). Discuss which is more appropriate for your research
question. This is transparent and lets the reader evaluate the sensitivity
of your conclusions.

**Step 4. If there are many flagged points, look for patterns.**

Do they share a common predictor value? Do they belong to a distinguishable
subgroup? If so, consider whether a separate model for that subgroup, or an
additional predictor that accounts for the subgroup, would be more
appropriate than deletion.

**The non-negotiable rule:** never remove observations solely to improve
$R^2$ or make diagnostic plots look cleaner. That is data manipulation.

### 8.1 A note on large datasets

With large $n$, the thresholds become very tight. For example, with
$n = 10{,}000$ and $p = 7$:

$$
\frac{2p}{n} = 0.0014 \quad \text{(leverage)}, \qquad 2\sqrt{\frac{p}{n}} = 0.053 \quad \text{(DFFITS)}
$$

Hundreds of observations will be flagged, and investigating each one
individually is impractical. The strategy shifts: instead of examining points
one at a time, look for *common characteristics* among the flagged
observations. Do they share a demographic? A time period? An unusual
combination of predictors? The flagged set often reveals a subpopulation that
the model handles poorly — which is more useful information than any single
outlier.

---

## 9. Visualizing All Four Diagnostics for a Single Model

It is useful to produce a single panel that displays leverage, standardized
residuals, and Cook's distance simultaneously, so you can see how the three
statistics interact for every observation.

```r
data(mtcars)
fit <- lm(mpg ~ wt + hp, data = mtcars)

diag_df <- data.frame(
  car      = rownames(mtcars),
  fitted   = fitted(fit),
  resid    = residuals(fit),
  std_res  = rstandard(fit),
  leverage = hatvalues(fit),
  cooks    = cooks.distance(fit)
)

n <- nrow(mtcars)
p <- length(coef(fit))
lev_thresh <- 2 * p / n

# Leverage vs. standardized residual, sized by Cook's distance
ggplot(diag_df, aes(x = leverage, y = std_res)) +
  geom_point(aes(size = cooks), colour = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = c(-3, 3), linetype = "dashed", colour = "firebrick") +
  geom_vline(xintercept = lev_thresh, linetype = "dashed", colour = "firebrick") +
  geom_text(data = subset(diag_df, leverage > lev_thresh | abs(std_res) > 2),
            aes(label = car), size = 2.5, nudge_y = 0.3, check_overlap = TRUE) +
  scale_size_continuous(range = c(1, 8), name = "Cook's D") +
  labs(title = "Leverage vs. Standardized Residual",
       subtitle = "Point size = Cook's distance. Dashed lines = thresholds.",
       x = "Leverage", y = "Standardized Residual") +
  theme_minimal(base_size = 13)
```

Points in the upper-right or lower-right quadrant (high leverage *and* large
standardized residual) are the ones to worry about. Large points (high Cook's
distance) in those quadrants are actively distorting your regression. Points
with high leverage but small residuals (right side, near $y = 0$) are
reinforcing the trend and are generally benign.

---

## 10. Exercises

1. **Diagnostics on `mtcars`.** Fit `lm(mpg ~ wt + hp, data = mtcars)`.
   Run `show_outliers()` with `topN = 5`. Which cars are flagged by
   leverage? Which by standardized residuals? Which by Cook's distance?
   Find the intersection of leverage and DFFITS using `intersect()`.
   Produce the leverage-vs-residual plot from Section 9 and confirm that
   your intersection points appear in the dangerous quadrant.

2. **Influence by removal.** Using the model from Exercise 1, identify the
   car with the highest Cook's distance. Remove it and refit. Compute the
   percentage change in the `wt` coefficient:
   $|\hat{\beta}\_{\text{new}} - \hat{\beta}\_{\text{old}}| / |\hat{\beta}\_{\text{old}}| \times 100$.
   Is the change greater than 5%? Plot both regression lines (solid = full
   data, dashed = without the car) and confirm visually.

3. **Anscombe Set IV.** Fit `lm(y4 ~ x4, data = anscombe)`. Compute
   `hatvalues()` for every observation. Which observation has the highest
   leverage? Compute its Cook's distance. Now remove it and refit. What
   happens to $R^2$? What happens to the slope? Write a sentence explaining
   why this single point controlled the entire regression.

4. **Create your own.** Using the `cars` dataset, add a point at
   `(speed = 45, dist = 200)`. Before fitting, predict whether this point
   will have (a) high or low leverage, and (b) a positive or negative
   residual. Fit the model, compute Cook's distance for the added point,
   and verify your predictions. Now try `(speed = 15, dist = 200)` instead.
   Which of the two fake points is more influential, and why?

5. **The four scenarios.** Reproduce the four-panel simulation from Section 6
   but change the added point in Panel D to `(22, yhat_at(22) + 15)` — same
   leverage, but now the residual is positive instead of negative. Does the
   line tilt upward instead of downward? Does Cook's distance change
   substantially? Explain.

6. **Large-sample behavior.** Generate a clean dataset: `set.seed(42);
   n <- 500; x <- rnorm(n, 50, 10); y <- 10 + 2*x + rnorm(n, 0, 8)`.
   Fit the regression and run `show_outliers()`. How many observations
   are flagged by leverage? By DFFITS? Remove all DFFITS-flagged points
   and refit. Does the slope change meaningfully? What does this tell you
   about outlier diagnostics in large samples?

---

## References

- Belsley, D.A., Kuh, E. & Welsch, R.E. (1980). *Regression Diagnostics*.
  Wiley.
- Cook, R.D. (1977). Detection of influential observation in linear
  regression. *Technometrics*, 19(1), 15–18.
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*.
  3rd ed. Sage. Ch. 11.
- Anscombe, F.J. (1973). Graphs in statistical analysis. *The American
  Statistician*, 27(1), 17–21.
