# Workshop 11: Introduction to Poisson Regression

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop

---

## 1. Completing the Trilogy

We have now built two members of the Generalized Linear Model (GLM) family:
linear regression for continuous outcomes and logistic regression for binary
outcomes. Our final addition to the GLM family (in this course) is Poisson regression/
It handles **count data** — non-negative integers like the number of
accidents, website visits, insurance claims, or goals scored.

| | **Linear** | **Logistic** | **Poisson** |
|---|---|---|---|
| Outcome | Continuous | Binary (0/1) | Count (0, 1, 2, ...) |
| Link function | Identity ($y$) | Logit ($\log\frac{p}{1-p}$) | Log ($\log \theta$) |
| Interpret coeff. via | Directly | Exponentiate → odds ratio | Exponentiate → rate ratio |
| R function | `lm()` | `glm(..., binomial)` | `glm(..., poisson)` |

The structure is similar in all three cases, with a linear predictor
$\beta_0 + \beta_1 X_1 + \cdots$ on a transformed scale, with
exponentiation to get back to an interpretable scale (for logistic and
Poisson). 

---

## 2. Why Not Just Use Linear Regression on Counts?

Counts have some properties that make linear regression inappropriate:

1. **Counts are non-negative.** You cannot have $-3$ accidents. Linear
   regression can predict negative values.
2. **Counts are discrete integers.** You cannot have 2.7 accidents. Linear
   regression predicts continuous values.
3. **The variance typically grows with the mean.** A neighborhood averaging
   50 accidents per year will have more variability than one averaging 5.
   Linear regression assumes constant variance.

We demonstrate the problem with `mtcars`. The variable `carb` records the
number of carburetors (1, 2, 3, 4, 6, or 8) — a count. (A carberator is a 
device that's really fun to take apart. It's dual function is to 1) enrich a 
weekend when you just want to ride your motorcycle
and 2) mix air and liquid fuel into a combustible fine mist.)

```r
library(ggplot2)
library(arm)
library(gridExtra)

data(mtcars)

# Linear regression on a count outcome
bad_model <- lm(carb ~ hp, data = mtcars)

ggplot(mtcars, aes(x = hp, y = carb)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, colour = "firebrick",
              linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  labs(title = "Linear Regression on a Count Outcome",
       subtitle = "The line crosses below zero for low-hp cars — impossible for a count",
       x = "Horsepower", y = "Number of Carburetors") +
  theme_minimal(base_size = 13)
```

```r
predict(bad_model, data.frame(hp = c(50, 60)))
```

For low-horsepower cars, the model predicts fewer than zero carburetors. While cars today use Electronic Fuel Injection
this dataset is pre-1980s. The outcome doesn't make sense since the model does not know the outcome is bounded.

---

## 3. The Poisson Regression Model

### 3.1 The model

Poisson regression models the log of the expected count as a linear function
of the predictors:

$$
\log(\theta_i) = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \cdots + \beta_k X_{ki}
$$

where $\theta_i = E[Y_i \mid X_i]$ is the expected count for observation
$i$. Exponentiating both sides:

$$
\theta_i = e^{\beta_0 + \beta_1 X_{1i} + \cdots + \beta_k X_{ki}}
$$

Because the exponential function is always positive, predicted counts cannot be negative.

### 3.2 Interpreting coefficients: rate ratios

Consider two observations differing by one unit in $X_1$:

$$
\log(\theta_{\text{new}}) - \log(\theta_{\text{old}}) = \beta_1
$$

$$
\log\!\left(\frac{\theta_{\text{new}}}{\theta_{\text{old}}}\right) = \beta_1
$$

$$
\frac{\theta_{\text{new}}}{\theta_{\text{old}}} = e^{\beta_1}
$$

A one-unit increase in $X_1$ **multiplies** the expected count by
$e^{\beta_1}$. This is called the **rate ratio**. The percentage change in
the expected count is:

$$
\%\Delta = (e^{\hat{\beta}_1} - 1) \times 100
$$

If $\hat{\beta}_1 = 0.14$, then $e^{0.14} = 1.150$, and the expected count
increases by 15.0%. If $\hat{\beta}_1 = -0.30$, then $e^{-0.30} = 0.741$,
and the expected count decreases by 25.9%.

This is algebraically identical to the logistic case, except the "odds" becomes "expected count." 

### 3.3 The parallel interpretation table

| | **Linear** | **Logistic** | **Poisson** |
|---|---|---|---|
| $\hat{\beta}$ means... | $Y$ changes by $\hat{\beta}$ | Log-odds change by $\hat{\beta}$ | $\log(\text{count})$ changes by $\hat{\beta}$ |
| To interpret, we... | Read directly | Exponentiate → odds ratio | Exponentiate → rate ratio |
| $e^{\hat{\beta}} = 1.15$ means... | (not applicable) | Odds $\times$ 1.15 (+15%) | Count $\times$ 1.15 (+15%) |
| $e^{\hat{\beta}} = 0.80$ means... | (not applicable) | Odds $\times$ 0.80 (−20%) | Count $\times$ 0.80 (−20%) |


---

## 4. First Example: Carburetors in `mtcars`

### 4.1 Fitting the model

```r
carb_glm <- glm(carb ~ hp + wt, data = mtcars, family = poisson)
display(carb_glm)
```

### 4.2 Interpreting the output

```r
b <- coef(carb_glm)
cat("Coefficients:\n")
cat("  Intercept:", round(b[1], 4), "\n")
cat("  hp:       ", round(b[2], 4), "\n")
cat("  wt:       ", round(b[3], 4), "\n\n")

cat("Rate ratios:\n")
cat("  hp (per unit): ", round(exp(b["hp"]), 4), "\n")
cat("  hp (per 100 units):", round(exp(100 * b["hp"]), 3), "\n")
cat("  wt (per 1000 lbs): ", round(exp(b["wt"]), 3), "\n\n")

cat("Percent changes:\n")
cat("  hp (per 100 units):", round((exp(100 * b["hp"]) - 1) * 100, 1), "%\n")
cat("  wt (per 1000 lbs): ", round((exp(b["wt"]) - 1) * 100, 1), "%\n")
```

The per-unit horsepower coefficient will be small — perhaps 0.008. That
looks like nothing. But over a realistic 100 hp range:
$e^{100 \times 0.008} = e^{0.8} = 2.23$ — the expected carburetor count
more than doubles. Small coefficients can be deceiving when the predictor
has a wide range. This is the exact same issue we encountered in logistic
regression where $\hat{\beta}_{\text{Age}} = -0.04$ translated to a 33%
odds reduction over 10 years.

### 4.3 Comparing the linear and Poisson fits

```r
p_lin <- ggplot(mtcars, aes(x = hp, y = carb)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick",
              linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  labs(title = "Linear (can go negative)", x = "Horsepower", y = "Carburetors") +
  theme_minimal(base_size = 11)

p_pois <- ggplot(mtcars, aes(x = hp, y = carb)) +
  geom_point(size = 2.5, colour = "steelblue", alpha = 0.7) +
  geom_smooth(method = "glm", method.args = list(family = "poisson"),
              se = FALSE, colour = "firebrick", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  labs(title = "Poisson (always positive)", x = "Horsepower", y = "Carburetors") +
  theme_minimal(base_size = 11)

grid.arrange(p_lin, p_pois, ncol = 2)
```

The Poisson curve stays above zero by construction. For high-horsepower cars,
both models produce similar predictions. For low-horsepower cars, the linear
model predicts negative carburetors while the Poisson model levels off above
zero.

---

## 5. Second Example: Simulated Accident Data

We simulate data inspired by the California young-driver accident study. This
lets us see the full workflow — exploratory plots, model fitting,
interpretation, and prediction — on data that runs without downloads.

### 5.1 Generating the data

```r
set.seed(211)

ages <- 16:26
genders <- c("Male", "Female")
months <- c("Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec")
days <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")

acc <- expand.grid(Age = ages, Gender = genders,
                   Month = factor(months, levels = months),
                   DayOfWeek = factor(days, levels = days))

# True model: young drivers, males, summer, weekends = more accidents
log_mu <- 4.5 - 0.15 * (acc$Age - 16) + 0.005 * (acc$Age - 16)^2 +
  0.7 * (acc$Gender == "Male") +
  ifelse(acc$Month %in% c("Jun","Jul","Aug"), 0.20,
  ifelse(acc$Month %in% c("Dec","Jan","Feb"), -0.10, 0)) +
  ifelse(acc$DayOfWeek %in% c("Fri","Sat"), 0.15,
  ifelse(acc$DayOfWeek == "Sun", 0.05, 0))

acc$Freq <- rpois(nrow(acc), lambda = exp(log_mu))
```

### 5.2 Exploratory plots

```r
p_age <- ggplot(acc, aes(x = factor(Age), y = Freq)) +
  geom_boxplot(fill = "steelblue", alpha = 0.4) +
  labs(title = "Accidents by Age", x = "Age", y = "Frequency") +
  theme_minimal(base_size = 11)

p_gen <- ggplot(acc, aes(x = Gender, y = Freq, fill = Gender)) +
  geom_boxplot(alpha = 0.5, show.legend = FALSE) +
  scale_fill_manual(values = c("Female" = "steelblue", "Male" = "firebrick")) +
  labs(title = "Accidents by Gender", x = NULL, y = "Frequency") +
  theme_minimal(base_size = 11)

p_mon <- ggplot(acc, aes(x = Month, y = Freq)) +
  geom_boxplot(fill = "steelblue", alpha = 0.4) +
  labs(title = "Accidents by Month", x = NULL, y = "Frequency") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_day <- ggplot(acc, aes(x = DayOfWeek, y = Freq)) +
  geom_boxplot(fill = "steelblue", alpha = 0.4) +
  labs(title = "Accidents by Day of Week", x = NULL, y = "Frequency") +
  theme_minimal(base_size = 11)

grid.arrange(p_age, p_gen, p_mon, p_day, nrow = 2)
```

Read each panel before fitting anything:

- **Age:** frequency is highest for the youngest drivers (16–18), drops
  sharply, then levels off. This is non-linear — a polynomial term is
  warranted.
- **Gender:** males have higher accident counts than females. Clear
  separation.
- **Month:** summer months (June, July, August) are higher. Winter months
  are lower.
- **Day of week:** Friday and Saturday stand out above the rest.

These patterns make substantive sense: 16-year-olds are new drivers with the
highest risk, males take more risks on the road, summer means more driving
(especially for young people on school break), and Friday/Saturday nights are
when young drivers are most likely to be out.

### 5.3 Fitting the model

Because the age effect is non-linear, we include a polynomial term using
`poly(Age, 2)`, which adds both a linear and quadratic component:

```r
acc_glm <- glm(Freq ~ poly(Age, 2) + Gender + Month + DayOfWeek,
               data = acc, family = poisson)
display(acc_glm)
```

The output has many coefficients — one for each level of Month (relative to
January) and each level of DayOfWeek (relative to Monday). 
We still need to exponentiate the coefficient to get
the multiplicative change in expected count.

### 5.4 Interpreting the key coefficients

```r
b <- coef(acc_glm)

# Gender effect
cat("Gender (Male vs Female):\n")
cat("  Coefficient:  ", round(b["GenderMale"], 4), "\n")
cat("  Rate ratio:   ", round(exp(b["GenderMale"]), 3), "\n")
cat("  % change:     ", round((exp(b["GenderMale"]) - 1) * 100, 1), "%\n\n")

# Month effects: find the largest
month_coefs <- b[grep("^Month", names(b))]
cat("Month with most accidents (relative to Jan):\n")
cat(" ", names(which.max(month_coefs)), "  coef =",
    round(max(month_coefs), 4), "\n")
cat("  Rate ratio:", round(exp(max(month_coefs)), 3), "\n")
cat("  % change:  ", round((exp(max(month_coefs)) - 1) * 100, 1), "%\n\n")

# Day of week effects
day_coefs <- b[grep("^DayOfWeek", names(b))]
cat("Day with most accidents (relative to Mon):\n")
cat(" ", names(which.max(day_coefs)), "  coef =",
    round(max(day_coefs), 4), "\n")
cat("  Rate ratio:", round(exp(max(day_coefs)), 3), "\n")
cat("  % change:  ", round((exp(max(day_coefs)) - 1) * 100, 1), "%\n")
```

Let's walk through one interpretation in full. If the July coefficient is 0.20:

$$
e^{0.20} = 1.221
$$

$$
\%\Delta = (1.221 - 1) \times 100 = 22.1\%
$$

The expected number of accidents in July is 22% higher than in January,
holding age, gender, and day of week constant.

For small coefficients, the approximation is
close, but for larger ones it diverges. Always exponentiate - it's good practice even if the values are similar.

### 5.5 The multiplicative structure

Because the model is:

$$
\theta = e^{\beta_0} \times e^{\beta_1 X_1} \times e^{\beta_2 X_2} \times \cdots
$$

each predictor contributes a multiplicative factor to the expected count. The
male/female ratio is $e^{\hat{\beta}_{\text{Male}}}$ at every age, every
month, and every day of the week (assuming no interactions). This is
analogous to how the odds ratio in logistic regression was constant across
predictor values.

```r
cat("Male-to-female accident ratio:", round(exp(b["GenderMale"]), 3), "\n")
cat("Is this > 2?", exp(b["GenderMale"]) > 2, "\n")
```

If you wanted the ratio to vary by age — perhaps young men are especially
risky — you would need an interaction term: `Gender * poly(Age, 2)`.

---

## 6. Prediction

Just as in logistic regression, `predict()` with `type = "response"` returns
predictions on the original scale (expected counts, not log-counts).

```r
new_dat <- data.frame(
  Age = c(17, 17, 25, 25),
  Gender = c("Male", "Female", "Male", "Female"),
  Month = factor("Jul", levels = levels(acc$Month)),
  DayOfWeek = factor("Sat", levels = levels(acc$DayOfWeek))
)

new_dat$predicted <- predict(acc_glm, newdata = new_dat, type = "response")

cat("Predicted accident counts (July, Saturday):\n")
print(new_dat[, c("Age", "Gender", "predicted")])
```

```r
cat("\nMale/Female ratio at age 17:",
    round(new_dat$predicted[1] / new_dat$predicted[2], 2), "\n")
cat("Male/Female ratio at age 25:",
    round(new_dat$predicted[3] / new_dat$predicted[4], 2), "\n")
```

Both ratios should be approximately equal to $e^{\hat{\beta}_{\text{Male}}}$
— confirming the multiplicative structure. The ratios will not be *exactly*
identical due to the polynomial age term, but they will be close because the
Gender coefficient enters additively on the log scale.

Now change the scenario — Monday in January:

```r
new_dat_2 <- data.frame(
  Age = c(17, 17, 25, 25),
  Gender = c("Male", "Female", "Male", "Female"),
  Month = factor("Jan", levels = levels(acc$Month)),
  DayOfWeek = factor("Mon", levels = levels(acc$DayOfWeek))
)

new_dat_2$predicted <- predict(acc_glm, newdata = new_dat_2, type = "response")

cat("Predicted accident counts (January, Monday):\n")
print(new_dat_2[, c("Age", "Gender", "predicted")])

cat("\nJuly-Sat / Jan-Mon ratio for 17M:",
    round(new_dat$predicted[1] / new_dat_2$predicted[1], 2), "\n")
```

The ratio between July-Saturday and January-Monday predictions tells you the
combined multiplicative effect of the month and day-of-week factors. This is
the kind of practical calculation that matters: how much more dangerous is a
summer weekend night compared to a winter weekday for a 17-year-old male?

---

## 7. Deviance

Deviance in Poisson regression works as it did in logistic
regression. The null deviance measures the fit of a model with no predictors.
The residual deviance measures the fit of your model. The difference — the
deviance reduction — quantifies how much the predictors improve the fit.

```r
null_dev  <- acc_glm$null.deviance
resid_dev <- acc_glm$deviance
dev_drop  <- null_dev - resid_dev
k <- length(coef(acc_glm)) - 1

cat("Null deviance:    ", round(null_dev, 1), "\n")
cat("Residual deviance:", round(resid_dev, 1), "\n")
cat("Deviance drop:    ", round(dev_drop, 1), "\n")
cat("df:               ", k, "\n")
cat("Chi-sq critical:  ", round(qchisq(0.95, k), 2), "\n")
cat("Model significant:", dev_drop > qchisq(0.95, k), "\n")
```

---

## 8. Offsets: Modeling Rates Instead of Counts

Sometimes raw counts are misleading because the **exposure** differs across
observations. If City A has 500 accidents and City B has 100, City A looks
more dangerous — but if City A has 10 times more drivers, its *rate* is
actually lower.

An **offset** adjusts for this. Instead of modeling the count directly, we
model the rate — counts per unit of exposure:

$$
\log(\theta_i) = \log(u_i) + \beta_0 + \beta_1 X_1 + \cdots
$$

where $u_i$ is the exposure (number of drivers, hours observed, population).
The $\log(u_i)$ term enters the model with a fixed coefficient of 1.

In R:

```r
# Syntax (not run — requires an exposure variable):
# glm(Freq ~ X1 + X2 + offset(log(exposure)), family = poisson)
```

We demonstrate with a simple simulated example:

```r
set.seed(211)
cities <- data.frame(
  accidents = c(rpois(20, lambda = 50), rpois(20, lambda = 150)),
  population = c(rep(10000, 20), rep(50000, 20)),
  has_camera = rep(c(0, 1), each = 20)
)

# Without offset: ignores population differences
mod_no_offset <- glm(accidents ~ has_camera, data = cities, family = poisson)

# With offset: models the rate (accidents per capita)
mod_offset <- glm(accidents ~ has_camera + offset(log(population)),
                  data = cities, family = poisson)

cat("Without offset — camera coefficient:", round(coef(mod_no_offset)["has_camera"], 3), "\n")
cat("With offset    — camera coefficient:", round(coef(mod_offset)["has_camera"], 3), "\n")
```

The model without the offset confounds the camera effect with the population
difference. The offset model isolates the camera effect by adjusting for the
fact that larger cities have more accidents simply because they have more
people.

---

## 9. Overdispersion

The Poisson distribution has an unusual property: **the mean equals the
variance.** If the expected count is 10, the variance is also 10. If the variance exceeds the
mean, this is called **overdispersion.**

### 9.1 Detection

If the residual deviance is much larger than the residual
degrees of freedom, suspect overdispersion. The ratio (residual deviance /
residual df) should be near 1 for well-behaved Poisson data. A ratio of 2
or more suggests overdispersion.

```r
cat("Residual deviance:", round(acc_glm$deviance, 1), "\n")
cat("Residual df:      ", acc_glm$df.residual, "\n")
cat("Ratio:            ", round(acc_glm$deviance / acc_glm$df.residual, 3), "\n")
```

### 9.2 Consequences

Overdispersion does not bias the coefficient estimates, but it makes the standard errors too small, which means the
$p$-values are too optimistic. You will reject null hypotheses too often.

### 9.3 The fix: quasipoisson

The simplest remedy is to switch to `family = quasipoisson`. This estimates
an extra dispersion parameter and inflates the standard errors accordingly.


```r
acc_quasi <- glm(Freq ~ poly(Age, 2) + Gender + Month + DayOfWeek,
                 data = acc, family = quasipoisson)
display(acc_quasi)
```

```r
# Compare standard errors
se_pois  <- summary(acc_glm)$coefficients[, "Std. Error"]
se_quasi <- summary(acc_quasi)$coefficients[, "Std. Error"]

cat("SE inflation factor:", round(mean(se_quasi / se_pois), 3), "\n")
```

The quasipoisson standard errors are uniformly larger than the Poisson
standard errors. If a predictor was borderline significant under Poisson, it
may lose significance under quasipoisson — which is the honest result.

An alternative is the **negative binomial** model, which adds a separate
variance parameter. This requires the `MASS` package:

```r
# library(MASS)
# acc_nb <- glm.nb(Freq ~ poly(Age, 2) + Gender + Month + DayOfWeek,
#                  data = acc)
```

For this course, awareness of overdispersion and the quasipoisson fix is
sufficient.

---

## 10. Putting It All Together: A Complete Analysis on `warpbreaks`

The `warpbreaks` dataset, built into R, records the number of breaks in yarn
during weaving. Weaving has an interesting history, and I encourage you to look into it. 
The Pilgrims, before coming to the US, worked in part in Leiden, The Netherlands as weavers before
leaving the big city life and cultural liberalism of the country (alongside some other considerations - e.g., Spain) 
to sail for the New World. Here, the predictors are `wool` (type A or B) and `tension` (low,
medium, high). 

```r
data(warpbreaks)
head(warpbreaks)
summary(warpbreaks)
```

### 10.1 Exploratory plot

```r
ggplot(warpbreaks, aes(x = tension, y = breaks, fill = wool)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("A" = "steelblue", "B" = "firebrick")) +
  labs(title = "Yarn Breaks by Tension and Wool Type",
       x = "Tension", y = "Number of Breaks", fill = "Wool") +
  theme_minimal(base_size = 13)
```

Low tension produces more breaks. Wool type B appears to have fewer breaks
than A, especially at low tension.

### 10.2 Fit and interpret

```r
yarn_glm <- glm(breaks ~ wool + tension, data = warpbreaks, family = poisson)
display(yarn_glm)
```

```r
b_yarn <- coef(yarn_glm)

cat("Rate ratios:\n")
cat("  Wool B vs A:     ", round(exp(b_yarn["woolB"]), 3),
    "  (", round((exp(b_yarn["woolB"]) - 1) * 100, 1), "%)\n")
cat("  Medium vs Low:   ", round(exp(b_yarn["tensionM"]), 3),
    "  (", round((exp(b_yarn["tensionM"]) - 1) * 100, 1), "%)\n")
cat("  High vs Low:     ", round(exp(b_yarn["tensionH"]), 3),
    "  (", round((exp(b_yarn["tensionH"]) - 1) * 100, 1), "%)\n")
```

If the wool B coefficient is $-0.21$, then $e^{-0.21} = 0.811$: wool B has
81% the breakage rate of wool A — a 19% reduction. If the high-tension
coefficient is $-0.32$, then $e^{-0.32} = 0.726$: high tension reduces
breakage by 27% relative to low tension.

### 10.3 Check for overdispersion

```r
cat("Residual deviance / df:",
    round(yarn_glm$deviance / yarn_glm$df.residual, 3), "\n")
```

If this ratio substantially exceeds 1, refit with quasipoisson and compare
the standard errors.

### 10.4 Prediction

```r
new_yarn <- expand.grid(wool = c("A", "B"), tension = c("L", "M", "H"))
new_yarn$predicted <- predict(yarn_glm, newdata = new_yarn, type = "response")

ggplot(new_yarn, aes(x = tension, y = predicted, fill = wool)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  scale_fill_manual(values = c("A" = "steelblue", "B" = "firebrick")) +
  labs(title = "Predicted Yarn Breaks by Wool and Tension",
       x = "Tension", y = "Expected Breaks", fill = "Wool") +
  theme_minimal(base_size = 13)
```

### 10.5 Adding an interaction

Perhaps the wool-type effect depends on the tension level. We add an
interaction:

```r
yarn_int <- glm(breaks ~ wool * tension, data = warpbreaks, family = poisson)
display(yarn_int)
```

```r
# Compare deviances
cat("Additive model deviance:", round(yarn_glm$deviance, 1), "\n")
cat("Interaction model deviance:", round(yarn_int$deviance, 1), "\n")
cat("Difference:", round(yarn_glm$deviance - yarn_int$deviance, 1), "\n")
cat("Critical value (df = 2):", round(qchisq(0.95, 2), 2), "\n")
```

If the deviance drop exceeds the $\chi^2$ critical value, the interaction is
significant — the effect of wool type genuinely differs across tension
levels.

---

## 11. Key Formulas

| Concept | Formula |
|---|---|
| Poisson model | $\log(\theta_i) = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k$ |
| Expected count | $\theta_i = e^{\beta_0 + \beta_1 X_1 + \cdots}$ |
| Rate ratio (1-unit change) | $\text{RR} = e^{\hat{\beta}_j}$ |
| Rate ratio ($\Delta$-unit change) | $\text{RR} = e^{\Delta \cdot \hat{\beta}_j}$ |
| Percent change | $\%\Delta = (e^{\hat{\beta}_j} - 1) \times 100$ |
| Model with offset | $\log(\theta_i) = \log(u_i) + \beta_0 + \beta_1 X_1 + \cdots$ |
| Overdispersion check | Residual deviance / residual df $\gg 1$ |
| Deviance test | Compare (null dev. $-$ resid. dev.) to $\chi^2_{0.95, k}$ |

---

## 12. Common Mistakes

**"The coefficient IS the percent change."** No. $\hat{\beta} = 0.23$ does
not mean 23%. It means $e^{0.23} - 1 = 25.9\%$. For small coefficients the
approximation is close, but for larger ones it diverges.

**"Poisson regression can predict negative counts."** It cannot. The
exponential function guarantees positive predictions. This is the
point of using the log link.

**"I can use linear regression if my counts are large."** Sometimes done as
an approximation, but it ignores the mean-variance relationship and can
produce negative predictions. 

**"The largest coefficient means the most important predictor."** Coefficients
for categorical variables depend on the reference level, and coefficients for
continuous variables depend on the scale. Compare rate ratios over meaningful
ranges for fair comparisons.

---

## 13. Exercises

1. **Rate ratio practice.** Using `carb_glm` from Section 4, compute the
   rate ratio for a 50-unit increase in horsepower. Compute the rate ratio
   for a 1,000 lb increase in weight. Which predictor has a larger effect
   over its respective range? Express both as percentage changes.

2. **Poisson vs. linear.** Fit both `lm(carb ~ hp + wt, data = mtcars)` and
   `glm(carb ~ hp + wt, data = mtcars, family = poisson)`. Predict the
   number of carburetors for a car with 60 hp and 1.5 tons. Does the linear
   model produce a plausible count? Does the Poisson model? Predict for a car
   with 300 hp and 5 tons. Which model's prediction is more plausible at
   this extreme?

3. **Reading the EDA.** Using the simulated accident data from Section 5,
   answer: (a) Which age has the highest median accident count? (b) Which day
   of the week has the highest? (c) Is the age relationship linear or
   non-linear? Justify each answer by reference to the plots.

4. **Coefficient interpretation.** Using `acc_glm` from Section 5.3, identify
   the month with the largest positive coefficient and the day with the
   largest positive coefficient. For each, compute the rate ratio and the
   percentage change relative to the reference level. Write a one-sentence
   interpretation of each.

5. **Prediction.** Using `acc_glm`, predict the expected accident count for:
   (a) a 16-year-old male on a Saturday in July, and (b) a 25-year-old
   female on a Tuesday in February. Compute the ratio of (a) to (b). Write
   a sentence interpreting what this ratio means.

6. **Overdispersion.** Compute the residual deviance / residual df ratio for
   `acc_glm`. If the ratio exceeds 1.5, refit with `family = quasipoisson`.
   Compare the standard error of the Gender coefficient under both models.
   Does the significance conclusion change?

7. **Complete analysis.** Using the `warpbreaks` data, fit the interaction
   model `breaks ~ wool * tension` with `family = poisson`. (a) Is the
   interaction significant? (b) Compute predicted break counts for all six
   wool-tension combinations. (c) Produce a bar plot of predicted values.
   (d) Check for overdispersion. (e) If overdispersion is present, refit
   with quasipoisson and report whether any terms lose significance.

8. **The GLM trilogy.** This exercise connects all three models. Using
   `mtcars`, fit: (a) `lm(mpg ~ wt + hp)` — linear, (b)
   `glm(vs ~ wt + hp, family = binomial)` — logistic, (c)
   `glm(carb ~ wt + hp, family = poisson)` — Poisson. For each model,
   state the interpretation of the `wt` coefficient in plain English.
   In which model(s) do you need to exponentiate? In which model(s) is the
   coefficient a direct additive change?

---

## References

- Cameron, A.C. & Trivedi, P.K. (2013). *Regression Analysis of Count Data*.
  2nd ed. Cambridge University Press.
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. Ch. 6.
- Agresti, A. (2013). *Categorical Data Analysis*. 3rd ed. Wiley. Ch. 4.
