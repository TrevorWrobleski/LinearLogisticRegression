# Workshop 9: Introduction to Logistic Regression

**ST211 — Linear and Logistic Regression** Supplemental Material for the workshop

---

## 1. The Problem

Every regression model we have built so far has had a continuous outcome:
miles per gallon, hourly pay, birthweight, house prices. The outcome could
take any value on an interval, and the machinery of OLS — minimizing the sum
of squared residuals — worked naturally.

But many of the questions we care about have binary outcomes. Did the patient
survive? Did the customer churn? Will the loan default? Did I pick this
motorcycle as a favorite? The answer is yes or no. 

### 1.1 What goes wrong: a demonstration

```r
library(ggplot2)
library(arm)
library(gridExtra)

data(mtcars)
```

The `mtcars` variable `vs` records engine shape: 0 = V-shaped, 1 = straight.
Let's try to predict it from weight using ordinary linear regression.

```r
bad_model <- lm(vs ~ wt, data = mtcars)

p_bad <- ggplot(mtcars, aes(x = wt, y = vs)) +
  geom_point(size = 2.5, alpha = 0.6, colour = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, colour = "firebrick",
              linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted", colour = "grey50") +
  labs(title = "Linear Regression on a Binary Outcome",
       subtitle = "Predictions escape [0, 1] — probabilities cannot do this",
       x = "Weight (1000 lbs)", y = "Engine Type (0 = V, 1 = Straight)") +
  theme_minimal(base_size = 13)
p_bad
```

The regression line crosses below 0 for heavy cars and above 1 for light
cars. These are supposed to be probabilities. A probability of $-0.15$ or
$1.12$ is nonsensical.

Verify with numbers:

```r
predict(bad_model, newdata = data.frame(wt = c(1.5, 5.5)))
```

For a 1,500-lb car the model predicts a probability greater than 1. For a
5,500-lb car it predicts a negative probability. The linear model does not
know that probabilities live on $[0, 1]$ — it just draws the best straight
line through the data, and that line can go anywhere on the real number line.

We need a model that produces predictions bounded between
0 and 1 by construction. One such model is **logistic regression**.

---

## 2. The Solution: Logistic Regression

### 2.1 The key idea

In linear regression, we model the outcome directly:

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

In logistic regression, we model the **log-odds** of the outcome:

$$
\log\!\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X
$$

where $p = P(Y = 1 \mid X)$ is the probability that the outcome equals 1.
The left-hand side is called the **logit** function — hence "logistic"
regression.

Why does this work? Because the logit maps probabilities from $[0, 1]$ to
the entire real line $(-\infty, +\infty)$. And a linear function
$\beta_0 + \beta_1 X$ lives on the entire real line. So the two sides of
the equation are compatible. When we invert the logit to get back to
probabilities, the result is always between 0 and 1.

**Logistic regression is linear regression on the log-odds scale.** The
S-shaped curve you see in plots is simply what happens when you transform
log-odds back into probabilities.

### 2.2 Understanding Odds: A Kentucky Derby Primer

Before we formalize the probability–odds–log-odds pipeline, let's build
intuition using a setting where odds are part of the everyday language:
horse racing.

#### What the tote board is telling you

Walk up to Churchill Downs on the first Saturday in May and look at the tote
board. Next to each horse you'll see something like:

| Horse | Morning-Line Odds |
|---|---|
| Fierceness | 5–1 |
| Sierra Leone | 8–1 |
| Mystik Dan | 18–1 |
| Catching Freedom | 10–1 |
| Forever Young | 6–1 |

*(Inspired by the 2024 Kentucky Derby field.)*

The number "5–1" is read as **"five to one against"** — it means the horse
is expected to lose 5 times for every 1 time it wins. In other words, if you
could re-run the race 6 times under identical conditions, the horse would win
about 1 of those 6 races.

Horse racing odds are quoted as **odds against**: the number of expected
failures per success. This is the *reciprocal* of the "odds" we use in
statistics, where by convention we express **odds in favor**: the number of
expected successes per failure. The relationship is simple:

$$
\text{odds against} = \frac{1}{\text{odds in favor}}
$$

So when the tote board says Fierceness is 5–1, the statistical odds in favor
of Fierceness winning are $1/5 = 0.20$.

#### From odds to probability — and back

For a horse at 5–1 against, we can recover the implied probability of
winning:

$$
p = \frac{1}{5 + 1} = \frac{1}{6} \approx 0.167
$$

The general formula when the tote board reads $a$–1 against is:

$$
p = \frac{1}{a + 1}
$$

Going in reverse — converting a probability into odds — is just as
straightforward. Suppose you believe a horse has a 25% chance of winning
($p = 0.25$). Then the odds in favor are:

$$
\text{odds} = \frac{p}{1 - p} = \frac{0.25}{0.75} = \frac{1}{3} \approx 0.333
$$

and the tote board would show this as 3–1 against (three expected losses per
win).

#### A worked example: the whole field

Let's convert the full mini-field from probability to odds to log-odds. This
is worth working through by hand:

| Horse | Odds Against | Implied Prob | Odds (in favor) | Log-Odds |
|---|---|---|---|---|
| Fierceness | 5–1 | $1/6 = 0.167$ | $0.167/0.833 = 0.200$ | $\log(0.200) = -1.61$ |
| Sierra Leone | 8–1 | $1/9 = 0.111$ | $0.111/0.889 = 0.125$ | $\log(0.125) = -2.08$ |
| Mystik Dan | 18–1 | $1/19 = 0.053$ | $0.053/0.947 = 0.056$ | $\log(0.056) = -2.89$ |
| Catching Freedom | 10–1 | $1/11 = 0.091$ | $0.091/0.909 = 0.100$ | $\log(0.100) = -2.30$ |
| Forever Young | 6–1 | $1/7 = 0.143$ | $0.143/0.857 = 0.167$ | $\log(0.167) = -1.79$ |

Notice three things:

1. **Every horse has negative log-odds.** That's because every horse has less
   than a 50% chance of winning — there are many horses in the field. A
   log-odds of zero corresponds to an even-money bet ($p = 0.50$), which you
   would see on the tote board as 1–1, also called **"evens."**

2. **Log-odds preserve the rank order.** Fierceness is the favorite (highest
   probability, highest log-odds, least negative). Mystik Dan is the longest
   shot (lowest probability, lowest log-odds, most negative). The log-odds
   transformation does not scramble the ranking — it just changes the scale.

3. **Differences in log-odds are more interpretable than differences in
   probability at the extremes.** The gap in probability between Fierceness
   (0.167) and Sierra Leone (0.111) is 5.6 percentage points. The gap
   between Sierra Leone (0.111) and Mystik Dan (0.053) is 5.8 percentage
   points — almost the same. But the odds tell us that Fierceness
   is 1.6× more likely than Sierra Leone, while Sierra Leone is 2.2× more
   likely than Mystik Dan. The log-odds scale makes this multiplicative
   relationship additive, which is exactly the property that makes it
   convenient for regression.

#### Why "odds" instead of "probability"?

You might wonder why don't we model probabilities
directly? 

- **Probabilities are bounded** between 0 and 1. They cannot go below 0 or
  above 1. This makes them awkward as the output of a linear model, because
  $\beta_0 + \beta_1 X$ can be any real number.
- **Odds are half-bounded** — they range from 0 to $+\infty$. Better, but
  still not symmetric: the odds of winning can go from 0 to infinity, but the
  odds of losing given the same information range over the same interval.
  There is an asymmetry between "very unlikely" (odds near 0) and "very
  likely" (odds near $\infty$).
- **Log-odds are unbounded and symmetric.** They range from $-\infty$ to
  $+\infty$, and a probability of 0.25 produces a log-odds of $-1.10$ while
  a probability of 0.75 produces $+1.10$ — equal magnitude, opposite sign.
  This is a natural home for a linear predictor.

So odds are the crucial middle step in the transformation from probability
(which is what we care about) to log-odds (which is what the model operates
on). Understanding odds — whether from Churchill Downs or from a logistic
regression table — is essential for reading and interpreting the model.

#### From the Derby to the ICU

In the rest of this workshop, instead of asking "what are the odds that Fierceness
wins?", we'll ask "what are the odds that this patient survives?" or "what
are the odds that this motorcycle is a favorite?" **Odds express how many times more
likely one outcome is compared to the other, and the log of those odds is
what our model actually estimates.**

### 2.3 Building intuition: probability → odds → log-odds

These three quantities carry the same information, but expressed on different
scales. Let's start with something we've seen before:

**Probability.** If there is a 75% chance a patient survives, $p = 0.75$.

**Odds.** Odds rewrite that same information as a ratio: $\text{odds} = p / (1 - p)$. For $p = 0.75$, the odds are
$0.75 / 0.25 = 3$ — the patient is three times more likely to survive than
to die. Odds range from $0$ to $+\infty$.

**Log-odds (logit).** Take the natural log of the odds:
$\log(p / (1 - p))$. For $p = 0.75$, the log-odds are $\log(3) = 1.10$.
Log-odds range from $-\infty$ to $+\infty$.

The following table is worth memorizing. 

| Probability | Odds | Log-odds |
|---|---|---|
| 0.10 | 0.11 | $-2.20$ |
| 0.25 | 0.33 | $-1.10$ |
| 0.50 | 1.00 | $0.00$ |
| 0.75 | 3.00 | $+1.10$ |
| 0.90 | 9.00 | $+2.20$ |

Notice the symmetry: probabilities of 0.25 and 0.75 produce log-odds of
equal magnitude but opposite sign. A log-odds of 0 corresponds to a
coin-flip probability of 0.50. Positive log-odds mean "more likely yes than
no"; negative log-odds mean "more likely no than yes."

### 2.4 The logistic function (inverse logit)

To convert from log-odds back to probability, we invert the logit:

$$
p = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
$$

This function is called the **logistic function** (or **sigmoid**). It is
the source of the S-shaped curve: when $\beta_0 + \beta_1 X$ is large and
negative, $p \approx 0$; when it is large and positive, $p \approx 1$; and
the transition between the two is smooth.

### 2.5 Seeing the fix

```r
p_good <- ggplot(mtcars, aes(x = wt, y = vs)) +
  geom_point(size = 2.5, alpha = 0.6, colour = "steelblue") +
  geom_smooth(method = "glm",
              method.args = list(family = "binomial"),
              se = TRUE, colour = "blue", linewidth = 0.8) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted", colour = "grey50") +
  labs(title = "Logistic Regression on a Binary Outcome",
       subtitle = "Predictions stay between 0 and 1 by construction",
       x = "Weight (1000 lbs)", y = "P(Straight Engine)") +
  theme_minimal(base_size = 13)

grid.arrange(p_bad, p_good, ncol = 2)
```

We're using the same data and the same predictors, but the
logistic curve never crosses the dotted lines at 0 and 1.

### 2.6 R syntax

The R function for logistic regression is `glm()` — **generalized** linear
model. It takes an additional argument, `family`, that specifies the type of
model:

```r
my_model <- glm(outcome ~ predictor, data = my_data,
                family = binomial(link = "logit"))
```

The `link = "logit"` tells R to use the logit transformation. The rest of
the syntax — the formula, the `data` argument — works the same as in `lm()`.
You can use `display()` from the `arm` package to inspect the output, just
as before.

---

## 3. The ICU Data: One Continuous Predictor

### 3.1 The data

The workshop examines survival data for 237 patients admitted to an intensive
care unit in an American hospital. The outcome is `Lived`: 1 if the patient
survived to discharge, 0 if they died in the ICU. The predictors are Age,
Sex, systolic blood pressure (SBP), and heart rate (HR).

```r
icu.dat <- read.csv("icu.csv", header = TRUE, stringsAsFactors = TRUE)
summary(icu.dat)
```

### 3.2 Visualizing binary outcomes

When the outcome is binary, a scatterplot of $Y$ against $X$ is not
informative — you just see two rows of points at $Y = 0$ and $Y = 1$. The
standard visualization for continuous predictors is a **boxplot grouped by
the outcome**:

```r
icu_cont <- icu.dat %>%
  dplyr::select(Lived, Age, SBP, HR) %>%
  mutate(Lived = factor(Lived, labels = c("Died", "Lived"))) %>%
  tidyr::pivot_longer(cols = c(Age, SBP, HR),
                      names_to = "variable", values_to = "value")

ggplot(icu_cont, aes(x = Lived, y = value)) +
  geom_boxplot(fill = "steelblue", alpha = 0.4) +
  facet_wrap(~ variable, scales = "free_y") +
  coord_flip() +
  labs(title = "ICU Data: Continuous Predictors by Survival Status") +
  theme_minimal(base_size = 13)
```

The boxplots are flipped on their side as a reminder that the logistic
relationship is non-linear — the predictor is on the y-axis and the binary
outcome defines the groups. Read these as: "For patients who died, the
distribution of Age looked like this; for patients who survived, it looked
like that."

**Age** shows a clear separation — patients who died tend to be older. **SBP**
shows a modest difference. **HR** shows almost no difference. From these
plots, we expect Age to be the strongest continuous predictor.

For categorical predictors, the appropriate visualization is a **bar plot**
showing the proportion of each outcome within each category:

```r
icu_sex <- icu.dat %>%
  mutate(Lived = factor(Lived, labels = c("Died", "Lived"))) %>%
  group_by(Sex, Lived) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Sex) %>%
  mutate(pct = n / sum(n))

ggplot(icu_sex, aes(x = Sex, y = pct, fill = Lived)) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "ICU Survival by Sex", y = "Proportion", fill = "Outcome") +
  theme_minimal(base_size = 13)
```

### 3.3 Fitting the model

```r
icu.glm <- glm(Lived ~ Age, data = icu.dat,
               family = binomial(link = "logit"))
display(icu.glm)
```

The output gives:
- Intercept: 3.19
- Age coefficient: $-0.04$

The fitted model is:

$$
\log\!\left(\frac{P(\text{Lived} = 1)}{P(\text{Lived} = 0)}\right) = 3.19 - 0.04 \times \text{Age}
$$

Two things to note: first, the Age coefficient is negative, as
we expected from the boxplot — older patients have lower log-odds of
survival. Second, the output looks almost identical to `lm()` output, but the
numbers now refer to log-odds, not the outcome directly.

### 3.4 The binned proportions plot

To see how the model fits the data, we can bin patients into 5-year age
groups, compute the observed proportion who survived in each bin, and overlay
the model's predicted probability curve.

```r
icu.dat$age_bin <- cut(icu.dat$Age, breaks = seq(15, 100, by = 5))

bin_props <- icu.dat %>%
  group_by(age_bin) %>%
  summarise(prop_lived = mean(Lived),
            mid_age = mean(Age),
            n = n(), .groups = "drop")

age_seq <- data.frame(Age = seq(15, 95, by = 0.5))
age_seq$pred_prob <- predict(icu.glm, newdata = age_seq, type = "response")

ggplot() +
  geom_point(data = bin_props, aes(x = mid_age, y = prop_lived, size = n),
             alpha = 0.7, colour = "steelblue") +
  geom_line(data = age_seq, aes(x = Age, y = pred_prob),
            colour = "firebrick", linewidth = 1) +
  labs(title = "ICU Survival Probability by Age",
       subtitle = "Points = observed proportions; Curve = logistic model",
       x = "Age", y = "P(Survived)") +
  scale_size_continuous(name = "Patients\nin bin") +
  theme_minimal(base_size = 13)
```

For patients around 15–20, the survival probability is approximately 90%.
It declines gradually to around 40% for patients in their 90s. The model
captures this decline with its S-curve — though over this particular age
range the curve is nearly linear. You would see the full S-shape only if the
data extended much further in both directions.

Note the use of `type = "response"` in `predict()`. This tells R to return
predicted probabilities rather than predicted log-odds. Without it, you get
the log-odds scale, which requires manual exponentiation via the inverse
logit to recover probabilities.

---

## 4. Interpreting Coefficients

### 4.1 The coefficient on the log-odds scale

The coefficient of Age is $-0.04$. This means: for every one-year increase
in age, the log-odds of survival decrease by 0.04. But log-odds are hard
to think about. We need a more natural scale.

### 4.2 Odds ratios

Exponentiate the coefficient:

$$
e^{-0.04} = 0.961
$$

This is the **odds ratio**. For every one-year increase in age, the odds of
survival are **multiplied by 0.961** — they decrease by about 3.9%.

```r
exp(coef(icu.glm)) 
```

But a one-year change is small. A more practical unit is 10 years:

$$
e^{10 \times (-0.04)} = e^{-0.4} = 0.670
$$

Over a 10-year increase in age, the odds of survival drop by about a third.

```r
exp(10 * coef(icu.glm)["Age"])
```

### 4.3 A critical property

The odds ratio is constant across the entire range of Age. Going from 25 to
35 gives the same odds ratio ($0.67$) as going from 65 to 75. This is
because the model is linear on the log-odds scale, and linearity means the
slope is the same everywhere. Exponentiation turns an additive constant into
a multiplicative constant.

This is analogous to the log-level model from Workshop 7: the coefficient
there represented a constant *percentage* change in the outcome, regardless
of the starting point. The same logic applies here — we are on a log
scale (log-odds instead of log-outcome), and the interpretation carries over
directly.

### 4.4 Odds ratios: when to use what summary

| Odds ratio | Meaning |
|---|---|
| $OR = 1$ | No association between predictor and outcome |
| $OR > 1$ | Higher values of $X$ are associated with **higher** probability of $Y = 1$ |
| $OR < 1$ | Higher values of $X$ are associated with **lower** probability of $Y = 1$ |

For Age: $OR = 0.96 < 1$, so higher age is associated with lower survival
probability. This matches the boxplot, the binned proportions plot, and our
intuition.

### 4.5 Why not just talk about probability changes?

You might wonder: "Why odds ratios? Why not just say how much the probability
changes?" The reason is that the probability change depends on where you
start. Consider two patients:

- **Patient A** (Age 25): predicted probability of survival = 0.90
- **Patient B** (Age 75): predicted probability of survival = 0.55

```r
inv_logit <- function(x) 1 / (1 + exp(-x))

p_25 <- inv_logit(3.19 - 0.04 * 25)
p_35 <- inv_logit(3.19 - 0.04 * 35)
p_65 <- inv_logit(3.19 - 0.04 * 65)
p_75 <- inv_logit(3.19 - 0.04 * 75)

cat("Age 25 → 35: probability drops from",
    round(p_25, 3), "to", round(p_35, 3),
    "  (change:", round(p_35 - p_25, 3), ")\n")
cat("Age 65 → 75: probability drops from",
    round(p_65, 3), "to", round(p_75, 3),
    "  (change:", round(p_75 - p_65, 3), ")\n")
```

The 10-year change produces a different absolute probability change depending
on where you start — the change is larger in the middle of the curve and
smaller at the extremes. That is why we report the odds ratio (which is
constant) rather than the probability change (which is not). The odds ratio
is the one number that summarizes the effect of Age without requiring you to
specify a starting point.

This is a direct analogy to the log model from Workshop 7: $e^{\hat{\beta}}$
gave us a constant multiplier that worked regardless of the baseline. Here,
$e^{\hat{\beta}}$ gives us a constant odds multiplier that works regardless
of where on the S-curve we are.

### 4.6 Centering the predictor

The intercept of 3.19 is the log-odds of survival at Age = 0 — a newborn.
This is meaningless for adult ICU patients. If we center Age at its mean, the
intercept becomes the log-odds of survival for a patient at the average age:

```r
icu.dat$cent.Age <- icu.dat$Age - mean(icu.dat$Age)
icu.glm.cent <- glm(Lived ~ cent.Age, data = icu.dat,
                     family = binomial(link = "logit"))
display(icu.glm.cent)
```

The slope does not change — centering only moves the reference point, just
as in linear regression.

---

## 5. Classification Tables

### 5.1 From probabilities to predictions

Logistic regression outputs a predicted probability for each observation. But
the actual outcome is 0 or 1, not a probability. To evaluate the model, we
need a rule that converts probabilities into binary predictions. The simplest
rule:

$$
\hat{Y}_i =
\begin{cases}
1 & \text{if } \hat{p}_i > 0.5 \\
0 & \text{if } \hat{p}_i \leq 0.5
\end{cases}
$$

Then we compare predictions to reality in a **classification table** (also
called a **confusion matrix**).

### 5.2 ICU classification table

```r
pred.glm <- as.numeric(icu.glm$fitted.values > 0.5)
glm.dat <- data.frame(predicted = pred.glm, observed = icu.dat$Lived)
table(glm.dat)
```

This produces:

|  | Observed 0 (Died) | Observed 1 (Lived) |
|---|---|---|
| **Predicted 0** | 19 | 9 |
| **Predicted 1** | 57 | 152 |

Read each cell:
- **Top-left (19):** correctly predicted deaths — **true negatives**
- **Top-right (9):** predicted death but patient survived — **false negatives**
- **Bottom-left (57):** predicted survival but patient died — **false positives**
- **Bottom-right (152):** correctly predicted survivors — **true positives**

### 5.3 Summary statistics

**Overall accuracy:**

$$
\frac{19 + 152}{237} = \frac{171}{237} = 0.722 \quad (72.2\%)
$$

**Sensitivity** (true positive rate — how well does the model identify
survivors?):

$$
\frac{152}{152 + 57} = \frac{152}{209} = 0.727 \quad (72.7\%)
$$

**Specificity** (true negative rate — how well does the model identify
deaths?):

$$
\frac{19}{19 + 9} = \frac{19}{28} = 0.679 \quad (67.9\%)
$$

```r
cat("Accuracy:   ", round((19 + 152) / 237, 3), "\n")
cat("Sensitivity:", round(152 / (152 + 57), 3), "\n")
cat("Specificity:", round(19 / (19 + 9), 3), "\n")
```

### 5.4 The base rate trap

Before concluding that 72% accuracy is good, ask: what would the laziest
possible model achieve? A model that predicts "survived" for every single
patient gets:

$$
\frac{161}{237} = 0.679 \quad (67.9\%)
$$

because about 68% of patients actually survived. Our model's 72% is only a
modest improvement over predicting the majority class every time. **Always
compare your model's accuracy to the base rate.** A model that barely beats
the base rate is not very useful, regardless of how impressive the absolute
accuracy sounds.

---

## 6. The Motorcycle Data: Logistic Regression in Practice

### 6.1 The question

The `moto_2020` dataset contains 656 motorcycles from 15 major manufacturers,
all from model year 2020. Among these, 59 are flagged as `trevors_fav = "Yes"`.
The question: **can we predict which bikes are favorites from their
specifications?**

This is a legitimate logistic regression problem with a twist: the outcome is
heavily imbalanced (9% Yes vs. 91% No). This imbalance will teach us
something important about classification thresholds.

```r
moto.dat <- read.csv("moto_2020.csv", header = TRUE, stringsAsFactors = TRUE)
moto.dat$trevors_fav_num <- as.numeric(moto.dat$trevors_fav == "Yes")

table(moto.dat$trevors_fav)
cat("Proportion of favorites:", round(mean(moto.dat$trevors_fav_num), 3), "\n")
```

### 6.2 Visualizing the predictors

```r
p_disp <- ggplot(moto.dat, aes(x = trevors_fav, y = displacement_cc)) +
  geom_boxplot(fill = c("grey80", "steelblue"), alpha = 0.7) +
  coord_flip() +
  labs(title = "Displacement", x = "Favorite?", y = "cc") +
  theme_minimal(base_size = 11)

p_pow <- ggplot(moto.dat %>% dplyr::filter(!is.na(power_hp)),
                aes(x = trevors_fav, y = power_hp)) +
  geom_boxplot(fill = c("grey80", "steelblue"), alpha = 0.7) +
  coord_flip() +
  labs(title = "Power", x = "Favorite?", y = "HP") +
  theme_minimal(base_size = 11)

grid.arrange(p_disp, p_pow, ncol = 2)
```

Favorites tend to have substantially higher displacement and more horsepower.
The separation is clearer than what we saw in the ICU data — the boxplots
barely overlap in the upper quartile range — which suggests displacement
should be a strong predictor.

For a categorical predictor, we use a proportional bar plot:

```r
moto_cat <- moto.dat %>%
  group_by(Category, trevors_fav) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Category) %>%
  mutate(pct = n / sum(n))

ggplot(moto_cat, aes(x = reorder(Category, -pct), y = pct,
                      fill = trevors_fav)) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "Favorites by Motorcycle Category",
       x = "Category", y = "Proportion", fill = "Favorite?") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

Some categories — Enduro, Allround, Super Motard — have a noticeably higher
proportion of favorites. Sport bikes and scooters have very few.

### 6.3 Fitting a logistic model

```r
moto.glm <- glm(trevors_fav_num ~ displacement_cc, data = moto.dat,
                family = binomial(link = "logit"))
display(moto.glm)
```

### 6.4 The S-curve

```r
disp_seq <- data.frame(displacement_cc = seq(49, 2500, by = 5))
disp_seq$pred_prob <- predict(moto.glm, newdata = disp_seq,
                               type = "response")

ggplot() +
  geom_jitter(data = moto.dat,
              aes(x = displacement_cc, y = trevors_fav_num),
              height = 0.03, alpha = 0.3, size = 1, colour = "steelblue") +
  geom_line(data = disp_seq, aes(x = displacement_cc, y = pred_prob),
            colour = "firebrick", linewidth = 1) +
  labs(title = "Logistic Regression: Favorite ~ Displacement",
       subtitle = "Curve = predicted probability from logistic model",
       x = "Displacement (cc)", y = "P(Favorite)") +
  theme_minimal(base_size = 13)
```

For bikes under 200cc — scooters, small commuters — the predicted probability
is near zero. It climbs through the mid-range and reaches maybe 25–30% for
the largest displacement bikes. Notice that the curve never gets close to 1,
even at 2,500cc. This makes sense: even among big bikes, most are not
favorites. The model has learned that displacement *increases* the
probability but is not sufficient on its own.

### 6.5 Odds ratio interpretation

A one-cc change is negligible, so we interpret per 100cc:

```r
cat("Odds ratio per 100cc increase:",
    round(exp(100 * coef(moto.glm)["displacement_cc"]), 3), "\n")
```

For every additional 100cc of displacement, the odds of being a favorite are
multiplied by this factor. A rider shopping between a 600cc and a 700cc
bike can expect the odds of the larger bike being a favorite to increase by
approximately this ratio — holding everything else constant (though in this
univariate model, nothing else is being held constant).

### 6.6 The linear-vs-logistic comparison on motorcycle data

```r
p_moto_bad <- ggplot(moto.dat, aes(x = displacement_cc, y = trevors_fav_num)) +
  geom_jitter(height = 0.03, alpha = 0.3, size = 1, colour = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick",
              linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted", colour = "grey50") +
  labs(title = "Linear Regression (wrong tool)",
       x = "Displacement (cc)", y = "P(Favorite)") +
  theme_minimal(base_size = 11)

p_moto_good <- ggplot(moto.dat, aes(x = displacement_cc, y = trevors_fav_num)) +
  geom_jitter(height = 0.03, alpha = 0.3, size = 1, colour = "steelblue") +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = FALSE, colour = "firebrick", linewidth = 0.8) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted", colour = "grey50") +
  labs(title = "Logistic Regression (right tool)",
       x = "Displacement (cc)", y = "P(Favorite)") +
  theme_minimal(base_size = 11)

grid.arrange(p_moto_bad, p_moto_good, ncol = 2)
```

The linear model predicts negative probabilities for the smallest bikes. The
logistic model stays bounded. Same lesson as Section 1 — but now you are
seeing it on data you will work with yourself.

---

## 7. Classification with Imbalanced Data

### 7.1 The default threshold fails

```r
pred.moto <- as.numeric(moto.glm$fitted.values > 0.5)
table(predicted = pred.moto, observed = moto.dat$trevors_fav_num)
```

Look at the table. The model predicts 0 for (almost) everything. It never
(or very rarely) predicts a bike as a favorite. And yet its accuracy is
approximately 91% — because 91% of bikes are *not* favorites.

This is the class imbalance problem in action. When the positive class is
rare, the 0.5 threshold is too high — the predicted probabilities from the
model may never reach 0.5, so the model takes the safe route and predicts the
majority class every time. The model is not wrong in a statistical sense: its
probabilities are well-calibrated. The problem is that the decision rule (the
0.5 cutoff) is inappropriate for this base rate.

### 7.2 Lowering the threshold

```r
# Threshold = 0.3
pred.30 <- as.numeric(moto.glm$fitted.values > 0.3)
cat("Threshold = 0.3:\n")
table(predicted = pred.30, observed = moto.dat$trevors_fav_num)

# Threshold = 0.15
pred.15 <- as.numeric(moto.glm$fitted.values > 0.15)
cat("\nThreshold = 0.15:\n")
table(predicted = pred.15, observed = moto.dat$trevors_fav_num)

# Threshold = 0.08
pred.08 <- as.numeric(moto.glm$fitted.values > 0.08)
cat("\nThreshold = 0.08:\n")
table(predicted = pred.08, observed = moto.dat$trevors_fav_num)
```

As you lower the threshold, the model begins flagging more bikes as potential
favorites. Some of those flags are correct (true positives increase), but
others are false alarms (false positives increase too). This is the
fundamental tradeoff:

| Lower threshold | Higher threshold |
|---|---|
| More true positives (catches more real favorites) | Fewer false positives (fewer false alarms) |
| More false positives (flags non-favorites) | More false negatives (misses real favorites) |

There is no universally correct threshold. The right choice depends on the
cost of each type of error in your specific application:

- **Medical screening:** false negatives are dangerous (missing a disease),
  so you lower the threshold to catch more cases, accepting more false
  positives.
- **Spam filtering:** false positives are annoying (real email sent to spam),
  so you raise the threshold, accepting more spam slipping through.
- **Predicting motorcycle taste:** low stakes either way, but the exercise
  teaches you that 0.5 is a default, not a law.

### 7.3 Accuracy is not enough

The motorcycle example makes this crystal clear. At the 0.5 threshold, the
model achieves approximately 91% accuracy — and it is completely useless. It
has learned nothing except "say no to everything."

This is why we report sensitivity and specificity alongside accuracy:

- **Accuracy** asks: "Of all predictions, how many were correct?"
- **Sensitivity** asks: "Of the actual positives, how many did we catch?"
- **Specificity** asks: "Of the actual negatives, how many did we correctly
  leave alone?"

A model with 91% accuracy but 0% sensitivity has not learned anything about
the positive class. You need all three numbers — or at minimum, accuracy
compared against the base rate — to evaluate a classifier.

---

## 8. A Single Binary Predictor

### 8.1 Visualization

When both the outcome and the predictor are binary, neither a scatterplot nor
a boxplot makes sense. The appropriate visualization is a **proportional bar
plot** — equivalent to a $2 \times 2$ contingency table rendered graphically.

In the ICU data, we can examine whether Sex predicts survival:

```r
icu_sex_table <- table(icu.dat$Sex, icu.dat$Lived)
prop.table(icu_sex_table, margin = 1)  # row proportions
```

The row proportions tell you: what fraction of females survived? What fraction
of males survived? If these proportions are similar, Sex is not a useful
predictor. If they differ, there may be an association.

### 8.2 Fitting and interpreting

```r
icu.glm.sex <- glm(Lived ~ Sex, data = icu.dat,
                    family = binomial(link = "logit"))
display(icu.glm.sex)
```

The coefficient of Sex (say, `SexMale`) gives the difference in log-odds
between males and the reference category (females). Exponentiate it:

$$
OR = e^{\hat{\beta}_{\text{Male}}}
$$

If $OR > 1$, males have higher odds of survival than females. If $OR < 1$,
males have lower odds. If $OR \approx 1$, there is no meaningful difference.

This works exactly like dummy variables in linear regression — the only
change is the scale. In linear regression, the coefficient was a difference
in means. In logistic regression, it is a difference in log-odds (or,
equivalently, a ratio of odds after exponentiation).

---

## 9. Connecting to What You Already Know

Logistic regression is not as foreign as it first appears. The table below
maps concepts from linear regression to their logistic counterparts:

| Linear regression | Logistic regression |
|---|---|
| Outcome $Y \in \mathbb{R}$ | Outcome $Y \in \{0, 1\}$ |
| `lm()` | `glm(..., family = binomial)` |
| Coefficients = change in $Y$ | Coefficients = change in log-odds |
| Compare coefficients directly | Exponentiate: $e^{\hat{\beta}}$ = odds ratio |
| $R^2$ | Deviance, classification accuracy |
| Residuals vs. fitted | Classification table, binned proportions |
| `predict()` returns $\hat{y}$ | `predict(..., type = "response")` returns $\hat{p}$ |
| MSPE for model comparison | Accuracy, sensitivity, specificity |
| Centering helps intercept | Centering helps intercept (same idea) |

The algebraic structure is preserved — you still write
$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots$ on the right-hand side.
The *ceteris paribus* interpretation from the reading week materials applies
on the log-odds scale: $\beta_1$ is the change in log-odds of the outcome
per unit increase in $X_1$, **holding all other predictors constant**.

The set theory from reading week becomes directly relevant: we are modeling
$P(Y = 1 \mid X)$, the probability of an event. That event is a set in a
sample space. The logit is just a transformation that makes it possible to
model this probability using the linear algebra we already know.

---

## 10. Exercises

1. **The linear model fails.** Using `mtcars`, fit `lm(vs ~ wt)` and
   `glm(vs ~ wt, family = binomial)`. Predict for `wt = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)` from both models. At which weight values does
   the linear model produce impossible probabilities? At which weight values
   do the two models agree most closely? Plot both curves on the same graph.

2. **Odds ratio practice.** Using the ICU logistic regression
   `glm(Lived ~ Age, ...)`:
   (a) Compute the odds ratio for a 1-year increase in Age.
   (b) Compute the odds ratio for a 10-year increase.
   (c) Compute the odds ratio for a 20-year increase.
   (d) Verify that the 20-year odds ratio equals the 10-year odds ratio
   *squared*. Why does this property hold?

3. **Probability is not odds.** For the ICU model, compute the predicted
   probability of survival at Ages 30, 50, 70, and 90. Then compute the
   *change* in probability for a 10-year increase at each starting age.
   Verify that the probability change is not constant (it depends on where
   you start), while the odds ratio is constant. Produce a plot of predicted
   probability vs. Age with vertical arrows showing the probability change
   at each starting age.

4. **Classification table by hand.** Using the ICU model:
   (a) Compute the predicted probability for each patient using
   `predict(..., type = "response")`.
   (b) Apply the 0.5 threshold to create predictions.
   (c) Build the classification table using `table()`.
   (d) Compute accuracy, sensitivity, and specificity.
   (e) Now use a threshold of 0.7 instead. How do sensitivity and
   specificity change? Explain the tradeoff in one sentence.

5. **The base rate test.** For both the ICU data and the motorcycle data,
   compute the accuracy of the "always predict the majority class" model.
   Compare this to the accuracy of the logistic model at the 0.5 threshold.
   In which dataset does the logistic model add more value beyond the base
   rate? Why?

6. **Motorcycle classification.** Fit `glm(trevors_fav_num ~ displacement_cc, ...)` on the motorcycle data. Build classification tables at thresholds
   of 0.5, 0.3, 0.15, and 0.08. For each threshold, compute accuracy,
   sensitivity, and specificity. Plot all three statistics as a function of
   the threshold. At what threshold does sensitivity first exceed 50%? What
   is the corresponding specificity?

7. **Binary predictor.** Fit `glm(Lived ~ Sex, ...)` on the ICU data.
   (a) Compute the odds ratio.
   (b) Verify that the odds ratio equals the cross-product ratio from the
   $2 \times 2$ table: $\frac{a \cdot d}{b \cdot c}$ where $a, b, c, d$
   are the cell counts.
   (c) Build the classification table. Does the model predict different
   outcomes for males and females, or does it predict the same class for
   everyone?

8. **From Workshop 7 to Workshop 9.** This exercise connects the log
   interpretation from Workshop 7 to the logistic interpretation. Fit
   `lm(log(mpg) ~ wt, data = mtcars)` and `glm(vs ~ wt, data = mtcars, family = binomial)`.
   For the log-linear model, $e^{\hat{\beta}_1}$ is the multiplicative
   change in mpg per 1,000 lbs. For the logistic model,
   $e^{\hat{\beta}_1}$ is the mul
