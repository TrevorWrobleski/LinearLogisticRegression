# Workshop 2: Data Frames, Subsetting, and Hypothesis Testing

**ST211 — Linear and Logistic Regression**
Supplemental Material for the workshop

---

## 1. Loading Data into R

In practice, data rarely lives inside your R script. It arrives as a file —
usually `.csv` or `.txt` — and the first task is to read it correctly.

### 1.1 The right way

```r
first.dat <- read.csv("age_hourpay.csv", header = TRUE)
```

`header = TRUE` tells R that the first row contains column names, not data.
This is the default for `read.csv()`, but it is good practice to state it
explicitly when your data is already labeled.

### 1.2 Common mistakes

**Using the wrong reader.** `read.table()` expects whitespace-separated data
by default. If you feed it a comma-separated file, it will collapse every row
into a single string:

```r
wrong.dat <- read.table("age_hourpay.csv", header = TRUE)
head(wrong.dat)
```

**Ignoring the header.** Setting `header = FALSE` treats the column names as a
data row, coerces everything to character, and corrupts the types of the
columns:

```r
wrong.dat.2 <- read.csv("age_hourpay.csv", header = FALSE)
head(wrong.dat.2)
```

> Always inspect your data immediately after loading. The bugs
> introduced by a wrong import typically do not result in a visible error. 


How can we check if we have properly loaded the data? 

Perhaps the simplest way is to look at the 'Enviroment' tab (upper right side normally) and click on the imported data. 
This actually runs a command in your console, and if you are interested in viewing the data in a table, you can try 

```r
View(NAME OF DATA)
```

---

## 2. Inspecting a Data Frame

A data frame is R's representation of a rectangular dataset: rows are
observations, columns are variables. After loading, you should check a few things.

Many hours have been wasted conducting analyses on data which were improperly loaded.

```r
dim(first.dat)        # number of rows and columns
colnames(first.dat)   # variable names
head(first.dat)       # first 6 rows; if you are interested in more rows, just add a comma and a number after the data, e.g., head(first.dat, 10)
tail(first.dat)       # last 6 rows; how would you see more or fewer rows here? 
summary(first.dat)    # five-number summary + mean for each column
```

`summary()` is particularly valuable. It immediately tells you the range of
each variable, flags the presence of `NA` values, and lets you spot absurdities
(e.g., negative ages, hourly pay inconsistent with reality) that signal data-quality problems.

### 2.1 Accessing individual columns

Two equivalent ways:

```r
first.dat$age        # by name (preferred — self-documenting)
first.dat[, 2]       # by column index
```

Use the `$` syntax in scripts. Use index syntax when you need to operate on
columns programmatically (e.g., looping over column numbers).

---

## 3. Subsetting and Filtering

Subsetting is the act of extracting a portion of your data that satisfies some
condition. In regression analysis, you will do this constantly — removing
outliers, restricting to a subpopulation, creating training and test sets.

### 3.1 Logical vectors as filters

When you write a comparison on a column, R returns a logical vector the same
length as the data:

```r
head(first.dat$age < 18)   # TRUE/FALSE for first 6 rows
```

You can use `which()` to get the row indices where the condition is `TRUE`, and
then count them:

```r
which(first.dat$age < 18)
length(which(first.dat$age < 18))
```

### 3.2 Three equivalent ways to subset

Suppose we want all observations where `age >= 18`:

```r
# Method 1: which() inside bracket notation
over18 <- first.dat[which(first.dat$age >= 18), ]

# Method 2: logical vector directly (no which)
over18 <- first.dat[first.dat$age >= 18, ]

# Method 3: subset()
over18 <- subset(first.dat, age >= 18)

dim(over18)  # confirm all three give the same result
```

All three are correct. Methods 1 and 2 differ in how they handle `NA` values:
`which()` silently drops `NA`s, whereas the direct logical approach propagates
them (you get rows of `NA`). In clean data this distinction is invisible; in
messy data it matters. `subset()` behaves like `which()`.

### 3.3 Compound conditions

You can combine conditions with `&` (and) and `|` (or):

```r
second.dat <- subset(first.dat, age >= 18 & hourpay <= 50)
dim(second.dat)
```

### 3.4 Tabulation after subsetting

```r
under18 <- first.dat[which(first.dat$age < 18), ]
table(under18$gender)
```

`table()` produces a frequency count — the categorical analog of a histogram.

---

## 4. Aggregation

A recurring task in data analysis is computing summary statistics *within
groups*. There are two clean approaches.

### 4.1 Base R: `aggregate()`

```r
aggregate(hourpay ~ gender, data = first.dat, FUN = mean)
```

The formula syntax `hourpay ~ gender` reads: "hourly pay *as a function of*
gender." This is the same `~` notation you will use for regression models.

### 4.2 `dplyr`

```r
library(dplyr)

first.dat %>%
  group_by(gender) %>%
  summarise(mean_pay = mean(hourpay),
            sd_pay   = sd(hourpay),
            n        = n())
```

`dplyr` is more verbose but scales better when you need multiple summaries. Use
whichever you prefer; just be consistent within a project.

---

## 5. Column and Row Summaries

```r
summary(over18)
colMeans(over18)          # mean of every column (all must be numeric)
head(rowSums(over18))     # sum across columns for each row
```

`colMeans()` will fail if any column is non-numeric (e.g., a character or
factor). In such cases, select numeric columns first:
`colMeans(over18[, sapply(over18, is.numeric)])`.

---

## 6. Plotting with `ggplot2`

### 6.1 Scatterplot with regression lines

```r
library(ggplot2)

p1 <- ggplot(over18, aes(x = age, y = log(hourpay), color = factor(gender))) +
  geom_point(size = 0.6, alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_manual(values = c("0" = "steelblue", "1" = "firebrick"),
                      labels = c("Female", "Male"),
                      name = "Gender") +
  labs(title = "log(Hourly Pay) by Age and Gender",
       x = "Age", y = "log(Hourly Pay)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

p1
```

A few things to notice:

- We plot `log(hourpay)` rather than `hourpay`. Earnings data are typically
  right-skewed; the logarithm compresses the tail and makes the relationship
  with age more nearly linear. You will formalize this idea when we discuss
  *transformations* later in the course.
- `geom_smooth(method = "lm")` overlays the ordinary least squares fit. Each
  color group gets its own line. This is a preview of what multiple regression
  with an interaction term does — it estimates separate slopes and intercepts
  for each group.

### 6.2 Jittered scatterplot

When many points share the same $x$ value (common with integer-valued age),
they stack on top of each other. `geom_jitter()` adds small random noise to
reduce overplotting:

```r
p2 <- ggplot(over18, aes(x = age, y = log(hourpay), color = factor(gender))) +
  geom_jitter(size = 0.4, alpha = 0.4, width = 0.3) +
  scale_color_manual(values = c("0" = "steelblue", "1" = "firebrick"),
                      name = "Gender") +
  labs(title = "Jittered Version",
       x = "Age", y = "log(Hourly Pay)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
```

### 6.3 Arranging multiple plots

```r
library(gridExtra)
grid.arrange(p1, p2, ncol = 1)
```

---

## 7. Saving Data

```r
write.csv(over18, "Over18_age_hourpay.csv", row.names = FALSE)
```

Typically, it is best to set `row.names = FALSE` unless you have a reason not to. The default
(`TRUE`) writes an extra unnamed column of row numbers that will appear as
`X` when you re-import the file — a perennial source of phantom columns in
student projects.

---

## 8. One-Sample and Two-Sample $t$-Tests: A Simulation Study

In Lecture 1 you encountered the $z$-test and the $t$-test as tools for
comparing means. In this section, let's try some simulations to visualize these.

### 8.1 The scenario

A government labor survey reports that the national average starting salary
for software engineers is \$68,000. Two firms — **PixelForge** and
**VoxelSystems** — are rumored to pay differently. You have salary data from a
sample of employees at each firm.

The question has two parts:

1. **One-sample test.** Does PixelForge's mean salary differ from the national
   average of \$68,000?
2. **Two-sample test.** Do PixelForge and VoxelSystems differ from each other?

We will simulate the data so that we *know the truth* — this is the great
advantage of simulation. When you know the data-generating process, you can
check whether your statistical procedure gives the right answer.

### 8.2 Generating the data

```r
set.seed(42)

n_pf <- 30    # PixelForge sample size
n_vs <- 300   # VoxelSystems sample size

pixel_forge    <- rnorm(n_pf, mean = 71000, sd = 5000)
voxel_systems  <- rnorm(n_vs, mean = 76000, sd = 6500)
```

Note the asymmetry: 30 observations for PixelForge, 300 for VoxelSystems. 
This allows us observe how sample size affects the precision of our
estimates and the power of our tests.

### 8.3 visualizing the two samples

Before any formal testing, *look at the data*.

```r
salary_df <- data.frame(
  salary  = c(pixel_forge, voxel_systems),
  company = factor(rep(c("PixelForge", "VoxelSystems"), c(n_pf, n_vs)))
)

ggplot(salary_df, aes(x = company, y = salary, fill = company)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.5) +
  scale_fill_manual(values = c("PixelForge" = "steelblue",
                                "VoxelSystems" = "firebrick")) +
  geom_hline(yintercept = 68000, linetype = "dashed", color = "gray40") +
  annotate("text", x = 0.55, y = 68500,
           label = "National avg: $68k", hjust = 0, size = 3.8,
           color = "gray30") +
  labs(title = "Starting Salaries by Firm",
       y = "Salary ($)", x = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
```

The boxplot gives you an immediate visual answer: VoxelSystems appears to pay
substantially more. PixelForge appears to be above \$68,000, but the spread is
large relative to the gap. The formal tests below quantify this.

### 8.4 Overlaid density plots

Boxplots compress the shape of the distribution into five numbers. Density
plots preserve the shape:

```r
ggplot(salary_df, aes(x = salary, fill = company)) +
  geom_density(alpha = 0.45, color = NA) +
  geom_vline(xintercept = 68000, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = c("PixelForge" = "steelblue",
                                "VoxelSystems" = "firebrick")) +
  annotate("text", x = 68000, y = Inf, label = "  National avg",
           hjust = 0, vjust = 1.5, size = 3.5, color = "gray30") +
  labs(title = "Salary Distributions",
       x = "Salary ($)", y = "Density") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", legend.title = element_blank())
```

Notice how much narrower the VoxelSystems density is. Is it because their salaries are less variable? 

No. It is because we have 300 observations, so the *kernel density estimate* is more precise. The
underlying population of VoxelSystems salaries is actually *more* dispersed.
Keep the distinction between sample precision and population variability
in mind.

### 8.5 One-sample $t$-test

**Question.** Is PixelForge's mean salary significantly different from the
national average of \$68,000?

**Formally:**

$$
H_0: \mu_{\text{PF}} = 68{,}000 \quad \text{vs} \quad H_1: \mu_{\text{PF}} \neq 68{,}000
$$

The test statistic is:

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

where $\bar{x}$ is the sample mean, $s$ is the sample standard deviation, and
$n$ is the sample size. Under $H_0$, this follows a $t$-distribution with
$n - 1$ degrees of freedom.

```r
t_one <- t.test(pixel_forge, mu = 68000)
t_one
```

**Reading the output.** The key quantities are:

| Field | Meaning |
|---|---|
| `t` | The observed test statistic |
| `df` | Degrees of freedom ($n - 1 = 29$) |
| `p-value` | Probability of observing a $t$ at least this extreme under $H_0$ |
| `95 percent confidence interval` | Range of plausible values for the true mean |
| `sample estimates: mean of x` | $\bar{x}$ |

If the p-value is below 0.05, we reject $H_0$ at the 5% level. If the 95%
confidence interval does not contain 68,000, this is equivalent — the two
criteria always agree.

**Interpretation.** We simulated PixelForge with a true mean of 71,000, which
*is* different from 68,000. Whether the test detects this depends on the sample
size and the signal-to-noise ratio. With only 30 observations and a standard
deviation of 5,000, the standard error is $5000 / \sqrt{30} \approx 913$. The
gap of 3,000 is about 3.3 standard errors — detectable, but not overwhelmingly
so. A smaller true difference, or more noise, would produce a non-significant
result despite a real effect.

### 8.6 visualizing the one-sample test

```r
t_obs <- t_one$statistic
df_val <- t_one$parameter

t_grid <- seq(-5, 5, length.out = 500)
t_dens <- dt(t_grid, df = df_val)
t_df   <- data.frame(t = t_grid, density = t_dens)

crit <- qt(0.975, df = df_val)  # critical value for two-sided test

ggplot(t_df, aes(x = t, y = density)) +
  geom_line(linewidth = 0.7) +
  geom_area(data = subset(t_df, t <= -crit),
            fill = "firebrick", alpha = 0.3) +
  geom_area(data = subset(t_df, t >= crit),
            fill = "firebrick", alpha = 0.3) +
  geom_vline(xintercept = t_obs, color = "steelblue",
             linewidth = 0.8, linetype = "dashed") +
  annotate("text", x = t_obs + 0.25, y = max(t_dens) * 0.6,
           label = paste0("t = ", round(t_obs, 2)),
           color = "steelblue", hjust = 0, size = 4.2) +
  labs(title = "One-Sample t-Test: PixelForge vs National Average",
       subtitle = bquote(H[0]*": "*mu == 68000*"    df = "*.(df_val)),
       x = "t", y = "Density") +
  theme_minimal(base_size = 13)
```

### 8.7 Two-sample $t$-test

**Question.** Do PixelForge and VoxelSystems have different mean salaries?

$$
H_0: \mu_{\text{PF}} = \mu_{\text{VS}} \quad \text{vs} \quad H_1: \mu_{\text{PF}} \neq \mu_{\text{VS}}
$$

```r
t_two <- t.test(pixel_forge, voxel_systems)
t_two
```

R defaults to **Welch's $t$-test**, which does *not* assume equal variances
across groups. This is typically the right choice. The classical
equal-variance (pooled) test is a special case that you can request with
`var.equal = TRUE`.

**Interpretation.** The true means are 71,000 and 76,000 — a gap of 5,000.
Given the sample sizes (30 and 300), expect a highly significant result. The
confidence interval for the *difference* in means tells you the range of
plausible values for $\mu_{\text{PF}} - \mu_{\text{VS}}$.

### 8.8 Why sample size matters: a power demonstration

A test can fail to reject $H_0$ even when $H_0$ is false — this is a
**Type II error**. The probability of *correctly* rejecting a false $H_0$ is
called **power**. Power depends on three things: the true effect size, the
noise level, and the sample size.

We demonstrate this by running the one-sample test thousands of times at
different sample sizes, each time drawing fresh data from the same true
distribution.

```r
set.seed(211)

true_mean  <- 71000
null_mean  <- 68000
true_sd    <- 5000
sample_sizes <- c(10, 20, 30, 50, 100, 200)
n_sims     <- 2000

power_results <- data.frame(n = integer(), rejected = logical())

for (n in sample_sizes) {
  p_vals <- replicate(n_sims, {
    x <- rnorm(n, mean = true_mean, sd = true_sd)
    t.test(x, mu = null_mean)$p.value
  })
  power_results <- rbind(power_results,
                         data.frame(n = n, rejected = p_vals < 0.05))
}

# Compute power (proportion of rejections) at each sample size
power_summary <- aggregate(rejected ~ n, data = power_results, FUN = mean)
colnames(power_summary)[2] <- "power"

ggplot(power_summary, aes(x = n, y = power)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  geom_point(color = "steelblue", size = 3) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "gray50") +
  annotate("text", x = max(sample_sizes), y = 0.82,
           label = "80% power", hjust = 1, color = "gray40", size = 3.8) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  labs(title = "Power of the One-Sample t-Test by Sample Size",
       subtitle = bquote("True "*mu == 71000*", "*H[0]*": "*mu == 68000*", "*sigma == 5000),
       x = "Sample Size (n)", y = "Power (proportion of rejections)") +
  theme_minimal(base_size = 13)
```

**What to observe.** With $n = 10$, the test rejects only a fraction of the
time — you would frequently fail to detect that PixelForge pays above the
national average, even though it genuinely does. By $n = 50$, power approaches
or exceeds the conventional 80% threshold. 

In a real study, an underpowered test wastes resources and produces uninformative
results.

### 8.9 visualizing the sampling distribution

To understand *why* power increases with $n$, it helps to see the sampling
distribution of $\bar{x}$ under both $H_0$ and $H_1$.

```r
# Sampling distributions for n = 30
se_30 <- true_sd / sqrt(30)

x_grid <- seq(63000, 77000, length.out = 500)

samp_df <- data.frame(
  x = rep(x_grid, 2),
  density = c(dnorm(x_grid, mean = null_mean, sd = se_30),
              dnorm(x_grid, mean = true_mean, sd = se_30)),
  distribution = rep(c("Under H0 (mu = 68k)", "Truth (mu = 71k)"), each = 500)
)

# Critical values on the salary scale
crit_upper <- null_mean + qt(0.975, df = 29) * se_30
crit_lower <- null_mean - qt(0.975, df = 29) * se_30

ggplot(samp_df, aes(x = x, y = density, color = distribution)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = c(crit_lower, crit_upper),
             linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Under H0 (mu = 68k)" = "gray50",
                                  "Truth (mu = 71k)" = "steelblue")) +
  annotate("text", x = crit_upper + 200, y = max(samp_df$density) * 0.5,
           label = "Rejection\nboundary", color = "gray40",
           size = 3.5, hjust = 0) +
  labs(title = "Sampling Distribution of the Sample Mean (n = 30)",
       x = "Sample Mean Salary ($)", y = "Density",
       color = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")
```

The gray curve is the distribution of $\bar{x}$ *if $H_0$ were true*. The blue
curve is the distribution of $\bar{x}$ given the true mean. Power is the area
of the blue curve that falls beyond the dashed rejection boundaries. As $n$
increases, both curves become narrower (the standard error shrinks), and they
overlap less — hence, power increases.

---

## 9. Exercises

1. **Data import debugging.** Load `age_hourpay.csv` using `read.table()` with
   the argument `sep = ","`. Compare the result to `read.csv()`. Are they
   identical? Use `identical()` to check, and if they differ, investigate why.

2. **Subsetting.** From `first.dat`, create a subset of women (`gender == 0`)
   aged 25–40 with hourly pay below £30. How many observations remain? Compute
   the mean and standard deviation of `hourpay` in this subset.

3. **Aggregation.** Using either `aggregate()` or `dplyr`, compute the median
   hourly pay by gender for adults (age $\geq$ 18). Produce a grouped boxplot
   of `log(hourpay)` by gender using `ggplot2`.

4. **One-sample test.** Using the `over18` data, test whether the mean hourly
   pay is significantly different from £12. Interpret the output: state the
   null hypothesis, the test statistic, the p-value, and your conclusion.

5. **Power exploration.** Modify the simulation in Section 8.8: keep the sample
   sizes the same, but reduce the true mean to 69,000 (so the effect size
   shrinks from 3,000 to 1,000). How does the power curve change? What sample
   size is now needed to reach 80% power? What does this tell you about the
   relationship between effect size and required sample size?

6. **Two-sample test with real data.** Split `over18` into two groups by
   gender. Perform a two-sample $t$-test on `log(hourpay)`. Before running the
   test, produce overlapping density plots of `log(hourpay)` by gender. Does
   the visual impression match the test result?

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer.
- R Core Team. `t.test()` documentation: `t.test`
