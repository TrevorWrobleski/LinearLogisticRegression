# Workshop 1: Introduction to R — Vectors, Arrays, and Thinking Statistically

**ST211 — Linear and Logistic Regression**
Supplemental Material for the workshop

---

## 1. Orientation: Your Working Environment

Before doing anything in R, you need to know *where you are* on your machine.
R always operates inside a **working directory** — the folder it reads from and
writes to by default. This is the single most common source of "file not found"
errors for new users.

```r
# What directory am I in?
getwd()

# What files are in it?
dir()
```

If you need to change directory:

```r
# setwd("path/to/your/folder")
```

> **Tip.** In RStudio, you can also set the working directory via
> *Session → Set Working Directory → Choose Directory*. For reproducibility,
> however, it's helpful to record `setwd()` at the top of your script.

---

## 2. Variables and Basic Operations

In R you assign values to names using
the `<-` operator.

```r
new.var <- 1
new.var
```

### 2.1 Mathematical functions

R has the standard mathematical functions built in. The exponential function
$e^x$ is called with `exp()`:

```r
exp(new.var)   # e^1 ≈ 2.718
exp(1)         # same thing
```

Re-assignment overwrites the previous value without warning:

```r
new.var <- 5
new.var^2      # 25
```

### 2.2 Viewing your workspace

To see every object you have created in your current session:

```r
ls()
```

This is the programmatic equivalent of the *Environment* pane in RStudio. It
becomes useful when you are debugging and need to check what exists.

---

## 3. Vectors

A **vector** is the fundamental data structure in R. Almost everything in R is a
vector or is built from vectors. You create one with `c()` ("combine"/"concatonate"):

```r
first.vector <- c(1, 7, 11, 43)
first.vector
is.vector(first.vector)
```

### 3.1 Vectorised arithmetic

One of R's most powerful features is that arithmetic operations are
**vectorised** — they apply element-by-element without explicit loops.

```r
first.vector + 1    # adds 1 to every element
first.vector * 2    # multiplies every element by 2
first.vector^2      # squares every element
```

If `first.vector` contained $n$
observations of a variable $X$, then `first.vector - mean(first.vector)`
instantly gives you the vector of deviations $X_i - \bar{X}$ — the first step
in computing variance, covariance, and nearly everything else.

You have to be careful with this though, because while it saves time, you also risk problems if there is a length mismatch. (Remember to always check 

```r
dim()
```

before running calculations.

### 3.2 Operations on two vectors

```r
second.vector <- c(1, 2, 1, 2)

first.vector + second.vector   # element-wise addition
first.vector^second.vector     # element-wise exponentiation
first.vector > second.vector   # element-wise logical comparison
```

When you write `first.vector > second.vector`, R returns a **logical vector** —
a vector of `TRUE`/`FALSE` values. Logical vectors are the mechanism behind
subsetting, filtering data frames, and conditional operations throughout the
course. Get comfortable with them now.

### 3.3 Comparing vectors: `identical()` vs `all.equal()`

```r
identical(first.vector, c(1, 7, 11, 43))      # TRUE
identical(c(0.3 - 0.1, 0), c(0.4 - 0.2, 0))  # FALSE (!)
all.equal(c(0.3 - 0.1, 0), c(0.4 - 0.2, 0))  # TRUE
```
What is going on here? 


It is a consequence of how computers represent decimal
numbers in binary (IEEE 754 floating-point arithmetic). The number $0.1$ has no
exact finite binary representation, so $0.3 - 0.1$ and $0.4 - 0.2$ differ at
roughly the 16th decimal place. `identical()` checks bit-for-bit equality;
`all.equal()` allows for machine-precision tolerance (by default,
$\approx 1.5 \times 10^{-8}$).

> When you later compare model fits,
> p-values, or likelihood values, never test floating-point numbers with `==`.
> Use `all.equal()` or test whether the absolute difference is below a tolerance.

---

## 4. Indexing and Subsetting Vectors

Indexing in R starts at **1** (not 0, as in Python or C).

```r
first.vector[2]         # 7 — the second element
first.vector[c(2, 4)]   # elements 2 and 4
first.vector[2:4]        # elements 2 through 4 (inclusive)
```

### 4.1 Removing elements

Use negative indices:

```r
first.vector[c(-1, -3)]  # removes elements 1 and 3; returns 7, 43
```

### 4.2 Logical subsetting

You can pass a logical condition directly inside the brackets:

```r
first.vector[first.vector > 8]  # returns 11, 43
```

If you need the *positions* rather than the values:

```r
which(first.vector > 8)                # returns 3, 4
first.vector[which(first.vector > 8)]  # same result as logical subsetting
```

Both approaches give the same elements, but `which()` is useful when you need
the indices themselves — for example, to modify specific entries or to cross-
reference positions across multiple vectors.

---

## 5. Arrays (Matrices)

An **array** is a multi-dimensional generalisation of a vector. A
two-dimensional array is simply a matrix.

```r
first.array <- array(first.vector, dim = c(2, 2))
first.array
```

Notice that R fills matrices **column-by-column** (column-major order):

```
     [,1] [,2]
[1,]    1   11
[2,]    7   43
```

If you supply a vector shorter than the array dimensions, R **recycles** it:

```r
second.array <- array(first.vector, dim = c(4, 2))
second.array
```

### 5.1 Array dimensions and navigation

```r
dim(first.array)          # 2 2

first.array[2, 2]         # element in row 2, column 2 → 43
first.array[2, ]          # entire row 2 → returns a vector
first.array[2, , drop = FALSE]  # row 2, but keep it as a 1×2 matrix
first.array[, 2]          # entire column 2
```

The `drop = FALSE` argument is worth remembering: by default, R
simplifies a single-row or single-column extraction to a vector, which can
cause unexpected behaviour in functions that expect a matrix.

### 5.2 Logical operations on arrays

```r
first.array > 8           # returns a logical matrix of the same shape
```

### 5.3 Constructing arrays by assignment

You can create an empty array and fill it column by column:

```r
another.array <- array(dim = c(4, 2))
another.array[, 1] <- c(1, 2)   # R recycles c(1,2) to fill 4 rows
another.array[, 2] <- c(3, 4)
another.array
```

### 5.4 Finding elements in arrays

```r
which(first.array > 8)                    # linear index (treats array as a single vector)
which(first.array > 8, arr.ind = TRUE)    # row and column indices
```

The `arr.ind = TRUE` option is useful for matrices and will become
important when you work with correlation matrices and need to locate, say, all
pairs of variables with correlation above some threshold.

---

## 6. Bringing It Together: A Small Simulation

We will simulate data from a known
distribution and inspect it to
build intuition for regression.

### 6.1 Simulating heights

Suppose the heights of LSE students follow a Normal distribution with mean
$\mu = 168$ cm and standard deviation $\sigma = 10$ cm. (Let's pause for a moment to think if this is reasonable.)
We can generate a
sample:

```r
set.seed(211)  # for reproducibility
n <- 200
heights <- rnorm(n, mean = 168, sd = 10)
```

Compute basic summaries using vectorised operations:

```r
mean(heights)
sd(heights)
range(heights)
```

### 6.2 Visualising the sample

You can do some plotting in base R, but I prefer to use `ggplot2`. Install it once with
`install.packages("ggplot2")`, then load it:

```r
library(ggplot2)

# Store data in a data frame (ggplot requires this)
height_df <- data.frame(height = heights)

ggplot(height_df, aes(x = height)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 25, fill = "steelblue", colour = "white", alpha = 0.8) +
  stat_function(fun = dnorm, args = list(mean = 168, sd = 10),
                linewidth = 0.8, colour = "firebrick") +
  labs(title = "Simulated Heights of 200 LSE Students",
       x = "Height (cm)", y = "Density") +
  theme_minimal(base_size = 13)
```

The histogram shows the empirical distribution of your sample; the red curve is
the *true* Normal density you sampled from. As $n$ grows, the histogram
converges to the curve — keep this idea in mind. We will discuss it further in subsequent classes.

### 6.3 The Q-Q plot

A **Q-Q (quantile-quantile) plot** compares the quantiles of your sample
against the quantiles of a theoretical distribution. If the data are well-
described by that distribution, the points lie on the diagonal.

```r
ggplot(height_df, aes(sample = height)) +
  stat_qq(colour = "steelblue", size = 1.5, alpha = 0.7) +
  stat_qq_line(colour = "firebrick", linewidth = 0.8) +
  labs(title = "Normal Q-Q Plot of Simulated Heights",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal(base_size = 13)
```

Because we drew from a Normal distribution, the points should track the line
closely. In later workshops, when you inspect *residuals* from a regression,
departures from this line will signal that something is wrong with your model.

### 6.4 A one-sample z-test by hand

Suppose we want to test whether this sample could plausibly have come from a
population with mean $\mu_0 = 170$ (instead of 168). The z-statistic is:

$$
z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

```r
mu_0 <- 170
z <- (mean(heights) - mu_0) / (10 / sqrt(n))
z

# Two-sided p-value
p_value <- 2 * pnorm(-abs(z))
p_value
```

Interpret the result: if $|z| > 1.96$, we reject $H_0$ at the 5% level.
Compare your computed p-value to 0.05.

### 6.5 Visualising the test

```r
z_grid <- seq(-4, 4, length.out = 500)
z_density <- dnorm(z_grid)
z_df <- data.frame(z = z_grid, density = z_density)

ggplot(z_df, aes(x = z, y = density)) +
  geom_line(linewidth = 0.8) +
  geom_area(data = subset(z_df, z <= -1.96),
            aes(x = z, y = density), fill = "firebrick", alpha = 0.35) +
  geom_area(data = subset(z_df, z >= 1.96),
            aes(x = z, y = density), fill = "firebrick", alpha = 0.35) +
  geom_vline(xintercept = z, colour = "steelblue", linewidth = 0.8,
             linetype = "dashed") +
  annotate("text", x = z + 0.3, y = 0.15, label = paste0("z = ", round(z, 2)),
           colour = "steelblue", size = 4.5, hjust = 0) +
  labs(title = "One-Sample z-Test",
       subtitle = expression(H[0]*": "*mu == 170*"  vs  "*H[1]*": "*mu != 170),
       x = "z", y = "Density") +
  theme_minimal(base_size = 13)
```

The shaded red regions are the rejection regions (each tail contains 2.5% of
the distribution). The dashed blue line is your observed test statistic. If it
falls inside a shaded region, you reject $H_0$.

---

## 7. Exercises

If you want to test yourself, try the following. They are ordered by
difficulty.

1. **Workspace basics.** Create three variables: `a <- 3`, `b <- 4`,
   `hyp <- sqrt(a^2 + b^2)`. Verify that `hyp` equals 5. Use `ls()` to
   confirm all three exist.

2. **Vector arithmetic.** Create a vector `x <- 1:20`. Without using any loop,
   compute the vector of squared deviations $(x_i - \bar{x})^2$. Then compute
   the sample variance manually using `sum()` and compare with `var(x)`.
   (Recall: $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.)

3. **Logical subsetting.** Using `heights` from Section 6, extract all
   observations above 185 cm. How many are there? What proportion of the sample
   do they represent? Does this proportion roughly match what you would expect
   from a $N(168, 10^2)$ distribution? (*Hint:* `1 - pnorm(185, 168, 10)`.)

4. **Array practice.** Create a $5 \times 3$ matrix where column 1 is
   `1:5`, column 2 is `(1:5)^2`, and column 3 is `(1:5)^3`. Use
   `which(..., arr.ind = TRUE)` to find all entries greater than 50.

5. **Simulation and testing.** Repeat the simulation in Section 6, but draw
   from a $t$-distribution with 5 degrees of freedom (use `rt()`), scaled to
   have approximately the same mean and spread. Produce the histogram with the
   Normal overlay. Does the Q-Q plot look different? Explain why.

6. **Two-sample test.** Generate two samples: `men <- rnorm(80, 175, 9)` and
   `women <- rnorm(120, 163, 8)`. Perform a two-sample t-test using `t.test()`
   and interpret the output. Then produce side-by-side boxplots using
   `ggplot2`.

---

## References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer.
- R Core Team. *An Introduction to R*.
  [https://cran.r-project.org/doc/manuals/r-release/R-intro.html](https://cran.r-project.org/doc/manuals/r-release/R-intro.html)
