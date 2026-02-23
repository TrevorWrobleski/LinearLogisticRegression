# Reading Week: Set theory, ceteris paribus, interaction geometry

**ST211 — Linear and Logistic Regression**
Supplemental Material for Reading Week

---

The three short notes below are not tied to a single workshop. They are
some important concepts you need for the course in general, and reading week 
is a good time to soldify this understanding. 
The first — set theory — underpins probability theory. It is also the basis of 
the logical operations you have been performing in R since Week 1.
The second — *ceteris paribus* — examines an important concept behind interpreting
regression coefficients, and traces it back to a Franciscan monk worried about the ethics of pricing. 
The third — the geometry of interaction terms — asks why we *multiply* variables together to
model interactions, and what that multiplication looks like in space.

You don’t need R for this workshop, but I’ve included a few snippets that might help with intuition.

---

# I. Sets & Logic

## 1. Why Sets Matter for Statistics

Every time you write a command like, `first.dat[first.dat$age >= 18, ]` in R, you are
performing a set operation. You are identifying the *set* of row indices for
which a condition holds, and extracting the corresponding observations. When
you write `subset(first.dat, age >= 18 & hourpay <= 50)`, you are computing
the *intersection* of two sets. When you compute a p-value, you are computing
the probability of an *event* — which is a set of outcomes in a sample space.

Set theory is the order behind statistics.

## 2. Definitions

A **set** is a collection of distinct objects, called **elements** or
**members**. We write $a \in A$ to mean "the element $a$ belongs to the set
$A$," and $a \notin A$ to mean it does not.

Sets can be defined by listing their elements:

$$
A = \{1, 3, 5, 7, 9\}
$$

or by stating a rule:

$$
A = \{x \in \mathbb{Z} : x \text{ is odd and } 1 \leq x \leq 9\}
$$

The colon reads "such that." Both definitions describe the same set.

A few important sets to have in your vocabulary:

- The **empty set** $\emptyset = \{\}$ contains no elements.
- The **universal set** $\Omega$ contains all objects under consideration. In
  probability, $\Omega$ is the sample space — the set of all possible
  outcomes of an experiment.
- A set $A$ is a **subset** of $B$, written $A \subseteq B$, if every element
  of $A$ is also in $B$.

## 3. Three Operations

Let $A$ and $B$ be subsets of a universal set $\Omega$.

### 3.1 Union ($A \cup B$)

The **union** of $A$ and $B$ is the set of elements belonging to $A$ *or* $B$
(or both):

$$
A \cup B = \{x \in \Omega : x \in A \text{ or } x \in B\}
$$

In R, this is the `|` operator. When you write

```r
subset(first.dat, age < 18 | hourpay > 100)
```

you are selecting observations in the union of $\{\text{age} < 18\}$ and
$\{\text{hourpay} > 100\}$.

### 3.2 Intersection ($A \cap B$)

The **intersection** of $A$ and $B$ is the set of elements belonging to $A$
*and* $B$ simultaneously:

$$
A \cap B = \{x \in \Omega : x \in A \text{ and } x \in B\}
$$

In R, this is the `&` operator:

```r
subset(first.dat, age >= 18 & hourpay <= 50)
```

You also used `intersect()` explicitly in Workshop 5 to find observations
flagged by *both* leverage and DFFITS:

```r
intersect(result$Leverage, result$DFFITS)
```

That was a set intersection on index vectors.

If $A \cap B = \emptyset$ — the two sets share no elements — we say $A$ and
$B$ are **disjoint** (or **mutually exclusive**). 

### 3.3 Complement ($A^c$)

The **complement** of $A$ is the set of everything in $\Omega$ that is *not*
in $A$:

$$
A^c = \{x \in \Omega : x \notin A\}
$$

In R, this is negation with `!`:

```r
subset(first.dat, !(age < 18))
```

And negative indexing achieves the same thing on row numbers:

```r
mtcars[-heaviest, ]  # all rows EXCEPT the heaviest car
```

### 3.4 Set Difference ($A \setminus B$)

The **difference** $A \setminus B$ contains elements in $A$ that are *not*
in $B$:

$$
A \setminus B = \{x \in \Omega : x \in A \text{ and } x \notin B\} = A \cap B^c
$$

Note that $A \setminus B \neq B \setminus A$ in general. In R:

```r
setdiff(flagged_leverage, flagged_cooks)
```

gives you observations with high leverage that are *not* flagged by Cook's
distance — exactly the "high leverage, small residual" points from Workshop 5
(Scenario C) that sit on the line and reinforce the trend rather than
distorting it.

## 4. De Morgan's Laws

Two identities, attributed to the 19th-century British mathematician
**Augustus De Morgan** (1806–1871), are useful in understanding
unions and intersections:

$$
(A \cup B)^c = A^c \cap B^c
$$

$$
(A \cap B)^c = A^c \cup B^c
$$

In words: the complement of a union is the intersection of the complements,
and the complement of an intersection is the union of the complements. 
They frequently show up when you negate a compound condition.

Suppose you have written:

```r
subset(first.dat, age >= 18 & hourpay <= 50)
```

and you now want the *opposite* subset — everyone who was **not** selected.
You might write `!(age >= 18 & hourpay <= 50)`. De Morgan's second law tells
you this is equivalent to:

```r
subset(first.dat, age < 18 | hourpay > 50)
```

The negation of an "and" becomes an "or," and each individual condition
flips. 

The laws hold in full generality, including for more than two sets:

$$
\left(\bigcup_{i=1}^{k} A_i\right)^c = \bigcap_{i=1}^{k} A_i^c, \qquad \left(\bigcap_{i=1}^{k} A_i\right)^c = \bigcup_{i=1}^{k} A_i^c
$$

In probability, De Morgan's laws let you move between statements
about events. For example, $P(\text{neither } A \text{ nor } B) = P(A^c \cap B^c) = P((A \cup B)^c) = 1 - P(A \cup B)$. Each step is an application
of either a De Morgan identity or the complement rule.

## 5. Visualizing Sets: Venn Diagrams

The Venn diagram, introduced by John Venn in 1880 (though Euler used similar
diagrams a century earlier), is the standard visualization. You have certainly
seen these before. What is worth noting is how directly they map to
statistical ideas.

Consider two events $A$ and $B$ within a sample space $\Omega$:

```
┌──────────────────────────────────────────┐
│  Ω                                       │
│        ┌───────────┐                     │
│    ┌───┤───┐       │                     │
│    │   │   │       │                     │
│    │ A │A∩B│   B   │                     │
│    │   │   │       │                     │
│    └───┤───┘       │                     │
│        └───────────┘                     │
│                              (A∪B)^c     │
└──────────────────────────────────────────┘
```

Each region corresponds to an observable event:

| Region | Set notation | Meaning |
|--------|-------------|---------|
| Left crescent | $A \setminus B$ | $A$ occurs but $B$ does not |
| Overlap | $A \cap B$ | Both $A$ and $B$ occur |
| Right crescent | $B \setminus A$ | $B$ occurs but $A$ does not |
| Outside both | $(A \cup B)^c$ | Neither occurs |

The entire rectangle is $\Omega$, and the areas (if drawn proportionally)
represent probabilities.

## 6. The Bridge to Probability

The reason set theory matters for this course is that **probability is defined
on sets.** A probability function $P$ assigns a number between 0 and 1 to
subsets of $\Omega$ (events), subject to three axioms (Kolmogorov, 1933):

1. $P(A) \geq 0$ for any event $A$.
2. $P(\Omega) = 1$.
3. If $A$ and $B$ are disjoint ($A \cap B = \emptyset$), then
   $P(A \cup B) = P(A) + P(B)$.

From these three axioms, everything else follows. The most immediately useful
result is the **addition rule** for events that are *not* disjoint:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

We subtract $P(A \cap B)$ because the intersection has been counted twice —
once in $P(A)$ and once in $P(B)$. The Venn diagram makes this visually
obvious: the overlap region is covered by both circles.

If you find this interesting, please search for "Set Theory" and you fill find some 
good resources online to explore this topic, especially necessary if you are interested
in higher level statistics (especially 500-level courses).

### An example you can verify in R

Recall the simulated heights from Workshop 1 ($n = 200$, $\mu = 168$,
$\sigma = 10$). Define:

- $A$ = the event that a randomly selected student is taller than 178 cm
- $B$ = the event that a randomly selected student is taller than 175 cm

Clearly $A \subset B$ (anyone taller than 178 is certainly taller than 175),
so $A \cap B = A$ and $P(A \cup B) = P(B)$. Boring case. Now try:

- $A$ = taller than 175 cm
- $B$ = shorter than 160 cm

These events are disjoint ($A \cap B = \emptyset$ — no one can be
simultaneously above 175 and below 160), so the addition rule simplifies to
$P(A \cup B) = P(A) + P(B)$:

```r
# Theoretical probabilities from the known distribution
pA <- 1 - pnorm(175, 168, 10)
pB <- pnorm(160, 168, 10)
pA + pB  # = P(A ∪ B) since disjoint
```

Now consider a non-trivial case:

- $A$ = taller than 175 cm
- $B$ = between 160 and 180 cm

These overlap (heights between 175 and 180 are in both). The addition rule
gives the correct answer; simply summing $P(A) + P(B)$ would overcount.

```r
pA <- 1 - pnorm(175, 168, 10)
pB <- pnorm(180, 168, 10) - pnorm(160, 168, 10)
pAB <- pnorm(180, 168, 10) - pnorm(175, 168, 10)  # the intersection

pA + pB - pAB  # addition rule
```

## 7. Why This Matters Going Forward

When we get to logistic regression later in the course, we will be modeling
probabilities directly. The events we care about — "this patient develops the
condition," "this applicant defaults on the loan" — are sets in a sample
space, and the rules governing them are the rules of set theory. The
conditional probability $P(A \mid B) = P(A \cap B) / P(B)$ is a ratio of set measures. 
If the language of sets is comfortable now, the transition to probabilistic modeling 
will be more natural.

Set theory also clarifies something about hypothesis testing that is easy to
confuse. When you compute a p-value, you are computing $P(T \in R \mid H_0)$
— the probability that the test statistic $T$ falls in the rejection region
$R$, given that the null hypothesis is true. The rejection region is a set
(typically the tails of a distribution, as we visualized in Workshops 1 and
2). The decision to reject or not is a question of set membership: does the
observed value belong to $R$, or to $R^c$?

---

# II. *Ceteris Paribus*

## 1. The Phrase

*Ceteris paribus* is Latin for "with other things being equal" — or, more
colloquially, "all else held constant." Every time you read or write a
sentence of the form "a one-unit increase in $X_1$ is associated with a
$\hat{\beta}_1$-unit change in $Y$, *holding all other predictors constant*,"
you are invoking *ceteris paribus*.

But what does it mean to "hold constant" variables that, in the real world,
move together? And where did this idea come from? 

## 2. Origins: Just Price (1295)

In the 13th century, questions aboutvalue, pricing, and exchange were 
matters of **moral theology**, they fell under the study of sin and virtue. 
The central concern was the *justum pretium*, the **just price**: '
at what price could a merchant sell a good
without committing the sin of avarice?

The Franciscan theologian **Petrus Olivi** (1248–1298) in his treatise 
*De emptionibus et venditionibus* ("On Buying and Selling") argued that 
economic value was not intrinsic to an object. Instead, it arose from three sources:

- *Virtuositas* — the objective quality or usefulness of the good
- *Raritas* — its scarcity
- *Complacibilitas* — the subjective desirability to the buyer

This was an early form of **subjective value theory**. 
If prices naturally fluctuate with scarcity and
demand, how do you determine whether a merchant is charging a "just" price
or simply responding to market conditions?

You had to compare it to other prices **under similar conditions** — the same
time, the same place, the same scarcity. This allowed the theologian to separate legitimate market variation
from exploitation. A merchant who charged more during a famine was
not necessarily greedy if scarcity had changed. But a merchant who charged
more while *ceteris paribus* had committed an injustice.


## 3. Formalization: Alfred Marshall and the Pound (1890)

The phrase became a scientific tool through the work of **Alfred Marshall**
(1842–1924), the English economist whose *Principles of Economics* (1890)
is one of the founding texts of modern economic analysis.

In a real economy, prices affect other prices else simultaneously. 
The price of bread depends on the wages of
bakers, which depend on the price of housing, which depends on interest
rates, which depend on government policy, and so on. Analyzing all of these
interconnections at once — what economists call **general equilibrium** — was
(and remains)… difficult.

Marshall's solution was **partial equilibrium analysis**: study one market or
one relationship at a time, temporarily freezing everything else. He addressed this by assuming  
the variables not under consideration were locked away in an enclosure, 
fenced in by the phrase *ceteris paribus*, so that they cannot
disturb the analysis. A demand curve shows how the quantity demanded of a good varies with its price,
*ceteris paribus* — holding income, preferences, and the prices of
substitutes fixed. 

## 4. Multiple Regression

In a multiple regression model

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon,
$$

the coefficient $\beta_1$ is the *ceteris paribus* effect of $X_1$: the
expected change in $Y$ per unit increase in $X_1$, **holding $X_2$ fixed**.
The partial derivative

$$
\frac{\partial\, E[Y]}{\partial X_1} = \beta_1
$$

literally holds $X_2$ constant while varying $X_1$.

In simple regression (one predictor), there is nothing to hold constant, and
$\beta_1$ captures the *total* association between $X_1$ and $Y$ — including
any indirect pathways through omitted variables. This is the key difference
between simple and multiple regression, and it is the reason we move to MLR:
**to make the *ceteris paribus* clause meaningful by specifying what,
exactly, is being held constant.**

### When the clause breaks down

The *ceteris paribus* interpretation is only as good as the set of variables
you hold constant. If there exists an important variable $X_3$ that you have
*not* included in the model, but which is correlated with both $X_1$ and
$Y$, then $\beta_1$ absorbs the effect of $X_3$ and no longer estimates the
isolated effect of $X_1$. This is **omitted variable bias**  and it is a
failure of *ceteris paribus*.

## 5. A Case Study: Yule and the Cause of Poverty (1899)

In 1899, the statistician **George Udny Yule** published *An Investigation
into the Causes of Changes in Pauperism in England*, attempting to settle a
Victorian debate: under the English Poor Law, welfare could be
administered in two ways: **In-Door Relief** (the poor had to enter a
workhouse — harsh conditions intended as a deterrent) and **Out-Door
Relief** (cash or food payments while living at home). Critics argued that
Out-Door Relief was a moral hazard becasue it made it too easy to get help, and more
people would choose to remain poor.

Yule collected data across English districts and estimated a model of the
form like this (numbers not exact):

$$
\hat{Y} = \alpha + 0.76\,X_1 + 0.33\,X_2 - 0.32\,X_3
$$

where $Y$ was the percentage change in the poverty rate, $X_1$ was the
change in the Out-Relief ratio, $X_2$ was the change in the proportion of
elderly, and $X_3$ was population change.

The coefficient $0.76$ was interpreted *ceteris paribus*: holding the age
structure and population constant, a 10% increase in Out-Relief was
associated with a 7.6% increase in poverty. Yule concluded that
"administration is the principal factor in pauperism" — generous welfare
policies *caused* poverty.

While this methodology was particularly impressive at the time, I encourage you
to reflect on the methodology and note its strengths and weaknesses and how you 
might approach the question

---

# III. Multiplication in Interaction Terms

## 1. The Question

When we introduced interaction terms in Workshop 4, we wrote models of the
form:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \cdot X_2) + \varepsilon
$$

and noted that $\beta_3$ allows the effect of $X_1$ to depend on the level
of $X_2$. But a natural question remains: **why multiplication?** If we
want to capture a new kind of relationship, why not add a third variable, or
use some other operation? Let's think through this.

## 2. What Addition Does

Consider the additive model:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon
$$

The effect of a one-unit increase in $X_1$ on $E[Y]$ is $\beta_1$, 
regardless of the value of $X_2$. The two predictors each make their own
separate, constant contribution. If you think of $X_1$ and $X_2$ as two dials
on a machine, the additive model says that turning one dial has nothing to do
with the position of the other.

Adding a third variable $X_3$ to the model would give it its own dial —
another independent, constant contribution. Addition, by its algebraic
nature, keeps effects separate.

## 3. What Multiplication Does

Now suppose the effect of $X_1$ genuinely depends on $X_2$. You need a
term that *links* the two. Multiplication is the natural candidate, because
it creates exactly the dependence we want. Write the interaction model and
group terms:

$$
Y = \beta_0 + (\beta_1 + \beta_3 X_2)\, X_1 + \beta_2 X_2 + \varepsilon
$$

The expression in parentheses — $\beta_1 + \beta_3 X_2$ — is the
**effective slope** of $X_1$, and it is no longer a fixed number. It is a
function of $X_2$. When $X_2$ is large and $\beta_3 > 0$, the effect of
$X_1$ is amplified. When $X_2$ is small, it is dampened. The product
$X_1 \cdot X_2$ is what makes this possible.

Note that the rearrangement is symmetric. We could equally write:

$$
Y = \beta_0 + \beta_1 X_1 + (\beta_2 + \beta_3 X_1)\, X_2 + \varepsilon
$$

Now $X_1$ modifies the slope of $X_2$. The interaction is mutual.

### A concrete example

Suppose we model a student's exam score ($Y$) from hours of weekly exercise
($X_1$) and hours of sleep the night before the exam ($X_2$):

$$
\hat{Y} = 40 + 2\,X_1 + 3\,X_2 + 0.5\,X_1 X_2
$$

The effective slope of exercise is $2 + 0.5\,X_2$. For a student who slept
4 hours: the benefit of one more hour of exercise is $2 + 0.5(4) = 4$
points. For a student who slept 8 hours: $2 + 0.5(8) = 6$ points.


## 4. The Geometry: Flat Planes and Twisted Surfaces

### The additive model is a flat plane

The additive model $E[Y] = \beta_0 + \beta_1 X_1 + \beta_2 X_2$ describes
a **plane** in three-dimensional $(X_1, X_2, Y)$ space. If you stand on this
plane and walk in the $X_1$ direction, you always climb (or descend) at the
same rate, regardless of where you are along $X_2$. The plane tilts, but it
does not twist.

If you slice this plane at a fixed value of $X_2$ — say, $X_2 = 5$ — you
get a straight line in the $(X_1, Y)$ plane:

$$
E[Y] = (\beta_0 + 5\beta_2) + \beta_1 X_1
$$

Now slice at $X_2 = 10$:

$$
E[Y] = (\beta_0 + 10\beta_2) + \beta_1 X_1
$$

The intercept has changed, but the slope ($\beta_1$) is identical. The two
lines are **parallel**. This is the geometric signature of an additive model:
all slices at different values of $X_2$ yield parallel lines in $X_1$.

### The interaction model is a twisted surface

The interaction model $E[Y] = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2$ 
describes a **ruled quadric** — more specifically, a
**hyperbolic paraboloid** when $\beta_3 \neq 0$. You can visualize this by
imagining a flat sheet of paper: hold two diagonally opposite corners and
push them up, while pushing the other two corners down. The sheet twists. It
is no longer flat, and the steepness of the surface in one direction depends
on where you stand in the other direction.

Slice the interaction surface at $X_2 = 5$:

$$
E[Y] = (\beta_0 + 5\beta_2) + (\beta_1 + 5\beta_3)\, X_1
$$

Slice at $X_2 = 10$:

$$
E[Y] = (\beta_0 + 10\beta_2) + (\beta_1 + 10\beta_3)\, X_1
$$

Now the slopes are different: $\beta_1 + 5\beta_3$ versus
$\beta_1 + 10\beta_3$. The lines are **no longer parallel**. They converge,
diverge, or even cross depending on the sign and magnitude of $\beta_3$.

**Non-parallel lines are the geometric signature of an interaction.**

This is precisely what you saw in Workshop 4 when we compared the additive
and interaction models for `mpg ~ wt * am_factor`. The additive model
produced parallel regression lines for automatic and manual cars; the
interaction model allowed those lines to have different slopes. The
interaction coefficient $\beta_3$ measured the *twist* — the degree to which
the surface departs from flatness.

## 5. Seeing the Surface

For those of you who prefer to see this in code, try this: 

```r
library(ggplot2)

# Define the interaction model: Y = 40 + 2*X1 + 3*X2 + 0.5*X1*X2
grid <- expand.grid(X1 = seq(0, 10, length.out = 50),
                    X2 = seq(0, 10, length.out = 50))
grid$Y_additive   <- 40 + 2 * grid$X1 + 3 * grid$X2
grid$Y_interaction <- 40 + 2 * grid$X1 + 3 * grid$X2 + 0.5 * grid$X1 * grid$X2

# Contour plot: additive model
p1 <- ggplot(grid, aes(x = X1, y = X2, z = Y_additive)) +
  geom_contour_filled(bins = 12) +
  labs(title = "Additive Model (no interaction)",
       subtitle = "Parallel, evenly spaced contours",
       x = expression(X[1]), y = expression(X[2])) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

# Contour plot: interaction model
p2 <- ggplot(grid, aes(x = X1, y = X2, z = Y_interaction)) +
  geom_contour_filled(bins = 12) +
  labs(title = "Interaction Model",
       subtitle = "Contours twist — spacing depends on position",
       x = expression(X[1]), y = expression(X[2])) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

library(gridExtra)
grid.arrange(p1, p2, ncol = 2)
```

In the additive model, the contour lines are straight and evenly spaced —
the surface is a plane. In the interaction model, the contours twist: the
spacing between them is wider in one corner and tighter in the other. That
compression is the interaction at work. Where contours are tightly packed,
$Y$ is changing rapidly; where they are spread apart, $Y$ is changing
slowly. The fact that this rate of change depends on your position in both
$X_1$ and $X_2$ simultaneously is exactly what makes it an interaction.

## 6. Why Not Some Other Operation?

(This is optional, but instructive.)

You might wonder: could we use a different operation — say, $|X_1 - X_2|$
or $\max(X_1, X_2)$ — to capture the dependence? In principle, you could
construct many functions of two variables. But multiplication has a special
status in linear regression for three reasons:

1. **It keeps the model linear in the parameters.** The product $X_1 X_2$
   is a new variable, and the model is still linear in $\beta_0, \beta_1,
   \beta_2, \beta_3$. This means OLS still applies — no new estimation
   machinery is needed.

2. **It corresponds to the first-order correction for non-additivity.** If
   the true surface $E[Y \mid X_1, X_2]$ is smooth but not additive, a
   Taylor expansion around any point includes a cross-term
   $\frac{\partial^2 E[Y]}{\partial X_1 \partial X_2} \cdot X_1 X_2$ as the
   leading interaction. Multiplication captures the lowest-order departure
   from additivity.

3. **It has a clean interpretation.** The partial derivative
   $\partial E[Y] / \partial X_1 = \beta_1 + \beta_3 X_2$ is immediately
   readable: the effect of $X_1$ changes at a constant rate as $X_2$ varies.
   Other operations would produce effects that are harder to state in a
   sentence — and interpretability matters when you are presenting
   results.

## 7. Looking Ahead

When we return after reading week, our first topic will be transformations
and non-linear relationships — what to do when the straight-line assumption
fails. You have already seen hints of this: the non-linearity in the `mpg ~ disp`
example from Workshop 3, and the use of `log(hourpay)` in Workshop 2.
Transformations like the logarithm bend the predictor or the response so that
the relationship becomes approximately linear, and the geometry of interaction
terms helps explain why: a log transform changes the shape of the surface
your model describes, turning curves into planes. After that, we will move to
prediction and validation — how to assess whether your model generalizes
to new data rather than merely fitting the data it was trained on. Finally,
we will arrive at logistic regression, where the response variable is
binary (yes/no, success/failure) and the set theory from Part I of this
document will become directly relevant: we will be modeling the probability
of an event, which is a set in a sample space, and the ceteris paribus
interpretation from Part II will carry over to a new scale — the log-odds.
For now, the main takeaway is this: addition keeps effects separate;
multiplication lets them depend on each other. The additive model is a
plane; the interaction model is a twisted surface. And the twist is measured
by a single coefficient, $\beta_3$, which tells you how rapidly the slope of
one variable changes as you move along the other.

---

## References

- Olivi, P. (c. 1295). *De emptionibus et venditionibus*. Discussed in:
  Kaye, J. (1998). *Economy and Nature in the Fourteenth Century*. Cambridge
  University Press.
- Marshall, A. (1890). *Principles of Economics*. Macmillan.
- Yule, G.U. (1899). An investigation into the causes of changes in
  pauperism in England. *Journal of the Royal Statistical Society*, 62(2),
  249–295.
- Kolmogorov, A. (1933). *Grundbegriffe der Wahrscheinlichkeitsrechnung*.
  Springer.
- Venn, J. (1880). On the diagrammatic and mechanical representation of
  propositions and reasonings. *Philosophical Magazine*, 10(59), 1–18.
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press.
