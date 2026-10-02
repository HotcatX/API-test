# Assignment 1: Probability Solutions

Source: the six questions in *Assignment1-1.pdf*, section 3, “Problems.”

Notation: `A ∩ B` means both events occur, `A ∪ B` means at least one occurs, and `Aᶜ` means A does not occur. Conditional probability `P(A | B)` means the probability of A given that B occurred. All logarithms in Question 6 use base 2.

## Question 1: Independent events

Given `P(A) = 0.4` and `P(B) = 0.3`, with A and B independent.

### (a) Probability that both events occur

Independence gives the multiplication rule:

```text
P(A ∩ B) = P(A) × P(B)
         = 0.4 × 0.3
         = 0.12 = 3/25 = 12%.
```

### (b) Probability that at least one event occurs

The addition rule subtracts the intersection so it is not counted twice:

```text
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
         = 0.4 + 0.3 − 0.12
         = 0.58 = 29/50 = 58%.
```

## Question 2: Testing independence

Given `P(A) = 0.5`, `P(B) = 0.4`, and `P(A | B) = 0.7`.

Because `P(B) > 0`, independence would require `P(A | B) = P(A)`. Here:

```text
P(A | B) = 0.7 ≠ 0.5 = P(A).
```

Therefore, **A and B are not independent**. Knowing that B occurred changes the probability of A from 50% to 70%.

The equivalent intersection check reaches the same conclusion:

```text
P(A ∩ B) = P(A | B) × P(B)
         = 0.7 × 0.4 = 0.28 = 7/25.

P(A) × P(B) = 0.5 × 0.4 = 0.20 = 1/5.

0.28 ≠ 0.20, so P(A ∩ B) ≠ P(A) × P(B).
```

## Question 3: Bayes' rule

Given `P(A) = 0.6`, `P(B | A) = 0.5`, and `P(B | Aᶜ) = 0.2`.

First find the complement probability:

```text
P(Aᶜ) = 1 − P(A) = 1 − 0.6 = 0.4.
```

The events A and Aᶜ partition the sample space, so the law of total probability gives:

```text
P(B) = P(B | A)P(A) + P(B | Aᶜ)P(Aᶜ)
     = (0.5 × 0.6) + (0.2 × 0.4)
     = 0.30 + 0.08
     = 0.38 = 19/50.
```

Now apply Bayes' rule:

```text
P(A | B) = P(B | A)P(A) / P(B)
         = (0.5 × 0.6) / 0.38
         = 0.30 / 0.38
         = 15/19
         ≈ 0.789474 ≈ 78.95%.
```

## Question 4: Positive test result

Let D mean that a person has the disease and T⁺ mean that the test is positive. The question supplies:

- Prevalence: `P(D) = 0.02`.
- Sensitivity (true positive rate): `P(T⁺ | D) = 0.95`.
- Specificity (true negative rate): `P(T⁻ | Dᶜ) = 0.90`.

Therefore:

```text
P(Dᶜ) = 1 − 0.02 = 0.98.
P(T⁺ | Dᶜ) = 1 − 0.90 = 0.10.
```

The false positive rate is 10%, not 90%. Compute the overall probability of a positive result:

```text
P(T⁺) = P(T⁺ | D)P(D) + P(T⁺ | Dᶜ)P(Dᶜ)
      = (0.95 × 0.02) + (0.10 × 0.98)
      = 0.019 + 0.098
      = 0.117 = 117/1000.
```

By Bayes' rule:

```text
P(D | T⁺) = P(T⁺ | D)P(D) / P(T⁺)
          = 0.019 / 0.117
          = 19/117
          ≈ 0.162393 ≈ 16.24%.
```

As a frequency check using the supplied rates, in a hypothetical group of 10,000 people, 200 would have the disease and 190 of those would test positive. Of the 9,800 healthy people, 980 would test positive. Thus, `190 / (190 + 980) = 19/117` of positive tests would be from people who have the disease. The low prevalence explains why this conditional probability is much lower than the sensitivity.

## Question 5: Expected value, variance, and sample mean

The distribution is:

| Score x | P(X = x) | x P(X = x) | x² P(X = x) |
|---:|---:|---:|---:|
| 85 | 0.375 = 3/8 | 31.875 | 2709.375 |
| 90 | 0.375 = 3/8 | 33.750 | 3037.500 |
| 95 | 0.125 = 1/8 | 11.875 | 1128.125 |
| 100 | 0.125 = 1/8 | 12.500 | 1250.000 |
| **Total** | **1** | **90** | **8125** |

### (a) Expected value

```text
E[X] = Σ x P(X = x)
     = 85(0.375) + 90(0.375) + 95(0.125) + 100(0.125)
     = 31.875 + 33.750 + 11.875 + 12.500
     = 90 points.
```

### (b) Variance

First compute the second moment:

```text
E[X²] = Σ x² P(X = x)
      = 85²(0.375) + 90²(0.375) + 95²(0.125) + 100²(0.125)
      = 2709.375 + 3037.500 + 1128.125 + 1250.000
      = 8125.
```

Then:

```text
Var(X) = E[X²] − (E[X])²
       = 8125 − 90²
       = 8125 − 8100
       = 25 points².
```

This is the variance of the given probability distribution, so no sample-variance correction is needed. A direct check using squared deviations is:

```text
Var(X) = (85 − 90)²(0.375) + (90 − 90)²(0.375)
       + (95 − 90)²(0.125) + (100 − 90)²(0.125)
       = 9.375 + 0 + 3.125 + 12.500
       = 25 points².
```

### (c) Sample mean and comparison

For the eight observations `{85, 90, 85, 95, 90, 85, 100, 90}`:

```text
x̄ = (85 + 90 + 85 + 95 + 90 + 85 + 100 + 90) / 8
  = 720 / 8
  = 90 points.

x̄ − E[X] = 90 − 90 = 0.
```

The sample mean equals the expected value for this particular sample. Its observed proportions are exactly the distribution probabilities: three scores of 85, three of 90, one of 95, and one of 100. This exact match is a feature of the supplied sample; it does not imply that every random sample will have mean 90. The expectation of the sample mean is 90 when each observation is drawn from this distribution.

## Question 6: Entropy in bits

The four message probabilities are `0.4`, `0.3`, `0.2`, and `0.1`, which sum to 1.

### (a) Entropy of the given distribution

Use Shannon entropy with base-2 logarithms:

```text
H(X) = −Σ p(x) log₂ p(x)
     = −[0.4 log₂(0.4) + 0.3 log₂(0.3)
         + 0.2 log₂(0.2) + 0.1 log₂(0.1)].
```

The individual contributions are:

| Probability p | log₂(p), approximately | −p log₂(p), approximately |
|---:|---:|---:|
| 0.4 | −1.321928095 | 0.528771238 |
| 0.3 | −1.736965594 | 0.521089678 |
| 0.2 | −2.321928095 | 0.464385619 |
| 0.1 | −3.321928095 | 0.332192809 |

Adding the contributions without intermediate rounding:

```text
H(X) = 1.8464393446710154 bits
     ≈ 1.84644 bits.
```

### (b) Entropy when all messages are equally likely

Each of the four messages now has probability `1/4 = 0.25`, and `log₂(1/4) = −2`:

```text
H(uniform) = −4[(1/4) log₂(1/4)]
           = −4[(1/4)(−2)]
           = 2 bits.
```

The uniform distribution has the maximum entropy for four possible messages, `log₂(4) = 2` bits. This is higher than the entropy in part (a), because the messages are equally unpredictable.

## Reproducing the calculations

From the repository root, run:

```bash
uv run python scripts/check_probability.py
```

Alternatively, `python3 scripts/check_probability.py` works with only the Python standard library. The script calculates the answers, verifies them against exact expected values and the accompanying `assignment/probability_answers.json`, and prints the working. It does not write or change any files.
