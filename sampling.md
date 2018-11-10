
# Stratified Sampling
- https://gist.github.com/spacelis/6088623

R 

```R
require(data.table)
require(sampling)

set.seed(1)
n <- 1e4
d <- data.table(age = sample(1:5, n, T), 
                lc = rbinom(n, 1 , .5),
                ants = rbinom(n, 1, .7))

# Sort
setkey(d, age, lc)

# Population size by strata
d[, .N, keyby = list(age, lc)]
#     age lc    N
#  1:   1  0 1010
#  2:   1  1 1002
#  3:   2  0  993
#  4:   2  1 1026
#  5:   3  0 1021
#  6:   3  1  982
#  7:   4  0  958
#  8:   4  1  940
#  9:   5  0 1012
# 10:   5  1 1056

# Select sample
set.seed(2)
s <- data.table(strata(d, c("age", "lc"), rep(30, 10), "srswor"))

# Sample size by strata
s[, .N, keyby = list(age, lc)]
#     age lc  N
#  1:   1  0 30
#  2:   1  1 30
#  3:   2  0 30
#  4:   2  1 30
#  5:   3  0 30
#  6:   3  1 30
#  7:   4  0 30
#  8:   4  1 30
#  9:   5  0 30
# 10:   5  1 30
```
