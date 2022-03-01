# Linear model selection and regularization

Some take out:

* If $p \ge n$, least square method may have high variance, hence we use shrinkage method which can reduce variance.

## 1. Problem with linear regression with least square (ls)

* Prediction Accuracy: linear regression has **low bias** but suffer from **high variance**, especially when $n \approx p$. It cannot handle $n<p$.
* Model Interpretability: It is often the case that some or many of the variables used in a multiple regression model are **in fact not associated with the response**. Including such irrelevant variables leads to unnecessary complexity in the resulting model.

## 2. Selected alternatives to LS

* **Subset Selection**. This approach involves identifying **a subset of the $p$ predictors** that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables.
* **Shrinkage**. This approach involves fitting a model involving all $p$ predictors. However, the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (also known as regularization) has the effect of **reducing variance.**
  Depending on what type of shrinkage is performed, **some of the coefficients may be estimated to be exactly zero**. Hence, shrinkage methods can also perform variable selection.
  * Ridge Regression (Norm 2) 
  * Lasso (Norm 1)  can estimate coefficients to zero
* **Dimension Reduction**. This approach involves projecting the $p$ predictors into a $M$-dimensional subspace, where $M<\mathrm{p}$. This is achieved by computing $M$ different linear combinations, or projections, of the variables. Then these $M$ projections are used as predictors to fit a linear regression model by least squares
  * **Best subset selection**
    * Let $\mathcal{M}_{0}$ denote the null model, which contains no predictors. This model simply predicts the sample mean for each observation.
    2. For $k=1,2, \ldots p$ :
    (a) Fit all $\left(\begin{array}{l}p \\ k\end{array}\right)$ models that contain exactly $k$ predictors.
    (b) Pick the best among these $\left(\begin{array}{l}p \\ k\end{array}\right)$ models, and call it $\mathcal{M}_{k}$. Here best is defined as having the smallest RSS, or equivalently largest $R^{2}$.
    3. Select a single best model from among $\mathcal{M}_{0}, \ldots, \mathcal{M}_{p}$ using crossvalidated prediction error, $C_{p}$ (AIC), BIC, or adjusted $R^{2}$.
  * **Forward stepwise selection** and  **Backward stepwise selection**
    * Forward stepwise selection (can be used even $n \le p$)
      * Let $\mathcal{M}_{0}$ denote the null model, which contains no predictors.
      2. For $k=0, \ldots, p-1$ :
      (a) Consider all $p-k$ models that augment the predictors in $\mathcal{M}_{k}$ with one additional predictor.
      (b) Choose the best among these $p-k$ models, and call it $\mathcal{M}_{k+1}$. Here best is defined as having smallest RSS or highest $R^{2}$.
      3. Select a single best model from among $\mathcal{M}_{0}, \ldots, \mathcal{M}_{p}$ using crossvalidated prediction error, $C_{p}$ (AIC), BIC, or adjusted $R^{2}$.
    * Backward stepwise selection (require $n \ge p$)
      * Let $\mathcal{M}_{p}$ denote the full model, which contains all $p$ predictors.
      * For $k=p, p-1, \ldots, 1$ :
        (a) Consider all $k$ models that contain all but one of the predictors in $\mathcal{M}_{k}$, for a total of $k-1$ predictors.
        (b) Choose the best among these $k$ models, and call it $\mathcal{M}_{k-1}$. Here best is defined as having smallest RSS or highest $R^{2}$.
      * Select a single best model from among $\mathcal{M}_{0}, \ldots, \mathcal{M}_{p}$ using crossvalidated prediction error, $C_{p}$ (AIC), BIC, or adjusted $R^{2}$.
    * $$\begin{aligned}
      &C_{p}=\frac{1}{n}\left(\mathrm{RSS}+2 d \hat{\sigma}^{2}\right) \\
      &\mathrm{AIC}=\frac{1}{n \hat{\sigma}^{2}}\left(\mathrm{RSS}+2 d \hat{\sigma}^{2}\right) \\
      &\mathrm{BIC}=\frac{1}{n}\left(\mathrm{RSS}+\log (n) d \hat{\sigma}^{2}\right) \\
      &\text { Adjusted } R^{2}=1-\frac{\mathrm{RSS} /(n-d-1)}{\mathrm{TSS} /(n-1)}
      \end{aligned}$$





## 3. Shrinkage in details

### 3.1 Shrinkage method I: Ridge regression

Ridge regression is very similar to least squares, except that the **coefficients are estimated by minimizing a slightly different quantity**. In particular, the ridge regression coefficient estimates $\beta^{R}$ are the values that minimize
$$
\sum_{i=1}^{n}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} \beta_{j} x_{i j}\right)^{2}+\lambda \sum_{j=1}^{p} \beta_{j}^{2}=\mathrm{RSS}+\lambda \sum_{j=1}^{p} \beta_{j}^{2}
$$
where $\lambda \geq 0$ is a **tuning parameter**, to be determined separately. The above equation trades off two different criteria. As with least squares, ridge regression seeks coefficient estimates that fit the data well, by making the $RSS$ small. However, the second term, $\lambda \sum_{j} \beta_{j}^{2}$, called a **shrinkage penalty**, is small when $\beta_{1}, \ldots, \beta_{p}$ are close to zero, and so it has the effect of shrinking penalty the estimates of $\beta_{j}$ towards zero.

Unlike least squares, which generates **only one set** of coefficient estimates, ridge regression will produce a different set of coefficient estimates, $\hat{\beta}_{\lambda}^{R}$, for each value of $\lambda$. **Selecting a good value for $\lambda$ is critical.**

We want to shrink the estimated association of each variable with the response; however, **we do not want to shrink the intercept**, which is simply a measure of the mean value of the response when $x_{i 1}=x_{i 2}=\ldots=x_{i p}=0$. If we assume that the variables-that is, the columns of the data matrix $X$-have been centered to have mean zero before ridge regression is performed, then the estimated intercept will take the form $\hat{\beta}_{0}=\bar{y}$.
The shrinkage penalty is not **scale invariant**. Therefore, it is best to **apply ridge regression after standardizing the predictors.**

**$\lambda$ Increases will lead to decrease of flexibility hence lower variance and higher bias.**

In general, in situations where the relationship between the response and the predictors is **close to linear**, the least squares estimates will have **low bias but may have high variance**. **This means that a small change in the training data can cause a large change in the least squares coefficient estimates.** In particular, when the number of variables $p$ is almost as large as the number of observations $n$, the least squares estimates will be extremely variable. **And if $p>n$, then the least squares estimates do not even have a unique solution,** whereas ridge regression can still perform well by trading off a small increase in bias for a large decrease in variance. **Hence, ridge regression works best in situations where the least squares estimates have high variance.**



### 3.2 Ridge regression compared to subset selection

Ridge regression also has **substantial computational advantages** over best subset selection, which requires searching through $2^{p}$ models. As we discussed previously, even for moderate values of $p$, such a search can be computationally infeasible. In contrast, for any fixed value of $\lambda$, ridge regression only fits a single model, and the model-fitting procedure can be performed quite quickly. In fact, one can show that the computations required to solve the penalized least square, simultaneously for all values of $\lambda$, are almost identical to those for fitting a model using least squares.



### 3.3 Shrinkage method II: Lasso

Lasso, short for Least Absolute Shrinkage and Selection Operator, different from Ridge regression, performs variable selection. Lasso coefficients, $\hat{\beta}_{\lambda}^{L}$, minimizes
$$
\sum_{i=1}^{n}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} \beta_{j} x_{i j}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|=\mathrm{RSS}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|
$$
In statistical parlance, the lasso uses an $l_{1}$ penalty instead of an $l_{2}$ penalty. The $l_{1}$ norm of a coefficient vector $\beta$ is given by $\|\beta\|_{1}=\sum\left|\beta_{j}\right|$.

### 3.4 Comparison among Ridge regression | Lasso | Best subset selection

The Ridge regression is equivalent to
$$
\underset{\beta}{\operatorname{minimize}}\left\{\sum_{i=1}^{n}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} \beta_{j} x_{i j}\right)^{2}\right\} \quad \text { subject to } \quad \sum_{j=1}^{p} \beta_{j}^{2} \leq s
$$
Lasso is equivalent to
$$
\underset{\beta}{\operatorname{minimize}}\left\{\sum_{i=1}^{n}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} \beta_{j} x_{i j}\right)^{2}\right\} \quad \text { subject to } \quad \sum_{j=1}^{p}\left|\beta_{j}\right| \leq s
$$
The best subset regression is equivalent to
$$
\underset{\beta}{\operatorname{minimize}}\left\{\sum_{i=1}^{n}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} \beta_{j} x_{i j}\right)^{2}\right\} \quad \text { subject to } \quad \sum_{j=1}^{p} I\left(\beta_{j} \neq 0\right) \leq s
$$


Neither ridge regression nor the lasso will universally dominate the other. In general, one might expect the **lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients,** and the remaining predictors have coefficients that are very small or that equal zero. **Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size.** However, the number of predictors that is related to the response is never known a priori for real data sets. A technique such as cross-validation can be used in order to determine which approach is better on a particular data set.

![](https://s2.loli.net/2022/02/28/RW39rSHIQytlbCG.png)

- Left: The ridge regression coefficient estimates are shrunken proportionally towards zero, relative to the least squares estimates. 

- Right: The lasso coefficient estimates are soft-thresholded towards zero.

In the case of a more general data matrix $X$, the story is a little more complicated than what is depicted in the previous figure, but the main ideas still hold approximately: **ridge regression more or less shrinks every dimension of the data by the same proportion**, whereas the **lasso more or less shrinks all coefficients toward zero by a similar amount, and sufficiently small coefficients are shrunken all the way to zero.**



### 3.5 How to choose $\lambda$ (Tuning parameter)?

We choose a grid of $\lambda$ values, and compute the cross-validation error for each value of $\lambda$, as described in Chapter 5 . We then select the tuning parameter value for which the cross-validation error is smallest. Finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter.







