# ISEN613 EDA (Engineering Data Analysis)

[toc]

## Questions on Logistic Regression

### Is logistic regression a method for classification or regression?

Logistic regression is a method for **classification**, and it's a **binary classification method.**

### How many classes (or levels) does the response variable have in a logistic regression model?

2 since it's a binary classification method.

### Does the slope(s) in logistic regression have the same interpretation as slope(s) in linear regression?

* Slope in linear regression: It means the relationship between variables, and it can be used to estimate **an average rate of change**.
* Slope in logistic regression:

### In fitting a logistic regression, how to generate predictions based on the fitted model?

1. First we need to decide our threshold, a numerical value which separates our input values either "0" or "1".
2. We have input value $x$, and put it into fitted model we can have predicted value $\hat{y}$, we can compare it with threshold to decide its final output value.

## Question on LDA-QDA

### What are the assumptions in LDA? Does logistic regression have such assumptions?

LDA stands for Linear Discriminant Analysis, it has some basic assumptions:

* All variables are normally distributed: 
  * $\text { Input variable in class } k: \quad X \sim N\left(\mu_{k}, \sigma_{k}^{2}\right)$

* Assume those normal distributions have **equal variance**:
  * $\sigma_{1}^{2}=\sigma_{2}^{2}=\cdots=\sigma_{K}^{2}=\sigma^{2}$

### What is the relationship of LDA and Bayes classifier?

Bayes classifier also known as gold standard for classification and has the best performance:
$$
\operatorname{Pr}(Y=k \mid X=x)=\frac{\pi_{k} f_{k}(x)}{\sum_{l=1}^{K} \pi_{l} f_{l}(x)}
$$
Here prior probability: $P\left(y_{i}=k\right)=\pi_{k}$,  $\text {Probability of observing } x_{i}$ $\text { from the } k \text { th class } f_{k}\left(x_{i}\right)=P\left(x_{i} \mid y_{i}=k\right)$.

But questions are:

* For real data, we do not know the **distribution of predictor(s)** $f_{k}(x)$ 
* The Bayes classifier is **impossible** to compute

### What are the suitable situations of LDA vs. logistic regression? (Differences)

**The linear coefficients are estimated differently.**

**MLE** for logistic models and estimated mean and variance based on **Gaussian** assumptions for the LDA. LDA makes more restrictive Gaussian assumptions and therefore expected to work better than logistic models **if they (Gaussian assumptions) are met.**

### What are the suitable situations of LDA vs. QDA? 

- QDA allows for different variances among classes
- QDA estimates a separate covariance matrix $\Sigma_{k}$ for each class $k$
- QDA need to estimate $(K) \times\left((p+1)+\frac{p(p+1)}{2}\right) \sim O\left (K p^{2}\right)$
- $\text { LDA need to estimate }(K) \times(p+1)+\frac{p(p+1)}{2} \sim O\left ( p^{2}\right)$

#### Which approach is better: LDA or QDA?

- QDA is **more flexible** than LDA.
- Boundaries are quadratic in QDA
- QDA works best when the **variances are very different** between classes and we **have enough observations** to accurately estimate the variances.
- LDA works best when the **variances are similar** among classes or we **don't have enough** data to accurately estimate the variances.

## Questions on KNN

### Is KNN a parametric method or nonparametric method?

KNN is a nonparametric method.

### What is the shape of decision boundary of KNN?

The decision boundary depends on the number of neighbors $K$, the greater the $K$ is, the less flexible the boundary gets, gradually it will get close to linear if $K$ Is large enough.

### When $K$ increases, does the flexibility of KNN increase or decrease?

Decrease

### How to find the optimal value of $K$ in applying KNN?

We can use k-fold cross validation.

## Comparison between KNN LDA QDA

### What is the shape of decision boundary for each classifier?

* Logistic regression: linear
* LDA: linear
* QDA: non-linear
* KDD: non-linear to linear (K grows)

### When a linear boundary works for the data, should we choose LDA or logistic regression?

- LDA assumes that the observations are drawn from the normal 

- LDA assumes common variance in each class
- If the normality assumption holds, LDA is better

### When a linear boundary does not work, should we choose QDA or KNN?

QDA serves as a compromise between the non-parametric KNN method and the linear LDA and logistic regression approaches. Since QDA assumes a quadratic decision boundary, it can accurately model a wider range of problems than can the linear methods. Though **not as flexible** as KNN, QDA can perform better in the presence of **a limited number of training observations** because it does make some assumptions about the form of the decision boundary.
