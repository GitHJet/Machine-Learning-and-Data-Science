# This is a repository of my Machine Learning course by Stanford University on Coursera

## Summary

In this course, I learnt machine learning techniques and practiced implementing them.

Topics learnt:

1. **Supervised learning** (parametric/non-parametric algorithms, support vector machines, kernels, neural networks).
2. **Unsupervised learning** (clustering, dimensionality reduction, recommender systems, deep learning).
3. **Best practices in machine learning** (bias/variance theory; innovation process in machine learning and AI).

Practiced applications on:

1. **Smart robots** (perception, control)
2. **Text understanding** (web search, anti-spam)
3. **Computer vision**
4. Medical informatics, audio, database mining, and other areas.

Click the ML Exercises sub-directory to read a summary of my work.


Derive the gradients of the sigmoid function and show that it can be rewritten as a function
of the function value (i.e., in some expression where only σ(x), but not x, is present)

<a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;\begin{align*}\sigma(x)'&=\left(\frac{1}{1&plus;e^{-x}}\right)'=\frac{-(1&plus;e^{-x})'}{(1&plus;e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1&plus;e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1&plus;e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1&plus;e^{-x})^2}=\frac{e^{-x}}{(1&plus;e^{-x})^2}&space;\newline&space;&=\left(\frac{1}{1&plus;e^{-x}}\right)\left(\frac{e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{&plus;1-1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}&space;-&space;\frac{1}{1&plus;e^{-x}}\right)=\sigma(x)(1&space;-&space;\sigma(x))\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\LARGE&space;\begin{align*}\sigma(x)'&=\left(\frac{1}{1&plus;e^{-x}}\right)'=\frac{-(1&plus;e^{-x})'}{(1&plus;e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1&plus;e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1&plus;e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1&plus;e^{-x})^2}=\frac{e^{-x}}{(1&plus;e^{-x})^2}&space;\newline&space;&=\left(\frac{1}{1&plus;e^{-x}}\right)\left(\frac{e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{&plus;1-1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}\right)=\sigma(x)\left(\frac{1&space;&plus;&space;e^{-x}}{1&plus;e^{-x}}&space;-&space;\frac{1}{1&plus;e^{-x}}\right)=\sigma(x)(1&space;-&space;\sigma(x))\end{align*}" title="\LARGE \begin{align*}\sigma(x)'&=\left(\frac{1}{1+e^{-x}}\right)'=\frac{-(1+e^{-x})'}{(1+e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1+e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1+e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1+e^{-x})^2}=\frac{e^{-x}}{(1+e^{-x})^2} \newline &=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{+1-1 + e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right)=\sigma(x)(1 - \sigma(x))\end{align*}" /></a>
