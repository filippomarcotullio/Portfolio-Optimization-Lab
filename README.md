Portfolio Optimization Lab

This project collects a set of small experiments on portfolio optimization using Python.
The main goal was to build, step by step, a minimal toolkit to work with historical market data, estimate returns and covariances, and compare a few basic allocation rules.

All data comes from Yahoo Finance.

What’s inside

The notebook includes:

price download and log-return computation

estimation of annualized mean returns and covariance

an equal-weight portfolio

a minimum-variance (Markowitz) portfolio with simple constraints

a basic version of the Black–Litterman model (with one example view)

comparison of weights, risk/return measures, and in-sample equity curves

The purpose is not to build a perfect model, but to understand how these methods work and learn how to implement them from scratch.

Tools and libraries

Python

pandas, numpy

yfinance for data

scipy for numerical optimization

matplotlib for visualization

How to run

Open the notebook in JupyterLab and run the cells from top to bottom.
No special configuration is required beyond the common Python libraries listed above.

Possible extensions

A few ideas I might explore later:

Sharpe-ratio optimization

out-of-sample testing

weight constraints (sector limits, max weight, etc.)

a more complete Black–Litterman implementation
