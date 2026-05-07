# Project Description

## Project Scope and Objectives

### Problem Statement
Finance Forecaster aims to aid financial decisions in stock market trading by predicting next-day financial movements on
the market index or stock of their choice. Enabling traders to make informed decisions on their purchasing or selling
actions based on historical market data, as analyzed and interpreted by our proprietary machine learning model. Through
our model pipeline we seek to help financial professionals and enthusiasts make informed business decisions on their
investments.

Making decisions on whether to buy or sell puts, calls, individual stock or index shares can feel like a coin flip;
Finance Forecasters seeks to help decision makers make confident decisions on their financial asset actions by providing
predictions on whether their index or stock of choice will move up or down the following day. Currently financial
professionals utilize their hard earned experince, and manual review of the data to make their decisions. A process
that is error-prone due to human error and the limitation of manual data analysis. By using time-series
analysis along with historical multi-year datasets pulled from well known and vetted data sources Finance Forecasters
will provide next-day predictions on individual share movements. Enabling financial decision makers to maintain confidence
in their actions by providing historical analysis of market movements. As well as making educated well-informed market
decisions accessible to professionals with less financial experience.

### Project Objectives and Expected Impact
Finance Forecaster intends to leverage Machine Learning and Artificial Intelligence to analyze historical market data for
next-day price directionality by functioning as a short-term predictive agent delivering trade recommendations based on
predicted confidence of a stock’s price and directional movement.

The model will initially be built around the Investco QQQ Trust index fund, which is a passively managed exchange-traded
fund (ETF) that tracks the Nasdaq 100 Index. The primary dataset will be used to establish a robust model and
benchmarks; however, the model architecture will be built and deployed to allow dynamic selection of any user input
publicly traded company with a ticker accessible using the Python yfinance library.

The methodology will employ a multifaceted approach by evaluating ARIMA and GARCH for statistical and volatility
forecasting in conjunction with XGBoost and LSTM for non-linear pattern recognition. These individual estimators may be
integrated into a robust hybrid ensemble framework to maximize the model’s predictive accuracy and resilience.


### Success Metrics
The performance of our proprietary financial prediction machine learning model will be measured through classification
and regression reports. The minimum performance accuracy that we are aiming for is a 60% accuracy minumum matching
from our model output to actual matket movements. We will also compare our model to current pre-existing financial
models as a baseline performance.

### Comparable Problems and Current Manual Solutions
Comparable problems to our problem domain include risk classification of financial markets (housing, tech) and consumers
(loan risk classification). As well as prediction of other volatile markets, such as commodity prediction.

Current manual solutions include consultations from experts such as professional fund managers and traders, as well as
manual analysis of existing datasets. Manual analysis would require using a significant amount of tabular data for each
individual stock or fund with index funds for benchmarking and then manually calculating metrics like moving averages,
yield curve, et al. Building predictive models like ARIMA and Garch would be challenging, but then a RNN by hand would
be extremely challenging.

### Assumptions Made
Our primary baseline assumption is that stock market movements can be predicted using historical data and regression
analysis to begin with, and that the market is not fully irrational and chaotic. We are assuming that there are baseline
patterns that can be identified by machine learning to inform of future movements.
We are also assuming that we can outperform manual review and analysis of market data to make informed decisions through
the implementation of Artificial Intelligence and Machine Learning.
