## Project Scope and Objectives

### Problem Statement
Finance Forecaster aims to aid financial decisions in stock market trading by predicting next-day financial movements on
the market index or stock of their choice. Enabling traders to make informed decisions on their purchasing or selling
actions based on historical market data, as analyzed and interpreted by our proprietary machine learning model.

Making decisions on whether to buy or sell puts, calls, individual stock or index shares can feel like a coin flip;
Finance Forecasters seeks to help decision makers make confident decisions on their financial asset actions by providing
predictions on whether their index or stock of choice will move up or down the following day. By using time-series
analysis along with historical multi-year datasets pulled from well known and vetted data sources Finance Forecasters
will provide next-day predictions on individual share movements. Enabling financial decision makers to maintain confidence
in their actions by providing historical analysis of market movements.

### Project Objectives and Expected Impact
This project intends to leverage Machine Learning and Artificial Intelligence to analyze historical market data for
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
Specify how success will be measured. Include quantitative and/or qualitative metrics that will be used to evaluate
project outcomes.

### Project Description
Provide a detailed overview of the project, including methodology, approach, and anticipated results. Maintain clarity
and organization while ensuring the description is comprehensive.


2. How will your solution be used?\
 The solution will be used by finance proffessionals to help analyze market movements in order to make business
decisions on investments.

3. What are the current solutions/workarounds (if any)?\
 Currently proffesional tools exist to help predict market indicators using models based on company analytics as
well as manual review of the data.

5. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?\
   Our market analysis is performed offline, using a supervised ensemble model.

6. How should performance be measured?\
  Performance is measured through classification and regression reports. The minimum performance we are looking for
is a 60% accuracy minimum matching from the model output to actual market movements. We can compare to pre-exisiting
models as a baseline performance of our model.

7. What are comparable problems? Can you reuse experience or tools?\
 Risk classification of financial markets and participants is comparable to our specific problem domain. We can use
proffesional trader experience to aid in the analysis of our model performance, as well as pre-existing finance
prediction models.

8. Is human expertise available?\
 Human expertise is availabe in the form of fund management groups

9. How would you solve the problem manually?
    This could be solved manually using a significant amount of tabular data for each individual stock or fund with
index funds for benchmarking and the manually calculating metrics like moving averages, yield curve, et al.
Building predictive models like ARIMA and Garch would be challenging, but then a RNN by hand woud be extremely
challenging.

10. List the assumptions you (or others) have made so far.
11. Verify assumptions if possible.
