# Finance Forecasters
## Kevin Aguilar, Shang Andrews, James Russo, Joseph Hughes
### [SE489] ML Engineering for Production (MLOps)

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

## Selection of Data
We are using data available from yfinance. We chose yfinance due to its public availabilty, thorough recording of financial market history, and it is freely accessible. In order to simplify our model development and experimentation we are limiting ourselves to only using the QQQ index fund for development. However, the objective is to allow users to track any stock or index fund.
We are currently accessing our data by running the download.py file to download the desired data in the requisite format from yfinance. Our data pipeline for downloading and checking in our data to dvc with version tagging, and feeding it to our model for training is a work in progress.

## Model Considerations
We have opted to develop our own model in order to limit the scope of predictions provided to next day financial movements of any stock or index. Using QQQ as our selected development and testing index. In this way we can control and extract only the relevant features and labels from the data. Additionally, by developing our own model we can run the required time-series and regression analysis while maintain high performance, by avoiding larger models that may require excess computation and training. Allowing us to stay focused on our desired objective of next day market predictions. As well as provide us the experience in developing a small scale machine learning model while building a continuous training, integration, and delivery pipeline around it.

## Open Source Tools
We are using yfinance for pulling our required data, sci-kit learn and scipy for model development, statsmodels and arch for statistical and time-series modeling, and fastapi and uvicorn for api development.

## Findings, challenges, and areas for improvement
We found that it is rather difficult to predict financial movements due to the natural volatility of the stock market, unique data requirements and transformations for time-series analysis, as well a human psychological factor involved. Particularly in markets being influenced due to participant interactions, including attempts to beat the market by using their own biases and predictions that are based in feeling rather than hard data. Determining wether to use linear regression or a more difficult but advanced RNN and LSTM analysis, designed for time-series forecasting was a main challenge of our model development.

Areas for improvement include refactoring our model by distributing the various modules into their own python files for improved separation of concerns, readability, and maintainability. As well as improving our data and training pipelines in order to automate the process. In order to maintain model accuracy we will likely need to employ dynamic training architectures such as orchestrated pull-based or message-based training in order to train the model at regular intervals to include updated financial information and movements, depending if we want real time data ingestion or not. Finally, documentation could use some work to become more thorough as we further refine our data gathering, training, and development processes.
