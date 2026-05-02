## Project Scope and Objectives
###Problem Statement
Clearly define the problem your project aims to address. Provide sufficient background to contextualize the issue and explain why it is relevant or important.

###Project Objectives and Expected Impact
Outline the primary goals of the project. Describe what you intend to achieve and the potential impact or applications of your work.

###Success Metrics
Specify how success will be measured. Include quantitative and/or qualitative metrics that will be used to evaluate project outcomes.

###Project Description (300+ words)
Provide a detailed overview of the project, including methodology, approach, and anticipated results. Maintain clarity and organization while ensuring the description is comprehensive.

1. Define the objective in business terms.

This project intends to leverage Machine Learning and Artificial Intelligence to analyze historical market data for next-day price directionality by functioning as a short-term predictive agent delivering trade recommendations based on predicted confidence of a stock’s price and directional movement. 
 
The model will initially be built around the Investco QQQ Trust index fund, which is a passively managed exchange-traded fund (ETF) that tracks the Nasdaq 100 Index. The primary dataset will be used to establish a robust model and benchmarks; however, the model architecture will be built and deployed to allow dynamic selection of any user input publicly traded company with a ticker accessible using the Python yfinance library. 
 
 
The methodology will employ a multifaceted approach by evaluating ARIMA and GARCH for statistical and volatility forecasting in conjunction with XGBoost and LSTM for non-linear pattern recognition. These individual estimators may be integrated into a robust hybrid ensemble framework to maximize the model’s predictive accuracy and resilience. 


2. How will your solution be used?\
 The solution will be used by finance proffessionals to help analyze market movements in order to make business decisions on investments.
  
3. What are the current solutions/workarounds (if any)?\
 Currently proffesional tools exist to help predict market indicators using models based on company analytics as well as manual review of the data.

5. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?\
   Our market analysis is performed offline, using a supervised ensemble model.
 
6. How should performance be measured?\
  Performance is measured through classification and regression reports. The minimum performance we are looking for is a 60% accuracy minimum matching from the model output to actual market movements. We can compare to pre-exisiting models as a baseline performance of our model.

9. What are comparable problems? Can you reuse experience or tools?\
 Risk classification of financial markets and participants is comparable to our specific problem domain. We can use proffesional trader experience to aid in the analysis of our model performance, as well as pre-existing finance prediction models. 

10. Is human expertise available?\
 Human expertise is availabe in the form of fund management groups 

11. How would you solve the problem manually?

    
13. List the assumptions you (or others) have made so far.
14. Verify assumptions if possible.
