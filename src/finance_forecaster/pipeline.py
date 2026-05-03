import pandas as pd
from .config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from .data.download import download_stock_data, download_external_market_features
from .data.features import add_log_returns, add_lstm_features
from .models.garch import fit_garch_model
from .models.lstm import train_lstm_model 
from .backtest.regime_aware import run_regime_aware_backtest
from .utils.plotting import save_lstm_training_plots, save_regime_aware_equity_curve


def run_pipeline(ticker: str = "qqq", start_date: str = "2015-01-01", end_date: str = "2025-01-01"):
    """Main pipeline orchestrator."""
    print(f"Starting pipeline for {ticker}...")
    
    # 1. Download data
    df = download_stock_data(ticker, start_date, end_date)
    df = add_log_returns(df)
    
    external = download_external_market_features(start_date, end_date)
    df = df.merge(external, left_index=True, right_index=True, how="left")
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
  
    df = fit_garch_model(df, ticker)
   
    df = add_lstm_features(df)
   
    print("LSTM training placeholder - connect your full LSTM logic here")
    
    print(f"Pipeline completed for {ticker}!")
    print(f"Reports saved in: {REPORTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    run_pipeline()
