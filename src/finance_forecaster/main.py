

from .pipeline import run_pipeline

if __name__ == "__main__":
    ticker = "qqq"
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    
    run_pipeline(ticker, start_date, end_date)
