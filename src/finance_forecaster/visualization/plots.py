

import matplotlib.pyplot as plt
import pandas as pd
import os

def save_lstm_training_plots(history_df: pd.DataFrame, ticker: str, figures_dir: str):
    """Save LSTM loss and accuracy plots."""
    os.makedirs(figures_dir, exist_ok=True)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df.get("val_loss", []), label="Validation Loss")
    plt.title(f"{ticker} LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"{ticker}_lstm_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Accuracy plot (if available)
    if "accuracy" in history_df.columns and "val_accuracy" in history_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df["accuracy"], label="Train Accuracy")
        plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
        plt.title(f"{ticker} LSTM Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, f"{ticker}_lstm_accuracy.png"), dpi=300, bbox_inches="tight")
        plt.close()


def save_regime_aware_equity_curve(backtest_df: pd.DataFrame, ticker: str, figures_dir: str):
    """Save equity curve plot."""
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{ticker}_regime_aware_equity_curve.png")
    
    plt.figure(figsize=(12, 5))
    plt.plot(backtest_df["date"], backtest_df["strategy_equity"], label="Strategy")
    plt.plot(backtest_df["date"], backtest_df["buy_hold_equity"], label="Buy & Hold")
    plt.title(f"{ticker} Regime-Aware Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved equity curve: {path}")
