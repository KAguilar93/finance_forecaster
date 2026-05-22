import pandas as pd
from arch import arch_model


def compute_expanding_garch_volatility(returns: pd.Series, min_window: int = 500) -> pd.Series:
    """Compute GARCH(1,1) volatility using expanding window."""
    print("Computing expanding-window GARCH volatility (this may take 1-3 minutes)...")

    returns_pct = returns.dropna() * 100
    conditional_vol = pd.Series(index=returns_pct.index, dtype=float, name="garch_conditional_volatility")

    for i in range(min_window, len(returns_pct)):
        window_returns = returns_pct.iloc[: i + 1]

        try:
            am = arch_model(window_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=True)
            res = am.fit(disp="off", show_warning=False)
            conditional_vol.iloc[i] = res.conditional_volatility.iloc[-1]
        except Exception:
            conditional_vol.iloc[i] = conditional_vol.iloc[i - 1] if i > 0 else window_returns.std()

    print("GARCH volatility computed.")
    return conditional_vol


def fit_garch_model(df: pd.DataFrame, ticker: str, min_window: int = 500) -> pd.DataFrame:
    """Fit GARCH model and add volatility features."""
    df = df.copy()

    df["garch_conditional_volatility"] = compute_expanding_garch_volatility(df["log_return"], min_window)

    df["garch_volatility_change"] = df["garch_conditional_volatility"].diff()
    df["garch_volatility_lag_1"] = df["garch_conditional_volatility"].shift(1)
    df["garch_volatility_lag_2"] = df["garch_conditional_volatility"].shift(2)

    print(f"GARCH features added for {ticker}")
    return df
