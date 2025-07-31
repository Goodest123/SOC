import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# --- 1. Data Download and Preprocessing Function ---
def get_and_prepare_data(ticker_symbol, interval_val, start_date_str_val, end_date_str_val):
    """
    Downloads historical data, standardizes columns, and converts index to datetime.
    """
    try:
        df = yf.download(tickers=ticker_symbol, interval=interval_val, start=start_date_str_val, end=end_date_str_val)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if df.empty:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

    new_columns = []
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            cleaned_col_name = str(col[0]).capitalize()
            new_columns.append(cleaned_col_name)
    else:
        for col in df.columns:
            new_columns.append(str(col).capitalize())
    df.columns = new_columns

    if 'Adj Close' in df.columns:
        if 'Close' in df.columns and df['Adj Close'].equals(df['Close']):
            df = df.drop(columns=['Adj Close'])
        elif 'Close' not in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)

    df = df.loc[:,~df.columns.duplicated()]

    desired_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    final_columns_present_and_desired = [col for col in desired_columns if col in df.columns]
    df = df[final_columns_present_and_desired]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if df.empty:
        return pd.DataFrame()
    
    return df


# --- 2. Add Technical Indicators Function ---
def add_indicators(df_input):
    """
    Adds RSI, MACD, EMA50, and ATR indicators to the DataFrame.
    """
    required_cols = ['Close', 'High', 'Low'] 
    for col_name in required_cols:
        if col_name not in df_input.columns:
            return df_input
        if df_input[col_name].isnull().all():
            pass

    min_indicator_len = 50 
    if len(df_input) < min_indicator_len:
        df_input['RSI'] = np.nan
        df_input['MACD_Line'] = np.nan
        df_input['MACD_Signal'] = np.nan
        df_input['MACD_Hist'] = np.nan
        df_input['EMA50'] = np.nan 
        return df_input

    df_input['RSI'] = ta.rsi(df_input['Close'], length=9)

    macd_df = ta.macd(df_input['Close'], fast=8, slow=21, signal=5)

    if macd_df is not None and not macd_df.empty:
        expected_macd_cols = ['MACD_8_21_5', 'MACDs_8_21_5', 'MACDh_8_21_5']
        missing_macd_cols = [col for col in expected_macd_cols if col not in macd_df.columns]

        if missing_macd_cols:
            df_input['MACD_Line'] = np.nan
            df_input['MACD_Signal'] = np.nan
            df_input['MACD_Hist'] = np.nan
        else:
            df_input['MACD_Line'] = macd_df['MACD_8_21_5']
            df_input['MACD_Signal'] = macd_df['MACDs_8_21_5']
            df_input['MACD_Hist'] = macd_df['MACDh_8_21_5']
    else:
        df_input['MACD_Line'] = np.nan
        df_input['MACD_Signal'] = np.nan
        df_input['MACD_Hist'] = np.nan
    
    df_input['EMA50'] = ta.ema(df_input['Close'], length=50)
    
    critical_indicator_cols = ['Close', 'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'EMA50'] 
    df_input = df_input.dropna(subset=[col for col in critical_indicator_cols if col in df_input.columns]) 

    return df_input


# --- Candlestick Pattern Functions ---
def is_strong_bullish_candle(open_price, high_price, low_price, close_price):
    """
    Checks if a candle is a strong bullish candle (large body, small upper wick, closes near high).
    """
    if close_price <= open_price: # Not a bullish candle
        return False
    
    body_size = close_price - open_price
    total_range = high_price - low_price
    
    if total_range == 0: # Avoid division by zero for flat candles
        return False

    body_ratio = body_size / total_range
    
    return body_ratio > 0.55

def is_strong_bearish_candle(open_price, high_price, low_price, close_price):
    """
    Checks if a candle is a strong bearish candle (large body, small lower wick, closes near low).
    """
    if close_price >= open_price: # Not a bearish candle
        return False
        
    body_size = open_price - close_price
    total_range = high_price - low_price
    
    if total_range == 0: # Avoid division by zero for flat candles
        return False

    body_ratio = body_size / total_range
    
    return body_ratio > 0.55


# --- Unified High-Momentum Pullback Backtest Logic Function ---
def run_backtest(df, ticker_symbol): 
    """
    Runs a backtest with a unified High-Momentum Pullback strategy.
    """
    if 'Volume' in df.columns:
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    else:
        df['Volume_SMA20'] = np.nan

    capital = 10000 
    risk_per_trade_capital_pct = 0.015 

    stop_loss_percent_of_entry = 0.0075
    take_profit_percent_of_entry = 0.015

    brokerage_fee_percent = 0.0003

    current_position = None 
    entry_price = 0
    entry_time = None
    stop_loss_level = None 
    take_profit_level = 0
    
    shares_held = 0 
    trade_id_counter = 0 

    trades = []
    trade_results_summary = {}

    max_drawdown = 0 
    peak_capital = capital 
    capital_history = [capital]

    close_col = 'Close'
    open_col = 'Open'
    high_col = 'High'
    low_col = 'Low'
    
    rsi_col = 'RSI'
    macd_line_col = 'MACD_Line'
    macd_signal_col = 'MACD_Signal'
    macd_hist_col = 'MACD_Hist'
    ema_long_col = 'EMA50' 
    volume_col = 'Volume'
    volume_sma_col = 'Volume_SMA20'
    required_backtest_cols = [close_col, open_col, high_col, low_col, rsi_col, macd_line_col, macd_signal_col, ema_long_col,volume_col, volume_sma_col]

    for col in required_backtest_cols:
        if col not in df.columns:
            return pd.DataFrame(), [capital], None

    for i in range(1, len(df)):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        if any(pd.isna(current_candle[col]) for col in required_backtest_cols) or \
           any(pd.isna(prev_candle[col]) for col in [rsi_col, macd_line_col, macd_signal_col, ema_long_col]): 
            capital_history.append(capital) 
            continue

        capital_history.append(capital)

        if current_position:
            current_high = current_candle[high_col]
            current_low = current_candle[low_col]
            current_close = current_candle[close_col]

            exit_reason = None
            exit_price = 0
            
            if current_position == 'long':
                if current_low <= stop_loss_level: 
                    exit_price = stop_loss_level
                    exit_reason = 'SL'
                elif current_high >= take_profit_level:
                    exit_price = take_profit_level
                    exit_reason = 'TP'
            else:
                if current_high >= stop_loss_level: 
                    exit_price = stop_loss_level
                    exit_reason = 'SL'
                elif current_low <= take_profit_level:
                    exit_price = take_profit_level
                    exit_reason = 'TP'
            
            if not exit_reason: 
                is_last_candle_of_data = (i == len(df) - 1)
                is_last_candle_of_day = False
                if i + 1 < len(df): 
                    if current_candle.name.date() != df.iloc[i+1].name.date():
                        is_last_candle_of_day = True

                if is_last_candle_of_day or is_last_candle_of_data:
                    exit_price = current_close 
                    exit_reason = 'EOD_Close' 
                    
            if exit_reason: 
                trade_pnl_dollars = shares_held * (exit_price - entry_price) if current_position == 'long' else shares_held * (entry_price - exit_price)
                
                trade_value_at_exit = shares_held * exit_price
                brokerage_cost = trade_value_at_exit * brokerage_fee_percent
                
                trade_pnl_dollars_net = trade_pnl_dollars - brokerage_cost
                capital += trade_pnl_dollars_net 

                trade_results_summary[trade_id_counter] = trade_pnl_dollars_net 

                drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                peak_capital = max(peak_capital, capital)

                pnl_pct = (trade_pnl_dollars_net / (shares_held * entry_price) * 100) if (shares_held * entry_price) != 0 else 0

                trades.append({
                    'Trade ID': trade_id_counter,
                    'Time': current_candle.name,
                    'Position': current_position,
                    'Entry Time': entry_time,
                    'Entry Price': round(entry_price, 2),
                    'Exit Price': round(exit_price, 2),
                    'Gross PnL ($)': round(trade_pnl_dollars, 2), 
                    'Brokerage ($)': round(brokerage_cost, 2),    
                    'Net PnL ($)': round(trade_pnl_dollars_net, 2), 
                    'PnL (%)': round(pnl_pct, 2),
                    'Capital': round(capital, 2),
                    'Exit Reason': exit_reason
                })

                current_position = None
                entry_price = 0
                entry_time = None
                stop_loss_level = None 
                take_profit_level = 0
                shares_held = 0

        else:
            current_close = current_candle[close_col]
            current_rsi = current_candle[rsi_col]
            prev_macd_line = prev_candle[macd_line_col]
            prev_macd_signal = prev_candle[macd_signal_col]
            current_macd_line = current_candle[macd_line_col]
            current_macd_signal = current_candle[macd_signal_col] 
            current_ema50 = current_candle[ema_long_col] 
            prev_ema50 = prev_candle[ema_long_col]       
            macd_buy_crossover = (prev_macd_line < prev_macd_signal and current_macd_line > current_macd_signal)
            macd_sell_crossover = (prev_macd_line > prev_macd_signal and current_macd_line < current_macd_signal)

            uptrend_confirmed_ema = (current_close > current_ema50 * (1 - 0.0015))
            downtrend_confirmed_ema = (current_close < current_ema50 * (1 + 0.0015))

            # --- Unified High-Momentum Pullback Entries ---
            # Long Entry
            if (uptrend_confirmed_ema and
                current_rsi < 60 and 
                macd_buy_crossover and
                current_candle[volume_col] > 1.1 * current_candle[volume_sma_col] and 
                is_strong_bullish_candle(current_candle[open_col], current_candle[high_col], current_candle[low_col], current_candle[close_col])):
                
                trade_id_counter += 1
                trade_results_summary[trade_id_counter] = 0

                current_position = 'long'
                entry_price = current_close
                entry_time = current_candle.name

                stop_loss_level = entry_price * (1 - stop_loss_percent_of_entry) 
                take_profit_level = entry_price * (1 + take_profit_percent_of_entry) 

                dollar_risk_per_share = entry_price * stop_loss_percent_of_entry 
                
                if dollar_risk_per_share <= 0: 
                    trade_id_counter -= 1
                    del trade_results_summary[trade_id_counter + 1]
                    continue

                shares_to_buy = (capital * risk_per_trade_capital_pct) / dollar_risk_per_share 
                shares_held = shares_to_buy 

            # Short Entry
            elif (downtrend_confirmed_ema and
                  current_rsi > 40 and 
                  macd_sell_crossover and
                  current_candle[volume_col] > 1.1 * current_candle[volume_sma_col] and
                  is_strong_bearish_candle(current_candle[open_col], current_candle[high_col], current_candle[low_col], current_candle[close_col])):
                
                trade_id_counter += 1
                trade_results_summary[trade_id_counter] = 0

                current_position = 'short'
                entry_price = current_close
                entry_time = current_candle.name

                stop_loss_level = entry_price * (1 + stop_loss_percent_of_entry) 
                take_profit_level = entry_price * (1 - take_profit_percent_of_entry) 

                dollar_risk_per_share = entry_price * stop_loss_percent_of_entry 

                if dollar_risk_per_share <= 0:
                    trade_id_counter -= 1
                    del trade_results_summary[trade_id_counter + 1]
                    continue

                shares_to_buy = (capital * risk_per_trade_capital_pct) / dollar_risk_per_share 
                shares_held = shares_to_buy
            
    if current_position:
        exit_price = df.iloc[-1][close_col] 
        trade_pnl_dollars = shares_held * (exit_price - entry_price) if current_position == 'long' else shares_held * (entry_price - exit_price)
        trade_value_at_exit = shares_held * exit_price
        brokerage_cost = trade_value_at_exit * brokerage_fee_percent
        trade_pnl_dollars_net = trade_pnl_dollars - brokerage_cost
        capital += trade_pnl_dollars_net
        trade_results_summary[trade_id_counter] = trade_pnl_dollars_net

        drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        peak_capital = max(peak_capital, capital)

        pnl_pct = (trade_pnl_dollars_net / (shares_held * entry_price) * 100) if (shares_held * entry_price) != 0 else 0

        trades.append({
            'Trade ID': trade_id_counter,
            'Time': df.iloc[-1].name,
            'Position': current_position,
            'Entry Time': entry_time,
            'Entry Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Gross PnL ($)': round(trade_pnl_dollars, 2),
            'Brokerage ($)': round(brokerage_cost, 2),
            'Net PnL ($)': round(trade_pnl_dollars_net, 2),
            'PnL (%)': round(pnl_pct, 2),
            'Capital': round(capital, 2),
            'Exit Reason': 'EOD_Close_Final' 
        })


    final_capital = capital
    total_trades_events = len(trades)
    winning_conceptual_trades = sum(1 for pnl in trade_results_summary.values() if pnl > 0)
    losing_conceptual_trades = sum(1 for pnl in trade_results_summary.values() if pnl < 0)
    total_conceptual_trades = len(trade_results_summary)
    win_rate = (winning_conceptual_trades / total_conceptual_trades) * 100 if total_conceptual_trades > 0 else 0
    total_pnl = capital - capital_history[0]
    total_return_pct = (total_pnl / capital_history[0]) * 100 if capital_history[0] > 0 else 0


    summary_data = {
        'Ticker': ticker_symbol,
        'Initial Capital': capital_history[0],
        'Final Capital': final_capital,
        'Total Profit ($)': round(total_pnl, 2),
        'Total Profit (%)': round(total_return_pct, 2),
        'Total Trade Events': total_trades_events,
        'Total Trades': total_conceptual_trades,
        'Winning Trades': winning_conceptual_trades,
        'Losing Trades': losing_conceptual_trades,
        'Avg Win Rate (%)': round(win_rate, 2),
    }

    return trades, capital_history, summary_data


def main():
    tickers = ["reliance.ns", "trent.ns", "tatasteel.ns", "eternal.ns", "Sbin.ns", "Vbl.ns", "Tcs.ns"]
    interval = "15m"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    all_summaries = []
    all_trades = {} # Dictionary to store trades for each ticker

    for current_ticker in tickers:
        df = get_and_prepare_data(current_ticker, interval, start_date_str, end_date_str)
        
        if not df.empty:
            df = add_indicators(df)
            if not df.empty: 
                trades, capital_history, summary = run_backtest(df, current_ticker)
                
                if summary is not None and 'Total Trades' in summary: 
                    all_summaries.append(summary)
                
                if trades: # Check if there are any trades
                    all_trades[current_ticker] = pd.DataFrame(trades)


    # Print number of trades for each ticker
    print("\n--- Number of Trades per Company ---")
    for ticker, trades_df in all_trades.items():
        print(f"{ticker}: {len(trades_df)}")


    if all_summaries:
        final_summary_df = pd.DataFrame(all_summaries)
        
        total_winning_trades_overall = final_summary_df['Winning Trades'].sum()
        total_trades_overall = final_summary_df['Total Trades'].sum()
        
        overall_avg_win_rate = (total_winning_trades_overall / total_trades_overall) * 100 if total_trades_overall > 0 else 0
        
        total_initial_capital = final_summary_df['Initial Capital'].sum()
        total_final_capital = final_summary_df['Final Capital'].sum()
        overall_total_profit = total_final_capital - total_initial_capital
        overall_total_profit_pct = (overall_total_profit / total_initial_capital) * 100 if total_initial_capital > 0 else 0

        print(f"\n--- Overall Summary ---")
        print(f"Overall Win Rate: {overall_avg_win_rate:.2f}%")
        print(f"Overall Profit Percentage: {overall_total_profit_pct:.2f}%")
        print(f"Total Number of Trades: {total_trades_overall}")
    else:
        print("No overall summary available.")

if __name__ == "__main__":
    main()
