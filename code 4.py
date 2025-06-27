import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# Ensure openpyxl is available for Excel output
# If you run this locally and get an error, you might need to install it: pip install openpyxl

# --- Parameters ---
tickers = ["NVDA", "AAPL", "GOOGL", "eternal.ns", "Sbin.ns", "Vbl.ns", "Tcs.ns"] # List of tickers to backtest, updated
interval = "15m" # Time interval is 15 minutes
# Set date range for data download
end_date = datetime.now()
start_date = end_date - timedelta(days=59) # Approximately 2 months of data

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# --- 1. Data Download and Preprocessing Function ---
def get_and_prepare_data(ticker_symbol, interval_val, start_date_str_val, end_date_str_val):
    """
    Downloads historical data, standardizes columns, and converts index to datetime.
    """
    print(f"Downloading data for {ticker_symbol}...")
    try:
        df = yf.download(tickers=ticker_symbol, interval=interval_val, start=start_date_str_val, end=end_date_str_val)
        if df.empty:
            print(f"WARNING: No data downloaded for {ticker_symbol}. Skipping this ticker.")
            return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to download data for {ticker_symbol}. Reason: {e}. Skipping this ticker.")
        return pd.DataFrame()

    # Column Standardization
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
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    elif df.index.tz != 'UTC':
        df.index = df.index.tz_convert('UTC')

    if df.empty:
        print(f"DataFrame for {ticker_symbol} is empty after standardization. Cannot proceed with this ticker.")
        return pd.DataFrame()
    
    return df


# --- 2. Add Technical Indicators Function ---
def add_indicators(df_input):
    """
    Adds RSI, MACD, and EMA50 indicators to the DataFrame.
    """
    required_cols = ['Close', 'High', 'Low'] 
    for col_name in required_cols:
        if col_name not in df_input.columns:
            print(f"ERROR: Missing required column '{col_name}' for indicator calculation. Skipping indicator calculation.")
            return df_input
        if df_input[col_name].isnull().all():
            print(f"WARNING: Column '{col_name}' is entirely NaN. Indicator calculations will likely fail.")

    min_indicator_len = 50 
    if len(df_input) < min_indicator_len:
        df_input['RSI'] = np.nan
        df_input['MACD_Line'] = np.nan
        df_input['MACD_Signal'] = np.nan
        df_input['MACD_Hist'] = np.nan
        df_input['EMA50'] = np.nan 
        return df_input

    df_input['RSI'] = ta.rsi(df_input['Close'], length=9)

    macd_df = ta.macd(df_input['Close'], fast=12, slow=26, signal=9)

    if macd_df is not None and not macd_df.empty:
        expected_macd_cols = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
        missing_macd_cols = [col for col in expected_macd_cols if col not in macd_df.columns]

        if missing_macd_cols:
            df_input['MACD_Line'] = np.nan
            df_input['MACD_Signal'] = np.nan
            df_input['MACD_Hist'] = np.nan
        else:
            df_input['MACD_Line'] = macd_df['MACD_12_26_9']
            df_input['MACD_Signal'] = macd_df['MACDs_12_26_9']
            df_input['MACD_Hist'] = macd_df['MACDh_12_26_9']
    else:
        df_input['MACD_Line'] = np.nan
        df_input['MACD_Signal'] = np.nan
        df_input['MACD_Hist'] = np.nan
    
    df_input['EMA50'] = ta.ema(df_input['Close'], length=50)
        
    critical_indicator_cols = ['Close', 'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'EMA50'] 
    df_input = df_input.dropna(subset=[col for col in critical_indicator_cols if col in df_input.columns]) 

    return df_input


# --- 3. Trend-Following Pullback Backtest Logic Function ---
def run_backtest(df, ticker_symbol): 
    """
    Runs a backtest with RSI and MACD signals, filtered by EMA50 trend,
    fixed SL/TP, and no partial profit taking or time-based exits.
    """
    capital = 10000 
    risk_per_trade_capital_pct = 0.015 

    stop_loss_percent_of_entry = 0.0075 
    take_profit_percent_of_entry = 0.0125 

    brokerage_fee_percent = 0.0003 # 0.03% brokerage per trade (0.03 / 100 = 0.0003)

    current_position = None 
    entry_price = 0
    entry_time = None
    stop_loss_level = 0
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

    required_backtest_cols = [close_col, open_col, high_col, low_col, 
                              rsi_col, macd_line_col, macd_signal_col, ema_long_col] 

    for col in required_backtest_cols:
        if col not in df.columns:
            print(f"ERROR: Missing crucial column '{col}' for backtesting. Cannot run backtest.")
            return pd.DataFrame(), [capital]

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

            if exit_reason:
                trade_pnl_dollars = shares_held * (exit_price - entry_price) if current_position == 'long' else shares_held * (entry_price - exit_price)
                
                # Calculate brokerage fee at exit
                trade_value_at_exit = shares_held * exit_price
                brokerage_cost = trade_value_at_exit * brokerage_fee_percent
                
                # Deduct brokerage from PnL and capital
                trade_pnl_dollars_net = trade_pnl_dollars - brokerage_cost
                capital += trade_pnl_dollars_net 

                trade_results_summary[trade_id_counter] = trade_pnl_dollars_net # Store net PnL

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
                    'Gross PnL ($)': round(trade_pnl_dollars, 2), # Add Gross PnL for clarity
                    'Brokerage ($)': round(brokerage_cost, 2),    # Add Brokerage Cost
                    'Net PnL ($)': round(trade_pnl_dollars_net, 2), # Use net PnL here
                    'PnL (%)': round(pnl_pct, 2),
                    'Capital': round(capital, 2),
                    'Exit Reason': exit_reason
                })

                current_position = None
                entry_price = 0
                entry_time = None
                stop_loss_level = 0
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

            # EMA50 Trend Confirmation: requires EMA50 slope
            uptrend_confirmed = (current_close > current_ema50 and current_ema50 > prev_ema50) 
            downtrend_confirmed = (current_close < current_ema50 and current_ema50 < prev_ema50)

            # --- Long Entry Conditions (Trend-Following Pullback) ---
            if (uptrend_confirmed and
                current_rsi >= 35 and current_rsi <= 55 and # RSI pullback range (REFINED, from suggestion #1)
                macd_buy_crossover and
                current_macd_line < 0): # MACD cross from negative territory (UNTOUCHED)
                
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

            # --- Short Entry Conditions (Trend-Following Pullback) ---
            elif (downtrend_confirmed and
                  current_rsi >= 45 and current_rsi <= 65 and # RSI pullback range (REFINED, from suggestion #1)
                  macd_sell_crossover and
                  current_macd_line > 0): # MACD cross from positive territory (UNTOUCHED)
                
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

    final_capital = capital
    total_trades_events = len(trades)
    winning_conceptual_trades = sum(1 for pnl in trade_results_summary.values() if pnl > 0)
    losing_conceptual_trades = sum(1 for pnl in trade_results_summary.values() if pnl < 0)
    total_conceptual_trades = len(trade_results_summary)
    win_rate = (winning_conceptual_trades / total_conceptual_trades) * 100 if total_conceptual_trades > 0 else 0
    total_pnl = capital - capital_history[0]

    summary_data = {
        'Ticker': ticker_symbol,
        'Initial Capital': capital_history[0],
        'Final Capital': final_capital,
        'Total Profit/Loss ($)': total_pnl,
        'Total Trade Events': total_trades_events,
        'Winning Conceptual Trades': winning_conceptual_trades,
        'Losing Conceptual Trades': losing_conceptual_trades,
        'Win Rate (%)': win_rate,
        'Max Drawdown (%)': max_drawdown * 100
    }

    return trades, capital_history, summary_data


# --- Main Execution Loop for Multiple Tickers ---
all_summaries = []
all_trades_dfs = {}
all_capital_histories = {}

print("Starting backtest for multiple tickers...")
for current_ticker in tickers:
    print(f"\n--- Running backtest for {current_ticker} ---")
    df = get_and_prepare_data(current_ticker, interval, start_date_str, end_date_str)
    
    if not df.empty:
        df = add_indicators(df)
        if not df.empty: 
            # Pass current_ticker as ticker_symbol to run_backtest
            trades, capital_history, summary = run_backtest(df, current_ticker)
            # IMPORTANT: Only append summary if it's valid (not None)
            if summary is not None: 
                all_summaries.append(summary)
            else:
                print(f"WARNING: Skipping summary for {current_ticker} due to incomplete data/backtest.")

            # Before adding to Excel, make datetimes timezone-unaware
            if not pd.DataFrame(trades).empty:
                trades_df_clean = pd.DataFrame(trades).copy()
                if 'Time' in trades_df_clean.columns and trades_df_clean['Time'].dt.tz is not None:
                    trades_df_clean['Time'] = trades_df_clean['Time'].dt.tz_localize(None)
                if 'Entry Time' in trades_df_clean.columns and trades_df_clean['Entry Time'].dt.tz is not None:
                    trades_df_clean['Entry Time'] = trades_df_clean['Entry Time'].dt.tz_localize(None)
                all_trades_dfs[current_ticker] = trades_df_clean
            else:
                all_trades_dfs[current_ticker] = pd.DataFrame() # Ensure it's an empty DataFrame if no trades

            if not pd.DataFrame({'Capital': capital_history}, index=df.index[:len(capital_history)]).empty:
                capital_hist_df_clean = pd.DataFrame({'Capital': capital_history}, index=df.index[:len(capital_history)]).copy()
                if capital_hist_df_clean.index.tz is not None:
                    capital_hist_df_clean.index = capital_hist_df_clean.index.tz_localize(None)
                all_capital_histories[current_ticker] = capital_hist_df_clean
            else:
                all_capital_histories[current_ticker] = pd.DataFrame() # Ensure it's an empty DataFrame if no history

        else:
            print(f"Skipping {current_ticker}: DataFrame empty after indicator calculation.")
    else:
        print(f"Skipping {current_ticker}: No data or error during download/preparation.")


# --- 4. Generate Excel Output ---
output_excel_file = "backtest_results.xlsx"
print(f"\nGenerating Excel output to {output_excel_file}...")

with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print("Summary sheet written.")
    else:
        print("No summary data to write.")

    for ticker_symbol, trades_df in all_trades_dfs.items():
        if not trades_df.empty:
            sheet_name = f"{ticker_symbol}_Trades"
            trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Sheet '{sheet_name}' written.")
        else:
            print(f"No trades for {ticker_symbol} to write.")

    for ticker_symbol, capital_hist_df in all_capital_histories.items():
        if not capital_hist_df.empty:
            sheet_name = f"{ticker_symbol}_Equity"
            capital_hist_df.to_excel(writer, sheet_name=sheet_name) 
            print(f"Sheet '{sheet_name}' written.")
        else:
            print(f"No capital history for {ticker_symbol} to write.")

print("\nBacktest results saved to Excel.")

# --- Calculate and Display Average Win Rate ---
if all_summaries:
    total_win_rate = 0
    num_tickers_with_trades = 0
    for summary in all_summaries:
        # Check if 'Total Conceptual Trades' exists in the summary dictionary
        if 'Total Conceptual Trades' in summary and summary['Total Conceptual Trades'] > 0: 
            total_win_rate += summary['Win Rate (%)']
            num_tickers_with_trades += 1
    
    average_win_rate = total_win_rate / num_tickers_with_trades if num_tickers_with_trades > 0 else 0
    print(f"\n--- Overall Backtest Summary Across All Tickers ---")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\nAverage Win Rate (among tickers with trades): {average_win_rate:.2f}%")
else:
    print("\nNo overall summary available.")
