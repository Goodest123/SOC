# SOC
My First Stock Trading Project: Learning and Strategy Development

Introduction
This project describes my first experience with intraday stock trading. It covers how I built my strategy, what I learned about technical indicators, and the real-world practice I gained.

Technical Indicators: My Core Tools
I started by learning about important technical indicators, which are key for understanding the market:

MACD (Moving Average Convergence Divergence): This indicator helped me understand momentum, showing how strong a trend is and when it might change direction.

RSI (Relative Strength Index): This was essential for figuring out when a stock was bought or sold too much, helping me predict when prices might pull back or bounce.

EMA (Exponential Moving Average): Specifically the EMA50, this became my main tool for identifying the overall trend. Its position relative to the price and whether it was going up or down helped confirm the trend's direction and strength.

Strategy Development: Making it Better Through Testing
My first trading strategy was too complicated. By repeatedly testing it with past data (backtesting), I made it simpler. The strategy now uses two main ideas:

Trading Pullbacks in a Trend: I learned to enter trades when the price temporarily dipped (in an uptrend) or went up (in a downtrend), expecting the main trend to continue. This involved using specific RSI ranges for these pullbacks and MACD crosses that showed momentum was returning to the trend. The EMA50's slope was crucial for confirming strong trends.

Trading Trend Reversals: I also added conditions to spot when a big change in the market's direction was happening. This involved the price crossing the EMA50, RSI signals showing the old trend was running out of steam, and MACD confirming a strong shift in momentum around the zero line.

I also included a fixed 0.75% Stop Loss (SL) (to limit losses) and 1.25% Take Profit (TP) (to lock in gains) for every trade. A 0.03% brokerage fee was also added for more realistic cost tracking.

Manual Trading: Lessons from Real Experience
Testing my strategy with past data built a foundation, but trading manually with small amounts of real money gave me important insights:

Slippage & Execution: I learned that trades don't always happen at the exact price I want, showing why careful trading is important.

Staying Disciplined: Trading with real money brought out emotions like fear and greed. This taught me to stick strictly to my trading rules.

Market Details: I started to understand how fast the market moves in real-time, and how news can suddenly change things.

Immediate Feedback: Seeing direct results from real trades quickly taught me what worked and what didn't, helping me improve.

What I Learned

This project taught me key trading rules:

Managing Risk: Protecting my money by using stop-losses is most important.

Patience & Discipline: Waiting for the right trading opportunities and following my plan is crucial for success.

Always Learning: The market changes all the time, so I need to keep learning and adapting.

Using Data: Analyzing data and trade results carefully is very valuable.
