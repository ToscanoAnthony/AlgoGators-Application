import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TSMStrategyBacktest:
    def __init__(self, start_date, end_date, initial_capital=100000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.weights = {
            'AAPL': 0.5696,
            'NVDA': 0.2405,
            'AMD': 0.1899
        }

        self.threshold = 0.01
        self.transaction_cost = 0.001
        self.lookback_short = 20
        self.lookback_long = 60

        # Volatility Filter
        self.volatility_window = 20
        self.volatility_lookback = 60
        self.volatility_threshold = 1.5

        #Trend Filter
        self.trend_window = 50

        self.short_term_weight = 0.70
        self.long_term_weight = 0.30

    def download_data(self):
        tickers = ['TSM', 'AAPL', 'NVDA', 'AMD']
        raw_data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)

        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' in raw_data.columns.levels[0]:
                self.data = raw_data['Adj Close'].copy()
            else:
                self.data = raw_data['Close'].copy()
        else:
            self.data = raw_data.copy()

        self.data = self.data.dropna()

    def calculate_signals(self):
        returns_short = self.data.pct_change(periods=self.lookback_short).shift(1)
        returns_long = self.data.pct_change(periods=self.lookback_long).shift(1)

        signal_short = (
                self.weights['AAPL'] * returns_short['AAPL'] +
                self.weights['NVDA'] * returns_short['NVDA'] +
                self.weights['AMD'] * returns_short['AMD']
        )

        signal_long = (
                self.weights['AAPL'] * returns_long['AAPL'] +
                self.weights['NVDA'] * returns_long['NVDA'] +
                self.weights['AMD'] * returns_long['AMD']
        )

        self.data['Signal'] = (self.short_term_weight * signal_short +
                               self.long_term_weight * signal_long)

        # Volatility Fitter
        tsm_returns = self.data['TSM'].pct_change()
        tsm_volatility = tsm_returns.rolling(self.volatility_window).std()
        avg_volatility = tsm_volatility.rolling(self.volatility_lookback).mean()
        volatility_ratio = tsm_volatility / avg_volatility
        volatility_adjustment = np.where(volatility_ratio > self.volatility_threshold, 0.5, 1.0)

        self.data['Volatility_Ratio'] = volatility_ratio
        self.data['Volatility_Adjustment'] = volatility_adjustment

        # Trend Filter
        tsm_ma = self.data['TSM'].rolling(self.trend_window).mean()
        in_uptrend = self.data['TSM'] > tsm_ma

        self.data['TSM_MA'] = tsm_ma
        self.data['In_Uptrend'] = in_uptrend

        self.data['Position'] = 0
        self.data.loc[self.data['Signal'] >= self.threshold, 'Position'] = 1
        self.data.loc[self.data['Signal'] <= -self.threshold, 'Position'] = -1

        # Trend Filter Application
        self.data.loc[(self.data['Position'] > 0) & (~in_uptrend), 'Position'] = 0
        self.data.loc[(self.data['Position'] < 0) & (in_uptrend), 'Position'] = 0

        # Volatility Filter Application
        self.data['Position'] = self.data['Position'] * volatility_adjustment

        self.data['Position'] = self.data['Position'].replace(0, np.nan).ffill().fillna(0)
        self.data['Position_Change'] = self.data['Position'].diff().abs()

        num_trades = (self.data['Position_Change'] != 0).sum()
        long_days = (self.data['Position'] > 0).sum()
        short_days = (self.data['Position'] < 0).sum()
        neutral_days = (self.data['Position'] == 0).sum()
        high_vol_days = (volatility_ratio > self.volatility_threshold).sum()

    def backtest_strategy(self):
        self.data['TSM_Returns'] = self.data['TSM'].pct_change()

        self.data['Strategy_Returns'] = (
                self.data['Position'].shift(1) * self.data['TSM_Returns'] -
                self.data['Position_Change'].shift(1) * self.transaction_cost
        )

        self.data['TSM_Cumulative'] = (1 + self.data['TSM_Returns']).cumprod()
        self.data['Strategy_Cumulative'] = (1 + self.data['Strategy_Returns']).cumprod()

        self.data['Portfolio_Value'] = self.initial_capital * self.data['Strategy_Cumulative']
        self.data['Buy_Hold_Value'] = self.initial_capital * self.data['TSM_Cumulative']

    def calculate_metrics(self):
        strategy_returns = self.data['Strategy_Returns'].dropna()
        tsm_returns = self.data['TSM_Returns'].dropna()

        if strategy_returns.std() != 0:
            sharpe_strategy = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe_strategy = 0

        if tsm_returns.std() != 0:
            sharpe_tsm = np.sqrt(252) * tsm_returns.mean() / tsm_returns.std()
        else:
            sharpe_tsm = 0

        def calculate_max_drawdown(cumulative_returns):
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min()

        max_dd_strategy = calculate_max_drawdown(self.data['Strategy_Cumulative'].dropna())
        max_dd_tsm = calculate_max_drawdown(self.data['TSM_Cumulative'].dropna())

        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        total_trades = len(strategy_returns[strategy_returns != 0])

        if total_trades > 0:
            win_rate = len(winning_trades) / total_trades
        else:
            win_rate = 0

        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())

        if gross_loss != 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = np.inf

        total_strategy_return = (self.data['Portfolio_Value'].iloc[-1] - self.initial_capital) / self.initial_capital
        total_tsm_return = (self.data['Buy_Hold_Value'].iloc[-1] - self.initial_capital) / self.initial_capital

        years = len(self.data) / 252
        annualized_strategy = (1 + total_strategy_return) ** (1 / years) - 1
        annualized_tsm = (1 + total_tsm_return) ** (1 / years) - 1

        annual_vol_strategy = strategy_returns.std() * np.sqrt(252)
        annual_vol_tsm = tsm_returns.std() * np.sqrt(252)

        self.metrics = {
            'Total Strategy Return': total_strategy_return,
            'Total Buy & Hold Return': total_tsm_return,
            'Annualized Strategy Return': annualized_strategy,
            'Annualized B&H Return': annualized_tsm,
            'Strategy Sharpe Ratio': sharpe_strategy,
            'B&H Sharpe Ratio': sharpe_tsm,
            'Strategy Max Drawdown': max_dd_strategy,
            'B&H Max Drawdown': max_dd_tsm,
            'Strategy Volatility': annual_vol_strategy,
            'B&H Volatility': annual_vol_tsm,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Number of Trades': (self.data['Position_Change'] != 0).sum(),
            'Total Transaction Costs': (self.data['Position_Change'] * self.transaction_cost).sum()
        }
        return self.metrics

    def print_metrics(self):
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE METRICS".center(70))
        print("=" * 70)

        for key, value in self.metrics.items():
            if 'Return' in key or 'Rate' in key or 'Drawdown' in key or 'Volatility' in key:
                print(f"{key:.<50} {value:>15.2%}")
            elif 'Ratio' in key or 'Factor' in key:
                print(f"{key:.<50} {value:>15.2f}")
            elif 'Costs' in key:
                print(f"{key:.<50} ${value:>14,.2f}")
            else:
                print(f"{key:.<50} {value:>15,}")

        print("=" * 70)

        strategy_final = self.data['Portfolio_Value'].iloc[-1]
        bh_final = self.data['Buy_Hold_Value'].iloc[-1]
        outperformance = strategy_final - bh_final

        print(f"\n{'Final Portfolio Values':.<50}")
        print(f"  Strategy: ${strategy_final:,.2f}")
        print(f"  Buy & Hold: ${bh_final:,.2f}")
        print(f"  Difference: ${outperformance:,.2f} ({(outperformance / bh_final) * 100:+.2f}%)")
        print("=" * 70 + "\n")

    def plot_results(self):
        fig = plt.figure(figsize=(18, 12))

        ax1 = plt.subplot(3, 2, 1)
        self.data['Strategy_Cumulative'].plot(ax=ax1, label='Strategy', linewidth=2, color='#2E86AB')
        self.data['TSM_Cumulative'].plot(ax=ax1, label='Buy & Hold TSM', linewidth=2, alpha=0.7, color='#A23B72')
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 2, 2)
        self.data['Portfolio_Value'].plot(ax=ax2, label='Strategy', linewidth=2, color='#06A77D')
        self.data['Buy_Hold_Value'].plot(ax=ax2, label='Buy & Hold', linewidth=2, alpha=0.7, color='#F18F01')
        ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1000:.0f}K'))

        ax3 = plt.subplot(3, 2, 3)
        strat_cumulative = self.data['Strategy_Cumulative'].dropna()
        strat_running_max = strat_cumulative.expanding().max()
        strat_drawdown = (strat_cumulative - strat_running_max) / strat_running_max

        tsm_cumulative = self.data['TSM_Cumulative'].dropna()
        tsm_running_max = tsm_cumulative.expanding().max()
        tsm_drawdown = (tsm_cumulative - tsm_running_max) / tsm_running_max

        strat_drawdown.plot(ax=ax3, color='#2E86AB', linewidth=2, label='Strategy')
        tsm_drawdown.plot(ax=ax3, color='#A23B72', linewidth=2, alpha=0.7, label='Buy & Hold')
        ax3.fill_between(strat_drawdown.index, strat_drawdown, 0, alpha=0.3, color='#2E86AB')
        ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

        ax4 = plt.subplot(3, 2, 4)
        ax4_twin = ax4.twinx()

        self.data['Signal'].plot(ax=ax4, linewidth=1.5, color='#4A4E69', alpha=0.7, label='Signal')
        ax4.axhline(y=self.threshold, color='#06A77D', linestyle='--', linewidth=2,
                    label=f'Buy Threshold (+{self.threshold * 100:.1f}%)')
        ax4.axhline(y=-self.threshold, color='#C1121F', linestyle='--', linewidth=2,
                    label=f'Sell Threshold (-{self.threshold * 100:.1f}%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

        self.data['Volatility_Ratio'].plot(ax=ax4_twin, linewidth=1, color='#F18F01',
                                           alpha=0.5, label='Vol Ratio', style='--')
        ax4_twin.axhline(y=self.volatility_threshold, color='#F18F01', linestyle=':',
                         linewidth=1.5, alpha=0.7)

        ax4.set_title('Trading Signal & Volatility Filter', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Signal Value', fontsize=11)
        ax4_twin.set_ylabel('Volatility Ratio', fontsize=11, color='#F18F01')
        ax4.legend(loc='upper left', fontsize=9)
        ax4_twin.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(3, 2, 5)
        ax5_twin = ax5.twinx()

        long_positions = self.data['Position'] > 0
        short_positions = self.data['Position'] < 0

        for i in range(len(self.data) - 1):
            if long_positions.iloc[i]:
                ax5.axvspan(self.data.index[i], self.data.index[i + 1], alpha=0.2, color='#06A77D')
            elif short_positions.iloc[i]:
                ax5.axvspan(self.data.index[i], self.data.index[i + 1], alpha=0.2, color='#C1121F')

        ax5_twin.plot(self.data.index, self.data['TSM'], color='black', linewidth=2, label='TSM Price')
        ax5_twin.plot(self.data.index, self.data['TSM_MA'], color='blue', linewidth=1,
                      linestyle='--', alpha=0.6, label=f'{self.trend_window}d MA')

        ax5.set_title('Trading Positions, Price & Trend Filter', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Position (Green=Long, Red=Short)', fontsize=10)
        ax5_twin.set_ylabel('TSM Price ($)', fontsize=10)
        ax5_twin.legend(loc='upper left', fontsize=10)
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(3, 2, 6)
        monthly_returns = self.data['Strategy_Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        colors = ['#06A77D' if x > 0 else '#C1121F' for x in monthly_returns]

        x_positions = range(len(monthly_returns))
        ax6.bar(x_positions, monthly_returns.values, color=colors, width=0.8)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Monthly Return', fontsize=11)
        ax6.set_xlabel('', fontsize=11)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

        num_bars = len(monthly_returns)
        step = max(12, num_bars // 10)
        tick_positions = list(range(0, num_bars, step))
        tick_labels = [monthly_returns.index[i].strftime('%Y') for i in tick_positions]

        ax6.set_xticks(tick_positions)
        ax6.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        ax6.set_xlim(-1, num_bars)


        plt.tight_layout()
        plt.savefig('tsm_strategy_backtest.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)

    def run(self):
        self.download_data()
        self.calculate_signals()
        self.backtest_strategy()
        self.calculate_metrics()
        self.print_metrics()
        self.plot_results()
        return self.data, self.metrics


if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10 * 365)

    backtest = TSMStrategyBacktest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_capital=100000
    )
    results_data, metrics = backtest.run()
    results_data.to_csv('tsm_strategy_results.csv')

    print("COMPLETE".center(70))
    input("\nPress Enter to close...")