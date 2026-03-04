"""
backtesting.py
📈 Motor de backtesting para estrategias de trading financiero.
Estrategias incluidas:
  - SMA Crossover  (Media Móvil Simple cruzada)
  - RSI Mean Reversion
  - Buy & Hold      (benchmark)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


BG_COLOR  = '#0D1117'
TXT_COLOR = '#E6EDF3'
GRID_CLR  = '#21262D'

def _dark():
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR, 'axes.facecolor': '#161B22',
        'axes.edgecolor': GRID_CLR,   'axes.labelcolor': TXT_COLOR,
        'xtick.color': TXT_COLOR,     'ytick.color': TXT_COLOR,
        'text.color': TXT_COLOR,      'grid.color': GRID_CLR,
        'grid.alpha': 0.4,            'font.family': 'DejaVu Sans',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Indicadores técnicos
# ─────────────────────────────────────────────────────────────────────────────

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal de Backtesting
# ─────────────────────────────────────────────────────────────────────────────

class Backtest:
    """Simulador de estrategias de trading sobre datos OHLCV."""

    def __init__(self, df: pd.DataFrame, ticker: str = 'ASSET',
                 initial_capital: float = 10_000.0, commission: float = 0.001):
        self.df              = df.copy()
        self.ticker          = ticker
        self.initial_capital = initial_capital
        self.commission      = commission   # 0.1 % por operación
        self.results: dict   = {}

    # ── Estrategia 1: SMA Crossover ──────────────────────────────────────────
    def run_sma_crossover(self, fast: int = 20, slow: int = 50) -> pd.DataFrame:
        df = self.df.copy()
        df['SMA_fast'] = compute_sma(df['Close'], fast)
        df['SMA_slow'] = compute_sma(df['Close'], slow)

        # Señal: 1 = compra, -1 = venta, 0 = sin posición
        df['Signal'] = 0
        df.loc[df['SMA_fast'] > df['SMA_slow'], 'Signal'] = 1
        df.loc[df['SMA_fast'] < df['SMA_slow'], 'Signal'] = -1
        df['Position'] = df['Signal'].diff()   # cambio de posición

        result = self._simulate(df, f'SMA({fast}/{slow})')
        self.results['SMA Crossover'] = result
        return result

    # ── Estrategia 2: RSI Mean Reversion ─────────────────────────────────────
    def run_rsi_strategy(self, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
        df = self.df.copy()
        df['RSI'] = compute_rsi(df['Close'])

        df['Signal'] = 0
        df.loc[df['RSI'] < oversold,   'Signal'] = 1   # sobrevendido → comprar
        df.loc[df['RSI'] > overbought, 'Signal'] = -1  # sobrecomprado → vender
        df['Position'] = df['Signal'].diff()

        result = self._simulate(df, f'RSI({oversold}/{overbought})')
        self.results['RSI Strategy'] = result
        return result

    # ── Buy & Hold (benchmark) ───────────────────────────────────────────────
    def run_buy_and_hold(self) -> pd.DataFrame:
        df = self.df.copy()
        df['Signal']   = 1
        df['Position'] = 0
        df.iloc[0, df.columns.get_loc('Position')] = 1   # compra al inicio

        result = self._simulate(df, 'Buy & Hold')
        self.results['Buy & Hold'] = result
        return result

    # ── Motor de simulación ──────────────────────────────────────────────────
    def _simulate(self, df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        capital    = self.initial_capital
        shares     = 0.0
        portfolio  = []
        trade_log  = []

        for i, row in df.iterrows():
            price = row['Close']

            # Comprar
            if row.get('Position', 0) == 1 and capital > 0:
                shares_bought = (capital * (1 - self.commission)) / price
                shares       += shares_bought
                capital       = 0
                trade_log.append({'Date': i, 'Action': 'BUY',
                                  'Price': price, 'Shares': shares_bought})
            # Vender
            elif row.get('Position', 0) == -1 and shares > 0:
                proceeds = shares * price * (1 - self.commission)
                capital += proceeds
                trade_log.append({'Date': i, 'Action': 'SELL',
                                  'Price': price, 'Shares': shares,
                                  'PnL': proceeds - self.initial_capital})
                shares = 0

            total_value = capital + shares * price
            portfolio.append({'Date': i, 'Portfolio_Value': total_value,
                               'Cash': capital, 'Shares': shares, 'Price': price})

        port_df = pd.DataFrame(portfolio).set_index('Date')
        port_df['Returns']    = port_df['Portfolio_Value'].pct_change()
        port_df['Cumulative'] = (port_df['Portfolio_Value'] / self.initial_capital - 1) * 100
        port_df['Drawdown']   = self._max_drawdown_series(port_df['Portfolio_Value'])
        port_df['Strategy']   = strategy_name

        # Métricas de resumen
        total_ret = (port_df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        ann_ret   = total_ret / (len(port_df) / 252)
        sharpe    = (port_df['Returns'].mean() / port_df['Returns'].std()) * np.sqrt(252) \
                    if port_df['Returns'].std() > 0 else 0
        max_dd    = port_df['Drawdown'].min()
        n_trades  = len([t for t in trade_log if t['Action'] == 'SELL'])

        port_df.attrs['metrics'] = {
            'Estrategia':         strategy_name,
            'Capital Inicial':    self.initial_capital,
            'Capital Final':      round(port_df['Portfolio_Value'].iloc[-1], 2),
            'Retorno Total (%)':  round(total_ret, 2),
            'Retorno Anual (%)':  round(ann_ret, 2),
            'Sharpe Ratio':       round(sharpe, 2),
            'Max Drawdown (%)':   round(max_dd, 2),
            'Num. Operaciones':   n_trades,
        }
        port_df.attrs['trades'] = pd.DataFrame(trade_log)
        return port_df

    @staticmethod
    def _max_drawdown_series(values: pd.Series) -> pd.Series:
        peak = values.cummax()
        return ((values - peak) / peak) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Visualizaciones de Backtesting
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_results(bt: Backtest) -> plt.Figure:
    _dark()
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'🔬 Backtesting — {bt.ticker}', fontsize=16,
                 color='#58A6FF', fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    colors = {'SMA Crossover': '#2196F3', 'RSI Strategy': '#FF9800', 'Buy & Hold': '#4CAF50'}

    # ── Panel 1: Valor del portafolio ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for name, result in bt.results.items():
        ax1.plot(result.index, result['Portfolio_Value'],
                 color=colors.get(name, 'white'), linewidth=1.8, label=name)
    ax1.axhline(bt.initial_capital, color='white', linewidth=0.8,
                linestyle='--', alpha=0.5, label='Capital inicial')
    ax1.set_title('Evolución del Portafolio ($)', color=TXT_COLOR)
    ax1.set_ylabel('Valor ($)')
    ax1.legend(loc='upper left', framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Retornos acumulados % ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for name, result in bt.results.items():
        ax2.plot(result.index, result['Cumulative'],
                 color=colors.get(name, 'white'), linewidth=1.5, label=name)
    ax2.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_title('Retorno Acumulado (%)', color=TXT_COLOR)
    ax2.set_ylabel('%'); ax2.grid(True, alpha=0.3)
    ax2.legend(framealpha=0.3, fontsize=8)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for name, result in bt.results.items():
        ax3.fill_between(result.index, result['Drawdown'], 0,
                         alpha=0.4, color=colors.get(name, 'grey'), label=name)
    ax3.set_title('Drawdown (%)', color=TXT_COLOR)
    ax3.set_ylabel('%'); ax3.grid(True, alpha=0.3)
    ax3.legend(framealpha=0.3, fontsize=8)

    # ── Panel 4: Tabla comparativa de métricas ────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    if bt.results:
        metrics_rows = []
        col_labels   = None
        for name, result in bt.results.items():
            m = result.attrs.get('metrics', {})
            if col_labels is None:
                col_labels = list(m.keys())[1:]   # omitir 'Estrategia'
            metrics_rows.append([str(v) for v in list(m.values())[1:]])

        row_labels = list(bt.results.keys())
        tbl = ax4.table(cellText=metrics_rows, colLabels=col_labels,
                        rowLabels=row_labels, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#161B22' if r > 0 else '#21262D')
            cell.set_text_props(color=TXT_COLOR)
            cell.set_edgecolor(GRID_CLR)

        ax4.set_title('📋 Resumen de Métricas', color=TXT_COLOR, pad=15)

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return fig


def plot_rsi_chart(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Gráfico de precio + RSI + MACD."""
    _dark()
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.08)

    rsi            = compute_rsi(df['Close'])
    macd, sig, hist = compute_macd(df['Close'])

    # Precio
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], color='#2196F3', linewidth=1.5)
    ax1.set_title(f'📉 Análisis Técnico — {ticker}', color='#58A6FF',
                  fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Precio'); ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, rsi, color='#FF9800', linewidth=1.2)
    ax2.axhline(70, color='#F44336', linewidth=0.8, linestyle='--', alpha=0.8)
    ax2.axhline(30, color='#4CAF50', linewidth=0.8, linestyle='--', alpha=0.8)
    ax2.fill_between(df.index, 70, rsi, where=(rsi > 70), alpha=0.25, color='#F44336')
    ax2.fill_between(df.index, 30, rsi, where=(rsi < 30), alpha=0.25, color='#4CAF50')
    ax2.set_ylabel('RSI'); ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.3)
    ax2.text(df.index[-1], 72, 'Sobrecomprado', color='#F44336', fontsize=7, ha='right')
    ax2.text(df.index[-1], 25, 'Sobrevendido',  color='#4CAF50', fontsize=7, ha='right')
    plt.setp(ax2.get_xticklabels(), visible=False)

    # MACD
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, macd, color='#2196F3', linewidth=1.2, label='MACD')
    ax3.plot(df.index, sig,  color='#FF9800', linewidth=1.0, label='Señal')
    bar_colors = ['#4CAF50' if h >= 0 else '#F44336' for h in hist]
    ax3.bar(df.index, hist, color=bar_colors, alpha=0.6, width=0.8)
    ax3.axhline(0, color='white', linewidth=0.6, linestyle='--', alpha=0.5)
    ax3.set_ylabel('MACD'); ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', framealpha=0.3, fontsize=8)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig