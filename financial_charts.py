"""
financial_charts.py
Módulo de visualizaciones financieras con matplotlib, seaborn y plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ── Paleta y estilo global ─────────────────────────────────────────────────────
COLORS    = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']
BG_COLOR  = '#0D1117'
TXT_COLOR = '#E6EDF3'
GRID_CLR  = '#21262D'

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  BG_COLOR,
        'axes.facecolor':    '#161B22',
        'axes.edgecolor':    GRID_CLR,
        'axes.labelcolor':   TXT_COLOR,
        'xtick.color':       TXT_COLOR,
        'ytick.color':       TXT_COLOR,
        'text.color':        TXT_COLOR,
        'grid.color':        GRID_CLR,
        'grid.alpha':        0.5,
        'font.family':       'DejaVu Sans',
        'axes.titlesize':    13,
        'axes.labelsize':    10,
    })


# ── 1. Dashboard KPIs ─────────────────────────────────────────────────────────
def plot_kpi_dashboard(kpis: dict, title: str = "📊 Dashboard Financiero") -> plt.Figure:
    set_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#58A6FF', y=1.02)
    fig.patch.set_facecolor(BG_COLOR)

    kpi_colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4']
    icons       = ['💰', '💸', '📈', '🏦', '🔢', '🎯']

    for ax, (label, value), color, icon in zip(axes.flat, kpis.items(), kpi_colors, icons):
        ax.set_facecolor('#161B22')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

        # Tarjeta
        rect = plt.Rectangle((0.05, 0.1), 0.9, 0.8, linewidth=2,
                              edgecolor=color, facecolor=color + '22', zorder=1)
        ax.add_patch(rect)

        formatted = f"${value:,.2f}" if isinstance(value, float) and abs(value) > 10 else str(value)
        ax.text(0.5, 0.68, f"{icon} {label}", ha='center', va='center',
                fontsize=9, color=TXT_COLOR, fontweight='bold', zorder=2)
        ax.text(0.5, 0.38, formatted, ha='center', va='center',
                fontsize=14, color=color, fontweight='bold', zorder=2)

    plt.tight_layout()
    return fig


# ── 2. Precio de acciones + volumen ───────────────────────────────────────────
def plot_stock_price(df: pd.DataFrame, ticker: str) -> plt.Figure:
    set_dark_style()
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Precio de cierre
    ax1.plot(df.index, df['Close'], color='#2196F3', linewidth=1.5, label='Precio Cierre')
    ax1.fill_between(df.index, df['Close'], df['Close'].min(), alpha=0.15, color='#2196F3')

    # Medias móviles
    for window, color, label in [(20, '#FF9800', 'MA 20'), (50, '#F44336', 'MA 50')]:
        ma = df['Close'].rolling(window).mean()
        ax1.plot(df.index, ma, color=color, linewidth=1, linestyle='--', label=label)

    ax1.set_title(f'📈 {ticker} — Precio & Volumen (2023)', fontsize=14,
                  color='#58A6FF', fontweight='bold', pad=12)
    ax1.set_ylabel('Precio (USD)')
    ax1.legend(loc='upper left', framealpha=0.3, labelcolor=TXT_COLOR)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Volumen
    vol_colors = ['#4CAF50' if c >= o else '#F44336'
                  for c, o in zip(df['Close'], df['Open'])]
    ax2.bar(df.index, df['Volume'], color=vol_colors, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volumen')
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig


# ── 3. Correlación entre acciones ────────────────────────────────────────────
def plot_correlation_heatmap(portfolio: dict) -> plt.Figure:
    set_dark_style()

    returns = pd.DataFrame({
        ticker: data['Close'].pct_change().dropna()
        for ticker, data in portfolio.items()
    })
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5,
                annot_kws={'size': 11, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('🔗 Correlación de Retornos del Portafolio', fontsize=14,
                 color='#58A6FF', fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


# ── 4. Retornos acumulados ─────────────────────────────────────────────────────
def plot_cumulative_returns(portfolio: dict) -> plt.Figure:
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    for (ticker, data), color in zip(portfolio.items(), COLORS):
        cum_ret = (1 + data['Close'].pct_change()).cumprod() - 1
        ax.plot(cum_ret.index, cum_ret * 100, color=color,
                linewidth=1.8, label=ticker)

    ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.fill_between(cum_ret.index, 0, 0, alpha=0)  # referencia
    ax.set_title('📊 Retornos Acumulados del Portafolio (%)', fontsize=14,
                 color='#58A6FF', fontweight='bold', pad=12)
    ax.set_ylabel('Retorno (%)')
    ax.legend(loc='upper left', framealpha=0.3, labelcolor=TXT_COLOR)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig


# ── 5. Análisis de transacciones ──────────────────────────────────────────────
def plot_transactions_analysis(df: pd.DataFrame) -> plt.Figure:
    set_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('💳 Análisis de Transacciones Financieras', fontsize=15,
                 color='#58A6FF', fontweight='bold')

    # --- Gráfico 1: Donut por categoría ---
    ax1 = axes[0]
    cat_spend = df[df['Monto'] < 0].groupby('Categoría')['Monto'].sum().abs().sort_values(ascending=False)
    wedges, texts, autotexts = ax1.pie(
        cat_spend, labels=cat_spend.index, autopct='%1.1f%%',
        colors=sns.color_palette('husl', len(cat_spend)),
        pctdistance=0.82, wedgeprops=dict(width=0.5)
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color('white')
    ax1.set_title('Distribución de Egresos', color=TXT_COLOR)

    # --- Gráfico 2: Balance mensual ---
    ax2 = axes[1]
    df2 = df.copy()
    df2['Mes'] = pd.to_datetime(df2['Fecha']).dt.to_period('M')
    monthly = df2.groupby('Mes')['Monto'].sum().reset_index()
    monthly['Mes_str'] = monthly['Mes'].astype(str).str[5:]
    bar_colors = ['#4CAF50' if v >= 0 else '#F44336' for v in monthly['Monto']]
    ax2.bar(monthly['Mes_str'], monthly['Monto'], color=bar_colors, alpha=0.85, edgecolor='none')
    ax2.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_title('Balance Mensual', color=TXT_COLOR)
    ax2.set_ylabel('Monto ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)

    # --- Gráfico 3: Balance acumulado ---
    ax3 = axes[2]
    ax3.plot(range(len(df)), df['Balance_Acumulado'], color='#2196F3', linewidth=1.5)
    ax3.fill_between(range(len(df)), 0, df['Balance_Acumulado'],
                     where=df['Balance_Acumulado'] >= 0, alpha=0.2, color='#4CAF50')
    ax3.fill_between(range(len(df)), 0, df['Balance_Acumulado'],
                     where=df['Balance_Acumulado'] < 0, alpha=0.2, color='#F44336')
    ax3.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.set_title('Balance Acumulado', color=TXT_COLOR)
    ax3.set_ylabel('Balance ($)')
    ax3.set_xlabel('Num. Transacción')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── 6. Volatilidad (Riesgo) ───────────────────────────────────────────────────
def plot_risk_return(portfolio: dict) -> plt.Figure:
    set_dark_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    risks, returns_ann, tickers = [], [], []
    for ticker, data in portfolio.items():
        daily_ret = data['Close'].pct_change().dropna()
        risks.append(daily_ret.std() * np.sqrt(252) * 100)
        returns_ann.append(daily_ret.mean() * 252 * 100)
        tickers.append(ticker)

    scatter = ax.scatter(risks, returns_ann, s=200, c=COLORS[:len(tickers)],
                         zorder=5, edgecolors='white', linewidths=0.8)
    for t, r, ret in zip(tickers, risks, returns_ann):
        ax.annotate(t, (r, ret), textcoords='offset points',
                    xytext=(8, 4), fontsize=10, color=TXT_COLOR, fontweight='bold')

    ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title('⚡ Riesgo vs. Retorno Anualizado', fontsize=14,
                 color='#58A6FF', fontweight='bold', pad=12)
    ax.set_xlabel('Volatilidad Anualizada (%)')
    ax.set_ylabel('Retorno Anualizado (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig