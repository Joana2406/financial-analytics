"""
main.py
🏦 Financial Data Analytics — Punto de entrada principal
Incluye: análisis, backtesting, predicciones y exportación a Excel.

Ejecutar con:
    python main.py
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data_generator import (
    generate_portfolio,
    generate_financial_transactions,
    generate_kpis,
)
from financial_charts import (
    plot_kpi_dashboard,
    plot_stock_price,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_transactions_analysis,
    plot_risk_return,
)
from backtesting import (
    Backtest,
    plot_backtest_results,
    plot_rsi_chart,
)
from predictions import (
    FinancialPredictor,
    plot_predictions,
    plot_monte_carlo,
)
from excel_export import export_to_excel


# ─────────────────────────────────────────────────────────────────────────────
def banner(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)


def save(fig: plt.Figure, filename: str):
    fig.savefig(f'outputs/{filename}', dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close(fig)
    print(f"    ✅ outputs/{filename}")


def main():
    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("=" * 55)
    print("   📊  FINANCIAL DATA ANALYTICS — Python Project")
    print("        Backtesting · Predicciones · Excel")
    print("=" * 55)

    # ══════════════════════════════════════════════════════
    # 1. GENERAR DATOS
    # ══════════════════════════════════════════════════════
    banner("1/5  Generando datos financieros...")
    portfolio    = generate_portfolio()
    transactions = generate_financial_transactions(n=500)
    kpis         = generate_kpis(transactions)

    transactions.to_csv('data/transactions.csv', index=False)
    for ticker, df in portfolio.items():
        df.to_csv(f'data/{ticker.replace("-","_")}.csv')
    print("  ✅ Datos guardados en /data")

    print("\n  ┌──────────────────────────────────────────────┐")
    for k, v in kpis.items():
        label = k.ljust(24)
        val   = f"${v:,.2f}" if isinstance(v, float) and abs(v) > 10 else str(v)
        print(f"  │  {label} {val.rjust(16)}  │")
    print("  └──────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════
    # 2. GRÁFICOS DE ANÁLISIS
    # ══════════════════════════════════════════════════════
    banner("2/5  Generando gráficos de análisis...")
    save(plot_kpi_dashboard(kpis),                 '01_kpi_dashboard.png')
    save(plot_stock_price(portfolio['JPM'], 'JPM'), '02_stock_JPM.png')
    save(plot_cumulative_returns(portfolio),        '03_cumulative_returns.png')
    save(plot_correlation_heatmap(portfolio),       '04_correlation.png')
    save(plot_transactions_analysis(transactions),  '05_transactions.png')
    save(plot_risk_return(portfolio),               '06_risk_return.png')

    # ══════════════════════════════════════════════════════
    # 3. BACKTESTING
    # ══════════════════════════════════════════════════════
    banner("3/5  Ejecutando Backtesting de estrategias (JPM)...")
    bt = Backtest(portfolio['JPM'], ticker='JPM',
                  initial_capital=10_000, commission=0.001)

    bt.run_sma_crossover(fast=20, slow=50)
    bt.run_rsi_strategy(oversold=30, overbought=70)
    bt.run_buy_and_hold()

    print("\n  Resumen de estrategias:")
    for name, result in bt.results.items():
        m = result.attrs['metrics']
        print(f"    {name:<20} Retorno: {m['Retorno Total (%)']:>7.2f}%  "
              f"Sharpe: {m['Sharpe Ratio']:>5.2f}  "
              f"Max DD: {m['Max Drawdown (%)']:>7.2f}%")

    save(plot_backtest_results(bt),               '07_backtest_results.png')
    save(plot_rsi_chart(portfolio['JPM'], 'JPM'),  '08_technical_analysis.png')

    # ══════════════════════════════════════════════════════
    # 4. PREDICCIONES
    # ══════════════════════════════════════════════════════
    banner("4/5  Entrenando modelos predictivos (GS)...")
    predictor = FinancialPredictor(portfolio['GS'], ticker='GS', test_size=0.2)

    metrics = {
        'Regresión Lineal':     predictor.fit_linear(),
        'Regresión Polinomial': predictor.fit_polynomial(degree=3),
        'EMA (span=20)':        predictor.fit_ema(span=20),
    }

    print("\n  Métricas de error:")
    for model, m in metrics.items():
        print(f"    {model:<25}  MAE={m['MAE']:>8.2f}  "
              f"RMSE={m['RMSE']:>8.2f}  MAPE={m['MAPE(%)']:>6.2f}%")

    best = min(metrics, key=lambda k: metrics[k]['MAPE(%)'])
    print(f"\n  ⭐ Mejor modelo: {best}")

    mc = predictor.monte_carlo_forecast(days=30, simulations=500)
    print(f"  Monte Carlo — mediana a 30 días: ${mc['percentiles']['p50'][-1]:,.2f}")
    print(f"  Rango P10–P90: ${mc['percentiles']['p10'][-1]:,.2f}"
          f" – ${mc['percentiles']['p90'][-1]:,.2f}")

    save(plot_predictions(predictor, metrics), '09_predictions.png')
    save(plot_monte_carlo(mc, 'GS'),           '10_monte_carlo.png')

    # ══════════════════════════════════════════════════════
    # 5. EXPORTAR A EXCEL
    # ══════════════════════════════════════════════════════
    banner("5/5  Exportando reporte a Excel...")
    export_to_excel(
        kpis         = kpis,
        portfolio    = portfolio,
        transactions = transactions,
        bt_results   = bt.results,
        predictor    = predictor,
        pred_metrics = metrics,
        output_path  = 'outputs/financial_report.xlsx',
    )

    print("\n" + "=" * 55)
    print("   ✅  Análisis completado exitosamente.")
    print()
    print("   📁  Archivos generados en /outputs:")
    print("       01–06  Gráficos de análisis (.png)")
    print("       07–08  Backtesting (.png)")
    print("       09–10  Predicciones y Monte Carlo (.png)")
    print("       financial_report.xlsx  ← Reporte Excel completo")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()