"""
data_generator.py
Genera datos financieros simulados realistas para el análisis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_stock_prices(ticker: str, start_date: str, days: int = 365, initial_price: float = 100.0) -> pd.DataFrame:
    """Genera precios de acciones usando movimiento browniano geométrico."""
    np.random.seed(hash(ticker) % 1000)

    dates = pd.date_range(start=start_date, periods=days, freq='B')  # días hábiles
    mu = 0.0003       # retorno diario esperado
    sigma = 0.018     # volatilidad diaria

    returns = np.random.normal(mu, sigma, days)
    price_series = initial_price * np.exp(np.cumsum(returns))

    # Generar OHLCV (Open, High, Low, Close, Volume)
    df = pd.DataFrame({'Date': dates, 'Close': price_series})
    df['Open']   = df['Close'].shift(1).fillna(initial_price) * np.random.uniform(0.998, 1.002, days)
    df['High']   = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.001, 1.015, days)
    df['Low']    = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.985, 0.999, days)
    df['Volume'] = np.random.randint(500_000, 5_000_000, days)
    df['Ticker'] = ticker

    return df.set_index('Date')


def generate_portfolio() -> dict[str, pd.DataFrame]:
    """Genera un portafolio con múltiples acciones del sector financiero."""
    stocks = {
        'BBVA':   {'price': 8.50,   'sector': 'Banca'},
        'HSBC':   {'price': 42.00,  'sector': 'Banca'},
        'GS':     {'price': 380.00, 'sector': 'Inversión'},
        'JPM':    {'price': 155.00, 'sector': 'Banca'},
        'BTC-USD':{'price': 42000,  'sector': 'Crypto'},
    }

    portfolio = {}
    for ticker, info in stocks.items():
        portfolio[ticker] = generate_stock_prices(
            ticker=ticker,
            start_date='2023-01-01',
            days=365,
            initial_price=info['price']
        )
    return portfolio


def generate_financial_transactions(n: int = 500) -> pd.DataFrame:
    """Genera transacciones financieras simuladas."""
    np.random.seed(42)

    categories = ['Nómina', 'Renta', 'Servicios', 'Alimentación',
                  'Inversiones', 'Préstamos', 'Dividendos', 'Transferencias']
    weights     = [0.15, 0.10, 0.10, 0.15, 0.20, 0.10, 0.10, 0.10]

    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in np.random.uniform(0, 365, n)]

    data = {
        'Fecha':     sorted(dates),
        'Categoría': np.random.choice(categories, n, p=weights),
        'Monto':     np.round(np.random.lognormal(mean=7, sigma=1.5, size=n), 2),
        'Tipo':      np.random.choice(['Ingreso', 'Egreso'], n, p=[0.4, 0.6]),
        'Banco':     np.random.choice(['BBVA', 'HSBC', 'Santander', 'Banamex'], n),
    }

    df = pd.DataFrame(data)
    df['Monto'] = df.apply(lambda r: r['Monto'] if r['Tipo'] == 'Ingreso' else -r['Monto'], axis=1)
    df['Balance_Acumulado'] = df['Monto'].cumsum()
    return df


def generate_kpis(transactions: pd.DataFrame) -> dict:
    """Calcula KPIs financieros clave."""
    ingresos  = transactions[transactions['Monto'] > 0]['Monto'].sum()
    egresos   = transactions[transactions['Monto'] < 0]['Monto'].sum()
    balance   = ingresos + egresos
    ratio_ahorro = (balance / ingresos * 100) if ingresos else 0

    return {
        'Total Ingresos':    round(ingresos, 2),
        'Total Egresos':     round(abs(egresos), 2),
        'Balance Neto':      round(balance, 2),
        'Ratio de Ahorro %': round(ratio_ahorro, 2),
        'Num. Transacciones': len(transactions),
        'Ticket Promedio':   round(transactions['Monto'].abs().mean(), 2),
    }