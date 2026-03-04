"""
predictions.py
🔮 Predicciones de precios financieros.
Modelos incluidos:
  - Regresión Lineal        (tendencia base)
  - Media Móvil Exponencial (suavizado adaptativo)
  - Regresión Polinomial    (captura curvas)
  - Intervalos de confianza Monte Carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error


BG_COLOR = '#0D1117'
TXT_CLR  = '#E6EDF3'
GRID_CLR = '#21262D'


def _dark():
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR, 'axes.facecolor': '#161B22',
        'axes.edgecolor': GRID_CLR,   'axes.labelcolor': TXT_CLR,
        'xtick.color': TXT_CLR,       'ytick.color': TXT_CLR,
        'text.color': TXT_CLR,        'grid.color': GRID_CLR,
        'grid.alpha': 0.4,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Preparación de features
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Agrega indicadores técnicos como features para los modelos."""
    feat = df[['Close']].copy()
    feat['Returns']    = feat['Close'].pct_change()
    feat['MA_10']      = feat['Close'].rolling(10).mean()
    feat['MA_20']      = feat['Close'].rolling(20).mean()
    feat['Volatility'] = feat['Returns'].rolling(10).std()
    feat['Momentum']   = feat['Close'] / feat['Close'].shift(10) - 1

    for i in range(1, lags + 1):
        feat[f'Lag_{i}'] = feat['Close'].shift(i)

    feat['Target']     = feat['Close'].shift(-1)   # precio del día siguiente
    feat['Day_Index']  = np.arange(len(feat))
    return feat.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de predicción
# ─────────────────────────────────────────────────────────────────────────────

class FinancialPredictor:
    """Contenedor de modelos predictivos para series de precios."""

    def __init__(self, df: pd.DataFrame, ticker: str = 'ASSET', test_size: float = 0.2):
        self.ticker    = ticker
        self.df        = df.copy()
        self.feat_df   = build_features(df)
        self.test_size = test_size
        self._split()
        self.models: dict = {}
        self.predictions: dict = {}

    def _split(self):
        n = len(self.feat_df)
        split = int(n * (1 - self.test_size))
        self.train = self.feat_df.iloc[:split]
        self.test  = self.feat_df.iloc[split:]

    # ── Modelo 1: Regresión Lineal ────────────────────────────────────────
    def fit_linear(self):
        X_train = self.train[['Day_Index']]
        y_train = self.train['Close']
        X_test  = self.test[['Day_Index']]
        y_test  = self.test['Close']

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.models['Linear']      = model
        self.predictions['Linear'] = pd.Series(y_pred, index=self.test.index)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        return {'MAE': round(mae,2), 'RMSE': round(rmse,2), 'MAPE(%)': round(mape,2)}

    # ── Modelo 2: Regresión Polinomial (grado 3) ──────────────────────────
    def fit_polynomial(self, degree: int = 3):
        X_train = self.train[['Day_Index']]
        y_train = self.train['Close']
        X_test  = self.test[['Day_Index']]
        y_test  = self.test['Close']

        poly  = PolynomialFeatures(degree=degree)
        X_tr  = poly.fit_transform(X_train)
        X_te  = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        self.models['Polynomial']      = (model, poly)
        self.predictions['Polynomial'] = pd.Series(y_pred, index=self.test.index)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        return {'MAE': round(mae,2), 'RMSE': round(rmse,2), 'MAPE(%)': round(mape,2)}

    # ── Modelo 3: EMA (suavizado exponencial) ─────────────────────────────
    def fit_ema(self, span: int = 20):
        ema = self.df['Close'].ewm(span=span, adjust=False).mean()
        self.predictions['EMA'] = ema.loc[self.test.index]
        y_test = self.test['Close']
        y_pred = self.predictions['EMA'].values
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        return {'MAE': round(mae,2), 'RMSE': round(rmse,2), 'MAPE(%)': round(mape,2)}

    # ── Proyección futura (Monte Carlo) ───────────────────────────────────
    def monte_carlo_forecast(self, days: int = 30, simulations: int = 500) -> dict:
        """Simula posibles trayectorias futuras del precio."""
        np.random.seed(42)
        returns   = self.df['Close'].pct_change().dropna()
        mu        = returns.mean()
        sigma     = returns.std()
        last_price = self.df['Close'].iloc[-1]

        paths = np.zeros((simulations, days))
        for s in range(simulations):
            r = np.random.normal(mu, sigma, days)
            paths[s] = last_price * np.exp(np.cumsum(r))

        percentiles = {
            'p10': np.percentile(paths, 10, axis=0),
            'p25': np.percentile(paths, 25, axis=0),
            'p50': np.percentile(paths, 50, axis=0),   # mediana
            'p75': np.percentile(paths, 75, axis=0),
            'p90': np.percentile(paths, 90, axis=0),
        }
        return {'paths': paths, 'percentiles': percentiles,
                'last_price': last_price, 'days': days}


# ─────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(predictor: FinancialPredictor, metrics: dict) -> plt.Figure:
    _dark()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'hspace': 0.4})
    fig.suptitle(f'🔮 Predicciones de Precio — {predictor.ticker}',
                 fontsize=15, color='#58A6FF', fontweight='bold')

    colors = {'Linear': '#FF9800', 'Polynomial': '#9C27B0', 'EMA': '#4CAF50'}

    # ── Subplot 1: Real vs Predicho ───────────────────────────────────────
    ax1 = axes[0]
    train_close = predictor.train['Close']
    test_close  = predictor.test['Close']

    ax1.plot(train_close.index, train_close, color='#2196F3',
             linewidth=1.5, label='Entrenamiento', alpha=0.8)
    ax1.plot(test_close.index, test_close, color='white',
             linewidth=1.8, label='Real (test)', linestyle='--')

    for name, pred in predictor.predictions.items():
        ax1.plot(pred.index, pred, color=colors.get(name,'red'),
                 linewidth=1.5, label=f'Predicción {name}', alpha=0.9)

    ax1.axvline(predictor.test.index[0], color='grey',
                linewidth=1, linestyle=':', alpha=0.7)
    ax1.text(predictor.test.index[0], ax1.get_ylim()[0],
             '  Test →', color='grey', fontsize=8, va='bottom')
    ax1.set_title('Comparación Real vs. Modelos', color=TXT_CLR)
    ax1.set_ylabel('Precio ($)'); ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.3, fontsize=8)

    # ── Subplot 2: Tabla de métricas ──────────────────────────────────────
    ax2 = axes[1]
    ax2.axis('off')
    rows, row_labels = [], []
    for model_name, m in metrics.items():
        rows.append([str(v) for v in m.values()])
        row_labels.append(model_name)

    col_labels = list(list(metrics.values())[0].keys())
    tbl = ax2.table(cellText=rows, colLabels=col_labels,
                    rowLabels=row_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#161B22' if r > 0 else '#21262D')
        cell.set_text_props(color=TXT_CLR)
        cell.set_edgecolor(GRID_CLR)

    ax2.set_title('📋 Métricas de Error por Modelo', color=TXT_CLR, pad=20)
    plt.tight_layout()
    return fig


def plot_monte_carlo(mc: dict, ticker: str) -> plt.Figure:
    _dark()
    fig, ax = plt.subplots(figsize=(12, 6))
    p = mc['percentiles']
    x = np.arange(mc['days'])

    # Bandas de confianza
    ax.fill_between(x, p['p10'], p['p90'], alpha=0.15, color='#2196F3', label='P10–P90')
    ax.fill_between(x, p['p25'], p['p75'], alpha=0.25, color='#2196F3', label='P25–P75')
    ax.plot(x, p['p50'], color='#2196F3',  linewidth=2,   label='Mediana (P50)')
    ax.plot(x, p['p10'], color='#F44336',  linewidth=0.8, linestyle='--', alpha=0.7)
    ax.plot(x, p['p90'], color='#4CAF50',  linewidth=0.8, linestyle='--', alpha=0.7)

    # Algunas trayectorias individuales
    for path in mc['paths'][:30]:
        ax.plot(x, path, color='white', linewidth=0.3, alpha=0.1)

    ax.axhline(mc['last_price'], color='white', linewidth=1,
               linestyle=':', alpha=0.6, label=f"Precio actual: ${mc['last_price']:.2f}")

    ax.set_title(f'🎲 Proyección Monte Carlo — {ticker} ({mc["days"]} días)',
                 fontsize=14, color='#58A6FF', fontweight='bold', pad=12)
    ax.set_xlabel('Días hacia adelante')
    ax.set_ylabel('Precio ($)')
    ax.legend(loc='upper left', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig