"""
dashboard.py
🌐 Dashboard web interactivo con Streamlit.
Ejecutar con: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_generator import generate_portfolio, generate_financial_transactions, generate_kpis
from backtesting import Backtest, compute_rsi
from predictions import FinancialPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: hex → rgba  (evita el bug de '#RRGGBBAA')
# ─────────────────────────────────────────────────────────────────────────────
def rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #070B14; color: #C9D1D9; }
    section[data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #21262D;
    }
    .kpi-card {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8B949E;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #E6EDF3;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #58A6FF;
        border-bottom: 1px solid #21262D;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    div[data-testid="metric-container"] {
        background: #0D1117;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #0D1117;
        border-bottom: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        color: #8B949E;
    }
    .stTabs [aria-selected="true"] {
        color: #58A6FF !important;
        border-bottom: 2px solid #58A6FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly base layout
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0D1117',
    plot_bgcolor='#161B22',
    font=dict(family='IBM Plex Mono', color='#C9D1D9', size=11),
    hovermode='x unified',
)

# Aplicar grid a todos los ejes de una figura
def apply_axes_style(fig):
    fig.update_xaxes(gridcolor='#21262D', zerolinecolor='#21262D')
    fig.update_yaxes(gridcolor='#21262D', zerolinecolor='#21262D')
    return fig

COLORS = ['#58A6FF', '#3FB950', '#F85149', '#E3B341', '#BC8CFF']

# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    portfolio    = generate_portfolio()
    transactions = generate_financial_transactions(n=500)
    kpis         = generate_kpis(transactions)
    return portfolio, transactions, kpis

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Analytics")
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    selected_ticker = st.selectbox("Acción principal", ['JPM', 'GS', 'BBVA', 'HSBC', 'BTC-USD'])
    capital_inicial = st.number_input("Capital inicial ($)", value=10_000, step=1_000)
    comision        = st.slider("Comisión (%)", 0.0, 0.5, 0.1, 0.05) / 100

    st.markdown("---")
    st.markdown("### 🔬 Backtesting")
    sma_fast = st.slider("SMA rápida",  5,  50,  20)
    sma_slow = st.slider("SMA lenta",  20, 200,  50)
    rsi_ob   = st.slider("RSI sobrecomprado", 60, 90, 70)
    rsi_os   = st.slider("RSI sobrevendido",  10, 40, 30)

    st.markdown("---")
    st.markdown("### 🔮 Predicciones")
    pred_ticker = st.selectbox("Acción a predecir", ['GS', 'JPM', 'BBVA', 'HSBC'])
    mc_days     = st.slider("Días a proyectar (MC)", 10, 90, 30)
    mc_sims     = st.slider("Simulaciones Monte Carlo", 100, 1000, 500, 100)

    st.markdown("---")
    st.caption("Datos simulados — uso educativo")

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Cargando datos..."):
    portfolio, transactions, kpis = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 16px 0;'>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:22px;
               font-weight:700;color:#E6EDF3;letter-spacing:2px;'>
    FINANCIAL ANALYTICS
  </span>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:11px;
               color:#58A6FF;margin-left:16px;letter-spacing:3px;'>
    DASHBOARD · 2023
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
kpi_colors  = ['#3FB950', '#F85149', '#58A6FF', '#E3B341', '#BC8CFF', '#39D353']
kpi_formats = {
    'Total Ingresos':     lambda v: f"${v:,.0f}",
    'Total Egresos':      lambda v: f"${v:,.0f}",
    'Balance Neto':       lambda v: f"${v:,.0f}",
    'Ratio de Ahorro %':  lambda v: f"{v:.1f}%",
    'Num. Transacciones': lambda v: f"{int(v):,}",
    'Ticket Promedio':    lambda v: f"${v:,.2f}",
}

kpi_cols = st.columns(6)
for col, (label, value), color in zip(kpi_cols, kpis.items(), kpi_colors):
    fmt = kpi_formats.get(label, lambda v: str(v))
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{fmt(value)}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Mercado", "💳 Transacciones", "🔬 Backtesting",
    "🔮 Predicciones", "📊 Portafolio"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MERCADO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_t = portfolio[selected_ticker]

    # Candlestick + volumen
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df_t.index,
        open=df_t['Open'], high=df_t['High'],
        low=df_t['Low'],   close=df_t['Close'],
        name='OHLC',
        increasing_line_color='#3FB950',
        decreasing_line_color='#F85149',
    ), row=1, col=1)
    for w, color, name in [(20, '#E3B341', 'MA20'), (50, '#BC8CFF', 'MA50')]:
        fig.add_trace(go.Scatter(
            x=df_t.index, y=df_t['Close'].rolling(w).mean(),
            name=name, line=dict(color=color, width=1.2, dash='dot'),
        ), row=1, col=1)
    vol_colors = ['#3FB950' if c >= o else '#F85149'
                  for c, o in zip(df_t['Close'], df_t['Open'])]
    fig.add_trace(go.Bar(
        x=df_t.index, y=df_t['Volume'],
        name='Volumen', marker_color=vol_colors, opacity=0.7,
    ), row=2, col=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f'📈 {selected_ticker} — Precio & Volumen (2023)',
        xaxis_rangeslider_visible=False, height=520,
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)'),
    )
    apply_axes_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Retornos acumulados
    st.markdown('<p class="section-title">Retornos Acumulados del Portafolio</p>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    for (ticker, data), color in zip(portfolio.items(), COLORS):
        cum = (1 + data['Close'].pct_change()).cumprod() - 1
        fig2.add_trace(go.Scatter(
            x=cum.index, y=cum * 100, name=ticker,
            line=dict(color=color, width=1.8),
        ))
    fig2.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=0.8)
    fig2.update_layout(**PLOTLY_LAYOUT, title='Retornos Acumulados (%)', height=380)
    apply_axes_style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # Correlación y Riesgo-Retorno
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-title">Correlación de Retornos</p>',
                    unsafe_allow_html=True)
        ret_df = pd.DataFrame({t: d['Close'].pct_change().dropna()
                               for t, d in portfolio.items()})
        corr = ret_df.corr()
        fig3 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1)
        fig3.update_layout(**PLOTLY_LAYOUT, height=350, title='Heatmap de Correlación')
        apply_axes_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Riesgo vs Retorno Anualizado</p>',
                    unsafe_allow_html=True)
        risks, rets, ticks = [], [], []
        for t, d in portfolio.items():
            r = d['Close'].pct_change().dropna()
            risks.append(r.std() * np.sqrt(252) * 100)
            rets.append(r.mean() * 252 * 100)
            ticks.append(t)
        fig4 = px.scatter(x=risks, y=rets, text=ticks, color=ticks,
                          color_discrete_sequence=COLORS,
                          labels={'x': 'Volatilidad (%)', 'y': 'Retorno (%)'})
        fig4.update_traces(marker_size=14, textposition='top center')
        fig4.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=0.8)
        fig4.update_layout(**PLOTLY_LAYOUT, height=350,
                           showlegend=False, title='Frontera Riesgo-Retorno')
        apply_axes_style(fig4)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRANSACCIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Balance acumulado
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(
            x=transactions['Fecha'], y=transactions['Balance_Acumulado'],
            fill='tozeroy', name='Balance',
            line=dict(color='#58A6FF', width=1.5),
            fillcolor=rgba('#58A6FF', 0.12),
        ))
        fig_bal.update_layout(**PLOTLY_LAYOUT, title='Balance Acumulado', height=300)
        apply_axes_style(fig_bal)
        st.plotly_chart(fig_bal, use_container_width=True)

        # Balance mensual
        tx = transactions.copy()
        tx['Mes'] = pd.to_datetime(tx['Fecha']).dt.to_period('M').astype(str)
        monthly = tx.groupby('Mes')['Monto'].sum().reset_index()
        bar_colors = ['#3FB950' if v >= 0 else '#F85149' for v in monthly['Monto']]
        fig_mon = go.Figure(go.Bar(
            x=monthly['Mes'], y=monthly['Monto'],
            marker_color=bar_colors,
        ))
        fig_mon.add_hline(y=0, line_color='#8B949E', line_width=0.8, line_dash='dash')
        fig_mon.update_layout(**PLOTLY_LAYOUT, title='Balance por Mes', height=280)
        apply_axes_style(fig_mon)
        st.plotly_chart(fig_mon, use_container_width=True)

    with col2:
        # Donut de egresos
        egresos   = transactions[transactions['Monto'] < 0]
        cat_spend = egresos.groupby('Categoría')['Monto'].sum().abs()
        fig_don = go.Figure(go.Pie(
            labels=cat_spend.index, values=cat_spend.values,
            hole=0.55, marker_colors=px.colors.qualitative.Set3,
        ))
        fig_don.update_layout(**PLOTLY_LAYOUT, title='Distribución Egresos',
                              height=300, showlegend=True,
                              legend=dict(font=dict(size=9)))
        apply_axes_style(fig_don)
        st.plotly_chart(fig_don, use_container_width=True)

        # Top categorías
        st.markdown('<p class="section-title">Top Categorías</p>',
                    unsafe_allow_html=True)
        for cat, val in cat_spend.sort_values(ascending=False).head(5).items():
            pct = val / cat_spend.sum() * 100
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:6px 0;border-bottom:1px solid #21262D;'>
              <span style='font-size:12px;color:#C9D1D9;'>{cat}</span>
              <span style='font-family:"IBM Plex Mono",monospace;
                           font-size:12px;color:#F85149;'>${val:,.0f}
                <span style='color:#8B949E;font-size:10px;'> {pct:.0f}%</span>
              </span>
            </div>
            """, unsafe_allow_html=True)

    # Tabla de últimas transacciones
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Últimas 20 Transacciones</p>',
                unsafe_allow_html=True)
    disp = transactions.tail(20)[
        ['Fecha', 'Categoría', 'Tipo', 'Banco', 'Monto', 'Balance_Acumulado']
    ].copy()
    disp['Monto']              = disp['Monto'].apply(lambda x: f"${x:,.2f}")
    disp['Balance_Acumulado']  = disp['Balance_Acumulado'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    with st.spinner("Ejecutando backtesting..."):
        bt = Backtest(portfolio[selected_ticker], ticker=selected_ticker,
                      initial_capital=capital_inicial, commission=comision)
        bt.run_sma_crossover(fast=sma_fast, slow=sma_slow)
        bt.run_rsi_strategy(oversold=rsi_os, overbought=rsi_ob)
        bt.run_buy_and_hold()

    bt_colors = {
        'SMA Crossover': '#58A6FF',
        'RSI Strategy':  '#E3B341',
        'Buy & Hold':    '#3FB950',
    }

    # Métricas rápidas
    m_cols = st.columns(len(bt.results))
    for col, (name, result) in zip(m_cols, bt.results.items()):
        m   = result.attrs['metrics']
        ret = m['Retorno Total (%)']
        with col:
            st.metric(
                label=name,
                value=f"${m['Capital Final']:,.0f}",
                delta=f"{ret:.2f}%",
                delta_color="normal" if ret >= 0 else "inverse",
            )

    # Evolución del portafolio
    fig_bt = go.Figure()
    for name, result in bt.results.items():
        fig_bt.add_trace(go.Scatter(
            x=result.index, y=result['Portfolio_Value'],
            name=name, line=dict(color=bt_colors.get(name, '#58A6FF'), width=2),
        ))
    fig_bt.add_hline(y=capital_inicial, line_dash='dash',
                     line_color='#8B949E', line_width=0.8,
                     annotation_text="Capital inicial")
    fig_bt.update_layout(**PLOTLY_LAYOUT,
                         title='Evolución del Portafolio por Estrategia', height=380)
    apply_axes_style(fig_bt)
    st.plotly_chart(fig_bt, use_container_width=True)

    col_dd, col_tech = st.columns(2)

    with col_dd:
        # Drawdown
        fig_dd = go.Figure()
        for name, result in bt.results.items():
            c = bt_colors.get(name, '#58A6FF')
            fig_dd.add_trace(go.Scatter(
                x=result.index, y=result['Drawdown'],
                name=name, fill='tozeroy',
                line=dict(color=c, width=1.5),
                fillcolor=rgba(c, 0.18),
            ))
        fig_dd.update_layout(**PLOTLY_LAYOUT, title='Drawdown (%)', height=300)
        apply_axes_style(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_tech:
        # RSI
        rsi_s = compute_rsi(portfolio[selected_ticker]['Close'])
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi_s.index, y=rsi_s,
            line=dict(color='#E3B341', width=1.5), name='RSI',
        ))
        fig_rsi.add_hline(y=rsi_ob, line_color='#F85149', line_dash='dash',
                          line_width=0.8, annotation_text=f"SB {rsi_ob}")
        fig_rsi.add_hline(y=rsi_os, line_color='#3FB950', line_dash='dash',
                          line_width=0.8, annotation_text=f"SV {rsi_os}")
        fig_rsi.add_hrect(y0=rsi_ob, y1=100, fillcolor='#F85149',
                          opacity=0.06, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=rsi_os, fillcolor='#3FB950',
                          opacity=0.06, line_width=0)
        fig_rsi.update_layout(**PLOTLY_LAYOUT,
                              title=f'RSI — {selected_ticker}', height=300)
        apply_axes_style(fig_rsi)
        fig_rsi.update_yaxes(range=[0, 100], gridcolor='#21262D')
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Tabla comparativa
    st.markdown('<p class="section-title">Comparativa de Métricas</p>',
                unsafe_allow_html=True)
    metrics_rows = []
    for name, result in bt.results.items():
        metrics_rows.append(result.attrs['metrics'].copy())
    metrics_df = pd.DataFrame(metrics_rows).set_index('Estrategia')
    st.dataframe(metrics_df.style.format({
        'Capital Inicial':   '${:,.2f}',
        'Capital Final':     '${:,.2f}',
        'Retorno Total (%)': '{:.2f}%',
        'Retorno Anual (%)': '{:.2f}%',
        'Sharpe Ratio':      '{:.2f}',
        'Max Drawdown (%)':  '{:.2f}%',
    }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    with st.spinner("Entrenando modelos predictivos..."):
        predictor = FinancialPredictor(
            portfolio[pred_ticker], ticker=pred_ticker, test_size=0.2
        )
        pred_metrics = {
            'Reg. Lineal':     predictor.fit_linear(),
            'Reg. Polinomial': predictor.fit_polynomial(degree=3),
            'EMA':             predictor.fit_ema(span=20),
        }
        mc = predictor.monte_carlo_forecast(days=mc_days, simulations=mc_sims)

    # Tarjetas de métricas
    pred_colors = ['#58A6FF', '#BC8CFF', '#3FB950']
    m_cols = st.columns(3)
    for col, (model, m), color in zip(m_cols, pred_metrics.items(), pred_colors):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {color};">
                <div class="kpi-label">{model}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:{color};">
                    MAPE: {m['MAPE(%)']:.2f}%
                </div>
                <div style="font-size:11px;color:#8B949E;margin-top:6px;">
                    MAE: ${m['MAE']:,.2f} &nbsp;·&nbsp; RMSE: ${m['RMSE']:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico predicciones vs real
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=predictor.train['Close'].index, y=predictor.train['Close'],
        name='Entrenamiento', line=dict(color='#8B949E', width=1.2),
    ))
    fig_pred.add_trace(go.Scatter(
        x=predictor.test['Close'].index, y=predictor.test['Close'],
        name='Real', line=dict(color='#E6EDF3', width=2, dash='dash'),
    ))
    for (name, pred), color in zip(predictor.predictions.items(), pred_colors):
        fig_pred.add_trace(go.Scatter(
            x=pred.index, y=pred,
            name=f'Pred. {name}', line=dict(color=color, width=1.8),
        ))
    # Línea vertical marcando inicio del test (usando shape en lugar de add_vline)
    vline_x = str(predictor.test.index[0])[:10]
    fig_pred.add_shape(
        type='line', xref='x', yref='paper',
        x0=vline_x, x1=vline_x, y0=0, y1=1,
        line=dict(color='#8B949E', width=1, dash='dot'),
    )
    fig_pred.add_annotation(
        x=vline_x, yref='paper', y=0.98,
        text='← Test', showarrow=False,
        font=dict(color='#8B949E', size=10),
        xanchor='left',
    )
    fig_pred.update_layout(
        **PLOTLY_LAYOUT,
        title=f'Predicciones vs. Precio Real — {pred_ticker}',
        height=380,
    )
    apply_axes_style(fig_pred)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Monte Carlo
    st.markdown('<p class="section-title">Proyección Monte Carlo</p>',
                unsafe_allow_html=True)
    p = mc['percentiles']
    x = list(range(mc['days']))

    fig_mc = go.Figure()

    # Trayectorias individuales
    for path in mc['paths'][:40]:
        fig_mc.add_trace(go.Scatter(
            x=x, y=path.tolist(), mode='lines',
            line=dict(color='#58A6FF', width=0.3),
            opacity=0.07, showlegend=False, hoverinfo='skip',
        ))

    # Banda P10–P90
    fig_mc.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p['p90']) + list(p['p10'])[::-1],
        fill='toself',
        fillcolor=rgba('#58A6FF', 0.08),
        line=dict(width=0),
        name='P10–P90',
    ))
    # Banda P25–P75
    fig_mc.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p['p75']) + list(p['p25'])[::-1],
        fill='toself',
        fillcolor=rgba('#58A6FF', 0.18),
        line=dict(width=0),
        name='P25–P75',
    ))
    # Líneas de percentil
    for label, data, color, dash in [
        ('Mediana P50', p['p50'], '#58A6FF', 'solid'),
        ('P10',         p['p10'], '#F85149', 'dot'),
        ('P90',         p['p90'], '#3FB950', 'dot'),
    ]:
        fig_mc.add_trace(go.Scatter(
            x=x, y=data.tolist(), name=label,
            line=dict(color=color, width=2 if dash == 'solid' else 1, dash=dash),
        ))

    fig_mc.add_hline(
        y=mc['last_price'], line_dash='dash',
        line_color='#E3B341', line_width=1,
        annotation_text=f"Hoy: ${mc['last_price']:.2f}",
    )
    fig_mc.update_layout(
        **PLOTLY_LAYOUT,
        title=f'Monte Carlo — {pred_ticker} ({mc_days} días, {mc_sims} simulaciones)',
        height=400,
    )
    apply_axes_style(fig_mc)
    apply_axes_style(fig_mc)
    fig_mc.update_xaxes(title_text='Días hacia adelante')
    fig_mc.update_yaxes(title_text='Precio ($)')
    st.plotly_chart(fig_mc, use_container_width=True)

    # Resumen numérico MC
    mc_cols = st.columns(4)
    mc_items = [
        ('Precio actual',           f"${mc['last_price']:,.2f}"),
        ('Mediana proyectada',      f"${p['p50'][-1]:,.2f}"),
        ('Pesimista P10',           f"${p['p10'][-1]:,.2f}"),
        ('Optimista P90',           f"${p['p90'][-1]:,.2f}"),
    ]
    for col, (label, val) in zip(mc_cols, mc_items):
        with col:
            st.metric(label=label, value=val)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PORTAFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Resumen del Portafolio</p>',
                unsafe_allow_html=True)

    port_rows = []
    for ticker, df in portfolio.items():
        r = df['Close'].pct_change().dropna()
        port_rows.append({
            'Ticker':            ticker,
            'Precio actual':     round(df['Close'].iloc[-1], 2),
            'Retorno total (%)': round((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100, 2),
            'Volatilidad (%)':   round(r.std() * np.sqrt(252) * 100, 2),
            'Sharpe Ratio':      round((r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0, 2),
            'Máximo':            round(df['High'].max(), 2),
            'Mínimo':            round(df['Low'].min(), 2),
        })

    port_df = pd.DataFrame(port_rows).set_index('Ticker')
    st.dataframe(
        port_df.style.format({
            'Precio actual':     '${:,.2f}',
            'Retorno total (%)': '{:.2f}%',
            'Volatilidad (%)':   '{:.2f}%',
            'Sharpe Ratio':      '{:.2f}',
            'Máximo':            '${:,.2f}',
            'Mínimo':            '${:,.2f}',
        }).background_gradient(subset=['Retorno total (%)'], cmap='RdYlGn'),
        use_container_width=True,
    )

    # Mini charts
    st.markdown('<p class="section-title">Precio de Cierre — Todos los Activos</p>',
                unsafe_allow_html=True)
    mini_cols = st.columns(len(portfolio))
    for col, ((ticker, df), color) in zip(mini_cols, zip(portfolio.items(), COLORS)):
        last_val  = df['Close'].iloc[-1]
        first_val = df['Close'].iloc[0]
        delta     = (last_val / first_val - 1) * 100
        delta_color = '#3FB950' if delta >= 0 else '#F85149'

        fig_mini = go.Figure(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=rgba(color, 0.12),
        ))
        fig_mini.update_layout(
            **PLOTLY_LAYOUT,
            title=f"<b>{ticker}</b>  <span style='color:{delta_color};font-size:11px;'>"
                  f"{'+'if delta>=0 else ''}{delta:.1f}%</span>",
            height=180,
            margin=dict(l=10, r=10, t=45, b=10),
            showlegend=False,
        )
        apply_axes_style(fig_mini)
        fig_mini.update_xaxes(showticklabels=False, gridcolor='#21262D')
        fig_mini.update_yaxes(showticklabels=False, gridcolor='#21262D')
        with col:
            st.plotly_chart(fig_mini, use_container_width=True)