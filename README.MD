# 📊 Financial Data Analytics - Python Project

## Requisitos previos
- Python 3.8+
- Visual Studio Code

---

## 🚀 Setup desde cero (comandos en terminal)

```bash
# 1. Crear carpeta del proyecto
mkdir financial_analytics
cd financial_analytics

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly openpyxl

# 5. Abrir en Visual Studio Code
code .
```

---

## 📁 Estructura del proyecto

```
financial_analytics/
│
├── data/
│   └── (aquí se generan los datos automáticamente)
│
├── main.py                  ← Análisis principal
├── data_generator.py        ← Generador de datos financieros
├── financial_charts.py      ← Gráficos y visualizaciones
├── requirements.txt         ← Dependencias
└── README.md
```

---

## ▶️ Cómo ejecutar

```bash
# Ejecutar el análisis completo
python main.py
```

---

## 📦 Extensiones recomendadas para VS Code
- **Python** (Microsoft)
- **Pylance**
- **Jupyter** (para notebooks interactivos)