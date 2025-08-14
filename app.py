import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="MMM Dashboard", layout="wide")

st.title("📊 Dashboard de Marketing Mix Modeling (Dummy Data)")
st.markdown("Visualización de **Ventas vs Inversión en Medios** con ROI y Contribuciones.")

# ---------- DATOS DUMMY ----------
np.random.seed(42)
meses = pd.date_range(start="2024-01-01", periods=12, freq="M")

data = pd.DataFrame({
    "Mes": meses,
    "Ventas": np.random.randint(20000, 40000, 12),
    "TV": np.random.randint(2000, 8000, 12),
    "Radio": np.random.randint(1000, 5000, 12),
    "Digital": np.random.randint(3000, 10000, 12)
})

data["Total Inversión"] = data[["TV", "Radio", "Digital"]].sum(axis=1)
data["ROI"] = data["Ventas"] / data["Total Inversión"]

# ---------- GRÁFICO: Ventas vs Inversión ----------
fig1 = px.line(
    data, x="Mes", y=["Ventas", "Total Inversión"],
    title="Ventas vs Inversión Total",
    markers=True
)

# ---------- GRÁFICO: Contribuciones ----------
contrib = data[["TV", "Radio", "Digital"]].sum().reset_index()
contrib.columns = ["Medio", "Inversión"]

fig2 = px.pie(
    contrib, values="Inversión", names="Medio",
    title="Contribución por Canal"
)

# ---------- GRÁFICO: ROI ----------
fig3 = px.bar(
    data, x="Mes", y="ROI",
    title="Retorno de Inversión (ROI)",
    text_auto=".2f"
)

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(fig3, use_container_width=True)

st.dataframe(data.style.format({
    "Ventas": "${:,.0f}",
    "TV": "${:,.0f}",
    "Radio": "${:,.0f}",
    "Digital": "${:,.0f}",
    "Total Inversión": "${:,.0f}",
    "ROI": "{:.2f}"
}))
