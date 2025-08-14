import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(
    page_title="🚀 MMM Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para mejorar el diseño
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con animación
st.markdown("""
# 🚀 Marketing Mix Model Dashboard
### Análisis Avanzado de Performance y ROI
---
""")

# Funciones auxiliares
def ajustar_ols(df, y_col, x_cols):
    """Función para ajustar modelo OLS"""
    df_model = df[x_cols].copy()
    df_model['Y'] = df[y_col]
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
    Y = df_model['Y']
    X = df_model.drop(columns=['Y'])
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model

def contribuciones(data, model_fit):
    """Función para calcular contribuciones"""
    X = sm.add_constant(data)
    params = model_fit.params.to_numpy()
    X_mat = X.to_numpy()
    contr = pd.DataFrame(X_mat * params)
    contr.iloc[:, 0] += model_fit.resid
    contr_tot = contr.sum()
    contr_pct = (contr_tot / contr_tot.sum()) * 100
    contr_abs = np.abs(contr_tot)
    contr_tot.index = model_fit.params.index
    contr_pct.index = model_fit.params.index
    contr_abs.index = model_fit.params.index
    salida = pd.concat([contr_tot, contr_pct, contr_abs], axis=1)
    salida.columns = ["Total", "%", "abs"]
    salida.sort_values(by="abs", ascending=False, inplace=True)
    salida = salida.drop("abs", axis=1)
    return salida

def format_number(num):
    """Formatear números para display"""
    if pd.isna(num):
        return "N/A"
    if abs(num) >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_percentage(num):
    """Formatear porcentajes"""
    return f"{num:.2f}%" if pd.notna(num) else "N/A"

# Sidebar para cargar datos
st.sidebar.markdown("## 📁 Configuración de Datos")

# Opción para subir datos o usar datos demo
data_option = st.sidebar.radio(
    "Selecciona origen de datos:",
    ["📊 Usar Datos Demo", "📤 Subir Datos Propios"]
)

if data_option == "📤 Subir Datos Propios":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo CSV con datos",
        type=['csv'],
        help="Asegúrate que tu CSV contenga las columnas necesarias"
    )
    
    if uploaded_file:
        try:
            df_rezagos = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Datos cargados: {df_rezagos.shape[0]} filas, {df_rezagos.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"Error al cargar datos: {e}")
            df_rezagos = None
    else:
        df_rezagos = None
        st.sidebar.warning("⚠️ Por favor sube un archivo CSV")
else:
    # Datos demo mejorados
    np.random.seed(42)
    n_periods = 156  # 3 años de datos semanales
    
    df_rezagos = pd.DataFrame({
        'fecha': pd.date_range('2021-01-01', periods=n_periods, freq='W'),
        'ventas': np.random.normal(450000000, 80000000, n_periods) + 
                 np.sin(np.arange(n_periods) * 2 * np.pi / 52) * 50000000,  # Estacionalidad
        'inversion_total_lag2': np.random.normal(18000000, 5000000, n_periods),
        'InvCompetidoresTot_lag3': np.random.normal(160000, 50000, n_periods),
        'megaprima': np.random.binomial(1, 0.08, n_periods),
        'navidad': np.random.binomial(1, 0.04, n_periods),
        'xmas2': np.random.binomial(1, 0.04, n_periods),
        'xmas3': np.random.binomial(1, 0.04, n_periods),
        'ddpe2': np.random.binomial(1, 0.06, n_periods),
        'temp_escolar': np.random.binomial(1, 0.25, n_periods),
        'aniversario': np.random.binomial(1, 0.04, n_periods),
        'halloween': np.random.binomial(1, 0.04, n_periods),
        'dsi': np.random.binomial(1, 0.06, n_periods),
        'quincena': np.random.binomial(1, 0.43, n_periods),
        'TPM_Diario': np.random.normal(2500, 500, n_periods)
    })
    st.sidebar.success("✅ Usando datos demo")

# Verificar si tenemos datos
if df_rezagos is not None:
    
    # Definir variables del modelo
    df_model = df_rezagos[[
        'inversion_total_lag2',
        'InvCompetidoresTot_lag3',
        'megaprima', 'navidad','xmas2', 'xmas3', 'ddpe2', 
        'temp_escolar', 'aniversario','halloween',
        'dsi', 'quincena', "TPM_Diario"
    ]]
    x_cols = df_model.columns.tolist()
    
    # Ajustar modelo
    with st.spinner('🔄 Ajustando modelo OLS...'):
        modelo = ajustar_ols(df_rezagos, 'ventas', x_cols)
        resultados_contrib = contribuciones(df_model, modelo)
    
    # Sidebar con info del modelo
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Estadísticas del Modelo")
    st.sidebar.metric("R²", f"{modelo.rsquared:.3f}")
    st.sidebar.metric("R² Ajustado", f"{modelo.rsquared_adj:.3f}")
    st.sidebar.metric("F-Statistic", f"{modelo.fvalue:.2f}")
    st.sidebar.metric("Observaciones", int(modelo.nobs))
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen Ejecutivo", 
        "🎯 Contribuciones", 
        "📈 Performance", 
        "💰 ROI Analysis",
        "🔍 Modelo Detallado"
    ])
    
    with tab1:
        st.markdown("## 🎯 Resumen Ejecutivo")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>💰 Ventas Totales</h3>
                <h2>{}</h2>
            </div>
            """.format(format_number(df_rezagos['ventas'].sum())), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 R² Modelo</h3>
                <h2>{:.1%}</h2>
            </div>
            """.format(modelo.rsquared), unsafe_allow_html=True)
        
        with col3:
            roi_total = df_rezagos['ventas'].sum() / df_rezagos['inversion_total_lag2'].sum()
            st.markdown("""
            <div class="metric-card">
                <h3>🚀 ROI Total</h3>
                <h2>{:.1f}x</h2>
            </div>
            """.format(roi_total), unsafe_allow_html=True)
        
        with col4:
            top_driver = resultados_contrib.index[1]  # Excluir const
            top_contrib = resultados_contrib.loc[top_driver, '%']
            st.markdown("""
            <div class="metric-card">
                <h3>🥇 Top Driver</h3>
                <h2>{:.1f}%</h2>
                <p>{}</p>
            </div>
            """.format(top_contrib, top_driver), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráfico principal: Ventas vs Inversión
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_main = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("📈 Evolución de Ventas", "💸 Inversión en Medios"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Ventas
            fig_main.add_trace(
                go.Scatter(
                    x=df_rezagos['fecha'] if 'fecha' in df_rezagos.columns else range(len(df_rezagos)),
                    y=df_rezagos['ventas'],
                    mode='lines',
                    name='Ventas',
                    line=dict(color='#667eea', width=3),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            # Inversión
            fig_main.add_trace(
                go.Scatter(
                    x=df_rezagos['fecha'] if 'fecha' in df_rezagos.columns else range(len(df_rezagos)),
                    y=df_rezagos['inversion_total_lag2'],
                    mode='lines+markers',
                    name='Inversión',
                    line=dict(color='#764ba2', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            fig_main.update_layout(
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_main, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Insights Clave")
            
            # Correlación
            correlation = df_rezagos['ventas'].corr(df_rezagos['inversion_total_lag2'])
            if correlation > 0.7:
                corr_text = "🟢 Fuerte"
            elif correlation > 0.4:
                corr_text = "🟡 Moderada"
            else:
                corr_text = "🔴 Débil"
            
            st.markdown(f"""
            <div class="success-box">
            <strong>Correlación Ventas-Inversión:</strong><br>
            {corr_text} ({correlation:.3f})
            </div>
            """, unsafe_allow_html=True)
            
            # Volatilidad
            volatilidad = df_rezagos['ventas'].std() / df_rezagos['ventas'].mean()
            st.markdown(f"""
            <div class="highlight-box">
            <strong>Volatilidad de Ventas:</strong><br>
            {volatilidad:.1%}
            </div>
            """, unsafe_allow_html=True)
            
            # Mejor período
            if 'fecha' in df_rezagos.columns:
                best_period = df_rezagos.loc[df_rezagos['ventas'].idxmax(), 'fecha']
                st.markdown(f"""
                **🏆 Mejor Período:**<br>
                {best_period.strftime('%B %Y') if hasattr(best_period, 'strftime') else 'Período ' + str(df_rezagos['ventas'].idxmax())}
                """)
    
    with tab2:
        st.markdown("## 🎯 Análisis de Contribuciones")
        
        # Métricas de contribuciones
        col1, col2, col3 = st.columns(3)
        
        top_3_contrib = resultados_contrib.head(4)[1:4]  # Excluir const, tomar top 3
        
        for i, (col, (idx, row)) in enumerate(zip([col1, col2, col3], top_3_contrib.iterrows())):
            with col:
                color = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"""
                ### {color} {idx}
                **Contribución:** {format_number(row['Total'])}  
                **Porcentaje:** {format_percentage(row['%'])}
                """)
        
        st.markdown("---")
        
        # Visualizaciones de contribuciones
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de barras horizontal
            contrib_viz = resultados_contrib[resultados_contrib.index != 'const'].head(10)
            
            fig_contrib = px.bar(
                x=contrib_viz['%'],
                y=contrib_viz.index,
                orientation='h',
                title="🎯 Top 10 Contribuciones (%)",
                color=contrib_viz['%'],
                color_continuous_scale='viridis',
                text=contrib_viz['%'].round(2)
            )
            
            fig_contrib.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_contrib.update_layout(
                height=500,
                yaxis={'categoryorder':'total ascending'},
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_contrib, use_container_width=True)
        
        with col2:
            # Gráfico de dona
            contrib_pie = resultados_contrib[resultados_contrib.index != 'const'].head(8)
            
            fig_pie = px.pie(
                values=contrib_pie['%'].abs(),
                names=contrib_pie.index,
                title="🍰 Distribución de Contribuciones",
                hole=0.4
            )
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500)
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabla detallada
        st.markdown("### 📋 Tabla Detallada de Contribuciones")
        
        contrib_display = resultados_contrib.copy()
        contrib_display['Total_Formatted'] = contrib_display['Total'].apply(format_number)
        contrib_display['Porcentaje_Formatted'] = contrib_display['%'].apply(format_percentage)
        
        st.dataframe(
            contrib_display[['Total_Formatted', 'Porcentaje_Formatted']],
            column_config={
                'Total_Formatted': st.column_config.TextColumn('💰 Contribución Total'),
                'Porcentaje_Formatted': st.column_config.TextColumn('📊 Porcentaje')
            },
            use_container_width=True
        )
    
    with tab3:
        st.markdown("## 📈 Análisis de Performance")
        
        # Métricas de performance
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ventas_media = df_rezagos['ventas'].mean()
            st.metric("📊 Ventas Promedio", format_number(ventas_media))
        
        with col2:
            inversion_media = df_rezagos['inversion_total_lag2'].mean()
            st.metric("💸 Inversión Promedio", format_number(inversion_media))
        
        with col3:
            eficiencia = ventas_media / inversion_media
            st.metric("⚡ Eficiencia Media", f"{eficiencia:.2f}x")
        
        with col4:
            variabilidad = df_rezagos['ventas'].std()
            st.metric("📊 Desv. Estándar", format_number(variabilidad))
        
        # Gráficos de performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de ventas
            fig_dist = px.histogram(
                df_rezagos,
                x='ventas',
                nbins=25,
                title="📊 Distribución de Ventas",
                marginal="box"
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Scatter plot ventas vs inversión
            fig_scatter = px.scatter(
                df_rezagos,
                x='inversion_total_lag2',
                y='ventas',
                title="💰 Ventas vs Inversión",
                trendline="ols",
                trendline_color_override="red"
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance por período (si hay fechas)
        if 'fecha' in df_rezagos.columns:
            st.markdown("### 📅 Performance Temporal")
            
            df_temporal = df_rezagos.copy()
            df_temporal['mes'] = df_temporal['fecha'].dt.to_period('M').astype(str)
            df_monthly = df_temporal.groupby('mes').agg({
                'ventas': 'sum',
                'inversion_total_lag2': 'sum'
            }).reset_index()
            df_monthly['roi'] = df_monthly['ventas'] / df_monthly['inversion_total_lag2']
            
            fig_temporal = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Ventas e Inversión Mensual", "ROI Mensual"),
                vertical_spacing=0.1
            )
            
            fig_temporal.add_trace(
                go.Scatter(x=df_monthly['mes'], y=df_monthly['ventas'], 
                          name='Ventas', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig_temporal.add_trace(
                go.Scatter(x=df_monthly['mes'], y=df_monthly['inversion_total_lag2'], 
                          name='Inversión', line=dict(color='green'), yaxis='y2'),
                row=1, col=1
            )
            
            fig_temporal.add_trace(
                go.Scatter(x=df_monthly['mes'], y=df_monthly['roi'], 
                          name='ROI', line=dict(color='red', width=3)),
                row=2, col=1
            )
            
            fig_temporal.update_layout(height=500)
            fig_temporal.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_temporal, use_container_width=True)
    
    with tab4:
        st.markdown("## 💰 Análisis de ROI")
        
        # Calcular ROI por variable
        media_variables = df_model.mean()
        roi_por_variable = {}
        
        for var in x_cols:
            if var in modelo.params.index:
                coef = modelo.params[var]
                media_var = media_variables[var]
                if media_var != 0:
                    roi_por_variable[var] = coef / media_var if media_var > 0 else 0
        
        roi_df = pd.DataFrame(list(roi_por_variable.items()), columns=['Variable', 'ROI'])
        roi_df = roi_df.sort_values('ROI', ascending=False)
        
        # Métricas de ROI
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            roi_total = df_rezagos['ventas'].sum() / df_rezagos['inversion_total_lag2'].sum()
            st.metric("🎯 ROI Total", f"{roi_total:.2f}x")
        
        with col2:
            if len(roi_df) > 0:
                best_roi_var = roi_df.iloc[0]['Variable']
                best_roi_val = roi_df.iloc[0]['ROI']
                st.metric("🏆 Mejor ROI", f"{best_roi_val:.2f}x", best_roi_var)
        
        with col3:
            avg_roi = roi_df['ROI'].mean() if len(roi_df) > 0 else 0
            st.metric("📊 ROI Promedio", f"{avg_roi:.2f}x")
        
        with col4:
            roi_volatility = roi_df['ROI'].std() if len(roi_df) > 0 else 0
            st.metric("📈 Volatilidad ROI", f"{roi_volatility:.2f}")
        
        # Visualizaciones de ROI
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de ROI por variable
            if len(roi_df) > 0:
                fig_roi = px.bar(
                    roi_df.head(10),
                    x='ROI',
                    y='Variable',
                    orientation='h',
                    title="💰 ROI por Variable",
                    color='ROI',
                    color_continuous_scale='RdYlGn'
                )
                fig_roi.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Simulación de curva de ROI
            investment_range = np.linspace(
                df_rezagos['inversion_total_lag2'].min(),
                df_rezagos['inversion_total_lag2'].max() * 1.5,
                50
            )
            
            # ROI decreciente simulado
            base_roi = roi_total
            roi_curve = base_roi * np.exp(-investment_range / df_rezagos['inversion_total_lag2'].mean() * 0.5)
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=investment_range,
                y=roi_curve,
                mode='lines',
                name='ROI Curve',
                line=dict(color='purple', width=3)
            ))
            
            # Punto actual
            current_inv = df_rezagos['inversion_total_lag2'].mean()
            fig_curve.add_trace(go.Scatter(
                x=[current_inv],
                y=[roi_total],
                mode='markers',
                name='ROI Actual',
                marker=dict(size=15, color='red', symbol='diamond')
            ))
            
            fig_curve.update_layout(
                title="📈 Curva de ROI vs Inversión",
                xaxis_title="Inversión ($)",
                yaxis_title="ROI (x)",
                height=400
            )
            
            st.plotly_chart(fig_curve, use_container_width=True)
        
        # Recomendaciones de optimización
        st.markdown("### 🎯 Recomendaciones de Optimización")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Variables de Alto ROI</h4>
            """, unsafe_allow_html=True)
            
            if len(roi_df) > 0:
                top_roi = roi_df.head(3)
                for _, row in top_roi.iterrows():
                    st.write(f"• **{row['Variable']}**: {row['ROI']:.2f}x")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
            <h4>⚠️ Variables de Bajo ROI</h4>
            """, unsafe_allow_html=True)
            
            if len(roi_df) > 0:
                low_roi = roi_df.tail(3)
                for _, row in low_roi.iterrows():
                    st.write(f"• **{row['Variable']}**: {row['ROI']:.2f}x")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab5:
        st.markdown("## 🔍 Modelo Detallado")
        
        # Estadísticas del modelo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Bondad de Ajuste")
            st.metric("R²", f"{modelo.rsquared:.4f}")
            st.metric("R² Ajustado", f"{modelo.rsquared_adj:.4f}")
            st.metric("AIC", f"{modelo.aic:.2f}")
            st.metric("BIC", f"{modelo.bic:.2f}")
        
        with col2:
            st.markdown("### 🎯 Significancia")
            st.metric("F-Statistic", f"{modelo.fvalue:.2f}")
            st.metric("Prob (F-stat)", f"{modelo.f_pvalue:.2e}")
            st.metric("Log-Likelihood", f"{modelo.llf:.2f}")
        
        with col3:
            st.markdown("### 📈 Datos")
            st.metric("Observaciones", int(modelo.nobs))
            st.metric("Variables", len(x_cols))
            st.metric("Grados Libertad", int(modelo.df_resid))
        
        # Tabla de coeficientes
        st.markdown("### 📋 Coeficientes del Modelo")
        
        coef_df = pd.DataFrame({
            'Coeficiente': modelo.params,
            'Std Error': modelo.bse,
            't-value': modelo.tvalues,
            'P-value': modelo.pvalues,
            'Conf Int Low': modelo.conf_int()[0],
            'Conf Int High': modelo.conf_int()[1]
        })
        
        # Formatear tabla
        coef_display = coef_df.copy()
        coef_display['Significancia'] = coef_display['P-value'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
        )
        
        st.dataframe(
            coef_display.round(6),
            column_config={
                'Coeficiente': st.column_config.NumberColumn('Coeficiente', format="%.6f"),
                'Std Error': st.column_config.NumberColumn('Error Std', format="%.6f"),
                't-value': st.column_config.NumberColumn('t-value', format="%.3f"),
                'P-value': st.column_config.NumberColumn('P-value', format="%.6f"),
                'Significancia': st.column_config.TextColumn('Sig.')
            },
            use_container_width=True
        )
        
        st.caption("Significancia: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Diagnósticos del modelo
        st.markdown("### 🔬 Diagnósticos del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuos vs fitted
            fitted_values = modelo.fittedvalues
            residuals = modelo.resid
            
            fig_resid = px.scatter(
                x=fitted_values,
                y=residuals,
                title="🔍 Residuos vs Valores Ajustados",
                labels={'x': 'Valores Ajustados', 'y': 'Residuos'}
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            fig_resid.update_layout(height=400)
            st.plotly_chart(fig_resid, use_container_width=True)
        
        with col2:
            # Q-Q plot aproximado
            from scipy import stats
            residuals_normalized = (residuals - residuals.mean()) / residuals.std()
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_normalized)))
            sample_quantiles = np.sort(residuals_normalized)
            
            fig_qq = px.scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                title="📊 Q-Q Plot (Normalidad)",
                labels={'x': 'Cuantiles Teóricos', 'y': 'Cuantiles Muestrales'}
            )
            
            # Línea de referencia
            min_q, max_q = min(theoretical_quantiles), max(theoretical_quantiles)
            fig_qq.add_trace(go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Línea de Referencia'
            ))
            
            fig_qq.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Resumen estadístico detallado
        st.markdown("### 📈 Resumen Estadístico Completo")

        st.components.v1.html(summary_html, height=600, scrolling=True)

else:
    # Estado cuando no hay datos
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>📁 No hay datos cargados</h2>
        <p>Por favor selecciona "Usar Datos Demo" en el sidebar o sube tu propio archivo CSV.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h4>🚀 Marketing Mix Model Dashboard</h4>
    <p>Análisis avanzado de performance y ROI • Optimización de inversión publicitaria</p>
    <p><i>Desarrollado con ❤️ usando Streamlit</i></p>
</div>
""", unsafe_allow_html=True)

# JavaScript para mejorar la experiencia
st.markdown("""
<script>
// Auto-refresh cada 5 minutos para datos demo
setTimeout(function(){
    if (window.location.search.includes('demo')) {
        window.location.reload();
    }
}, 300000);
</script>
""", unsafe_allow_html=True)
