import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="TobiPets Dashboard",
    page_icon="🐕",
    initial_sidebar_state="expanded"
)

# Configurar estilo de matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# CSS personalizado
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E86AB;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.insight-box {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2E86AB;
    margin: 1rem 0;
    color: #212529;
}
.success-box {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    color: #212529;
}
.question-box {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #6c757d;
    margin: 1rem 0;
    color: #212529;
}
</style>
""", unsafe_allow_html=True)

# Header principal
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        # Centrar la imagen usando HTML
        st.markdown(
            '<div style="display: flex; justify-content: center;">'
            '<img src="data:image/png;base64,{}" width="200">'
            '</div>'.format(
                __import__('base64').b64encode(
                    open('tobipets.png', 'rb').read()).decode()
            ),
            unsafe_allow_html=True
        )
    except:
        # Alternativa simple si no se puede cargar la imagen
        st.markdown(
            '<div style="text-align: center;">'
            '<p style="color: #666;">Logo TobiPets</p>'
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown('<h1 class="main-header">TobiPets Analytics Dashboard</h1>',
                unsafe_allow_html=True)

# Funciones auxiliares


@st.cache_data
def load_data():
    """Carga y preprocesa los datos"""
    try:
        df = pd.read_excel('Assessment.xlsx', sheet_name='Data Ordenes')
        # Renombrar columnas
        column_mapping = {
            'Fecha': 'fecha',
            'Tipo de cliente': 'tipo_de_cliente',
            'Ingresos netos': 'ingresos_netos',
            'Estado': 'estado',
            'Artículos vendidos': 'articulos_vendidos',
            'Cupón(es)': 'cupones',
            'ID del Cliente': 'id_del_cliente',
            'Pedido #': 'pedido_numero'
        }
        df = df.rename(columns=column_mapping)

        # Preprocesamiento
        df['cupones'] = df['cupones'].fillna('Sin_cupon')
        df['usa_cupon'] = df['cupones'] != 'Sin_cupon'
        df['es_refund'] = df['estado'] == 'refunded'
        df['fecha'] = pd.to_datetime(df['fecha'])

        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.info("Asegúrate de tener el archivo 'Assessment.xlsx' en el directorio")
        return None


def create_kpi_cards(df_sales):
    """KPIs principales según las preguntas específicas"""
    total_ordenes = df_sales['pedido_numero'].nunique()
    valor_neto_total = df_sales['ingresos_netos'].sum()
    ticket_promedio = df_sales.groupby('id_del_cliente')[
        'ingresos_netos'].sum().mean()
    pct_por_tipo = df_sales['tipo_de_cliente'].value_counts(normalize=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Órdenes", f"{total_ordenes:,}")

    with col2:
        st.metric("Valor Neto Total", f"₡{valor_neto_total:,.0f}")

    with col3:
        st.metric("Ticket Promedio por Cliente", f"₡{ticket_promedio:,.0f}")

    with col4:
        st.write("**% Órdenes por Tipo:**")
        for tipo, pct in pct_por_tipo.items():
            st.write(f"• {tipo}: {pct:.1%}")


def analyze_inconsistencies(df):
    """Análisis de inconsistencias en los datos"""
    st.markdown("### Verificación de Inconsistencias")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pedidos con 0 artículos:**")
        zero_articles = (df['articulos_vendidos'] == 0).sum()
        st.metric("Pedidos con 0 artículos", zero_articles)

        if zero_articles > 0:
            zero_df = df[df['articulos_vendidos'] == 0]
            st.write("Detalles:")
            st.dataframe(
                zero_df[['pedido_numero', 'estado', 'tipo_de_cliente', 'ingresos_netos']])

    with col2:
        st.write("**Valores negativos/cero:**")
        negative_income = (df['ingresos_netos'] < 0).sum()
        zero_income = (df['ingresos_netos'] == 0).sum()
        st.metric("Ingresos negativos", negative_income)
        st.metric("Ingresos = 0", zero_income)

        st.write("**Valores faltantes:**")
        nulls = df.isnull().sum()
        st.write(nulls[nulls > 0] if nulls.sum() >
                 0 else "No hay valores faltantes")


def create_evolution_plot(df_sales):
    """Evolución semanal del valor neto"""
    df_sales['semana'] = df_sales['fecha'].dt.to_period(
        'W').apply(lambda r: r.start_time)
    weekly = df_sales.groupby('semana')['ingresos_netos'].sum().reset_index()
    weekly['ingresos_k'] = weekly['ingresos_netos'] / 1000

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weekly['semana'], weekly['ingresos_k'], marker='o',
            linewidth=3, markersize=8, color='#2E86AB')
    ax.set_title('Evolución Semanal del Valor Neto',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor Neto (miles $)', fontsize=12)
    ax.set_xlabel('Semana', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Añadir etiquetas de valor
    for x, y in zip(weekly['semana'], weekly['ingresos_k']):
        ax.text(x, y + 0.5, f'{y:.1f}K', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_orders_by_status_plot(df):
    """Cantidad de órdenes por estado"""
    estado_counts = df['estado'].value_counts().reset_index()
    estado_counts.columns = ['estado', 'n_ordenes']
    estado_counts['pct'] = 100 * estado_counts['n_ordenes'] / \
        estado_counts['n_ordenes'].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(estado_counts['estado'], estado_counts['n_ordenes'],
                  color=['#2E86AB', '#A23B72', '#F18F01'])

    # Añadir etiquetas de porcentaje
    for i, (bar, row) in enumerate(zip(bars, estado_counts.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{row.pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_title('Distribución de Órdenes por Estado', fontweight='bold')
    ax.set_xlabel('Estado')
    ax.set_ylabel('Número de Órdenes')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_customer_value(df_sales):
    """Análisis de valor por tipo de cliente"""
    # Métricas por tipo de cliente
    customer_metrics = df_sales.groupby('tipo_de_cliente')['ingresos_netos'].agg([
        'count', 'mean', 'median', 'std', 'sum'
    ]).round(2)
    customer_metrics.columns = ['Órdenes',
                                'Promedio', 'Mediana', 'Std', 'Total']

    # Test estadístico
    new_values = df_sales[df_sales['tipo_de_cliente']
                          == 'new']['ingresos_netos']
    returning_values = df_sales[df_sales['tipo_de_cliente']
                                == 'returning']['ingresos_netos']
    t_stat, p_value = stats.ttest_ind(
        new_values, returning_values, equal_var=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Métricas por Tipo de Cliente:**")
        st.dataframe(customer_metrics)

        if p_value < 0.05:
            st.success(
                f"✅ Diferencia estadísticamente significativa (p={p_value:.4f})")
        else:
            st.warning(f"⚠️ Diferencia NO significativa (p={p_value:.4f})")

    with col2:
        # Histograma de distribución por tipo de cliente
        fig, ax = plt.subplots(figsize=(8, 5))

        # Separar datos por tipo de cliente
        new_data = df_sales[df_sales['tipo_de_cliente']
                            == 'new']['ingresos_netos']
        returning_data = df_sales[df_sales['tipo_de_cliente']
                                  == 'returning']['ingresos_netos']

        # Crear histograma superpuesto
        bins = np.linspace(df_sales['ingresos_netos'].min(
        ), df_sales['ingresos_netos'].max(), 25)

        ax.hist(new_data, bins=bins, alpha=0.7, label='new',
                color='lightblue', edgecolor='black')
        ax.hist(returning_data, bins=bins, alpha=0.7,
                label='returning', color='darkblue', edgecolor='black')

        ax.set_xlabel('Ingresos Netos')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Ingresos por Tipo de Cliente')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Formatear eje x para mostrar valores en miles
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'₡{x/1000:.0f}K'))

        plt.tight_layout()
        st.pyplot(fig)

    return customer_metrics


def analyze_coupon_impact(df_sales):
    """Análisis del impacto de cupones"""
    # Valor promedio por uso de cupón
    coupon_analysis = df_sales.groupby('usa_cupon').agg({
        'ingresos_netos': ['mean', 'count'],
        'articulos_vendidos': 'mean'
    }).round(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Valor promedio
    avg_by_coupon = df_sales.groupby('usa_cupon')['ingresos_netos'].mean()
    labels = ['Sin Cupón', 'Con Cupón']
    bars1 = ax1.bar(labels, avg_by_coupon.values, color=['#FFB3BA', '#BAFFC9'])
    ax1.set_title('Valor Neto Promedio por Uso de Cupón')
    ax1.set_ylabel('Valor Promedio ($)')

    for bar, value in zip(bars1, avg_by_coupon.values):
        ax1.text(bar.get_x() + bar.get_width()/2., value + 500,
                 f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

    # Gráfico 2: Artículos promedio
    avg_articles = df_sales.groupby('usa_cupon')['articulos_vendidos'].mean()
    bars2 = ax2.bar(labels, avg_articles.values, color=['#FFD93D', '#6BCF7F'])
    ax2.set_title('Artículos Promedio por Uso de Cupón')
    ax2.set_ylabel('Artículos Promedio')

    for bar, value in zip(bars2, avg_articles.values):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.1,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig, coupon_analysis


def analyze_articles_by_status(df):
    """Análisis de artículos promedio por estado"""
    articles_by_status = df.groupby(
        'estado')['articulos_vendidos'].mean().reset_index()
    articles_by_status.columns = ['estado', 'promedio_articulos']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(articles_by_status['estado'], articles_by_status['promedio_articulos'],
                  color=['#2E86AB', '#A23B72', '#FF6B6B'])

    ax.set_title('Promedio de Artículos por Estado de Orden')
    ax.set_ylabel('Artículos Promedio')

    for bar, row in zip(bars, articles_by_status.itertuples()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{row.promedio_articulos:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig, articles_by_status


def analyze_returning_customers(df_sales):
    """Análisis específico de clientes returning"""
    df_return = df_sales[df_sales['tipo_de_cliente'] == 'returning']

    metrics = {
        "% Órdenes returning": df_return['pedido_numero'].nunique() / df_sales['pedido_numero'].nunique() * 100,
        "% Ingresos returning": df_return['ingresos_netos'].sum() / df_sales['ingresos_netos'].sum() * 100,
        "% Uso de cupón (returning)": df_return['usa_cupon'].mean() * 100,
        "Ticket promedio (returning)": df_return['ingresos_netos'].mean(),
        "Artículos promedio (returning)": df_return['articulos_vendidos'].mean()
    }

    return metrics


# Cargar datos
df = load_data()
if df is None:
    st.stop()

# Separar datos
df_sales = df[~df['es_refund']].copy()
df_refunds = df[df['es_refund']].copy()

# Tabs principales
tab1, tab2, tab3 = st.tabs([
    "Sección 1: Análisis Exploratorio",
    "Sección 2: Insights de Comportamiento",
    "Sección 3: Casos de Negocio"
])

# TAB 1: Análisis Exploratorio
with tab1:
    st.markdown('<h2 class="section-header">Sección 1: Análisis Exploratorio</h2>',
                unsafe_allow_html=True)

    # Información del dataset
    st.markdown("### Información General del Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Duplicados", df.duplicated().sum())

    # Verificación de inconsistencias
    analyze_inconsistencies(df)

    # KPIs principales
    st.markdown('<h3 class="section-header">KPIs Principales</h3>',
                unsafe_allow_html=True)
    create_kpi_cards(df_sales)

    # Visualizaciones principales
    st.markdown('<h3 class="section-header">Visualizaciones</h3>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Evolución Semanal del Valor Neto**")
        fig_evolution = create_evolution_plot(df_sales)
        st.pyplot(fig_evolution)

    with col2:
        st.markdown("**Órdenes por Estado**")
        fig_status = create_orders_by_status_plot(df)
        st.pyplot(fig_status)

# TAB 2: Insights de Comportamiento
with tab2:
    st.markdown('<h2 class="section-header">Sección 2: Insights de Comportamiento</h2>',
                unsafe_allow_html=True)

    # Pregunta 1: Valor por tipo de cliente
    st.markdown("""
    <div class="question-box">
    <h4>¿Qué tipo de cliente (new vs returning) genera mayor valor neto por orden?</h4>
    </div>
    """, unsafe_allow_html=True)

    customer_metrics = analyze_customer_value(df_sales)

    returning_avg = customer_metrics.loc['returning', 'Promedio']
    new_avg = customer_metrics.loc['new', 'Promedio']

    st.markdown(f"""
    <div class="insight-box">
    <h4>Respuesta:</h4>
    <p>Los clientes <strong>returning</strong> generan mayor valor promedio por orden: <strong>₡{returning_avg:,.0f}</strong> vs <strong>₡{new_avg:,.0f}</strong> de los nuevos.</p>
    <p>Esto representa una diferencia del <strong>{((returning_avg/new_avg - 1) * 100):.1f}%</strong> más valor por orden.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta 2: Diferencias por uso de cupones
    st.markdown("""
    <div class="question-box">
    <h4>¿Hay diferencias en valor neto o cantidad de artículos entre quienes usan cupones y quienes no?</h4>
    </div>
    """, unsafe_allow_html=True)

    fig_coupon, coupon_analysis = analyze_coupon_impact(df_sales)
    st.pyplot(fig_coupon)

    with_coupon_avg = df_sales[df_sales['usa_cupon']]['ingresos_netos'].mean()
    without_coupon_avg = df_sales[~df_sales['usa_cupon']
                                  ]['ingresos_netos'].mean()

    st.markdown(f"""
    <div class="insight-box">
    <h4>Respuesta:</h4>
    <p><strong>Valor Neto:</strong> Los pedidos con cupón tienen un valor promedio de <strong>₡{with_coupon_avg:,.0f}</strong> vs <strong>₡{without_coupon_avg:,.0f}</strong> sin cupón.</p>
    <p><strong>Conclusión:</strong> {'Los cupones aumentan' if with_coupon_avg > without_coupon_avg else 'Los cupones reducen'} el valor promedio del pedido.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta 3: Artículos por estado
    st.markdown("""
    <div class="question-box">
    <h4>¿Cuál es el promedio de artículos por orden según el estado? ¿Detectás alguna anomalía?</h4>
    </div>
    """, unsafe_allow_html=True)

    fig_articles, articles_data = analyze_articles_by_status(df)
    st.pyplot(fig_articles)

    refunded_avg = articles_data[articles_data['estado']
                                 == 'refunded']['promedio_articulos'].iloc[0]
    completed_avg = articles_data[articles_data['estado']
                                  == 'completed']['promedio_articulos'].iloc[0]

    st.markdown(f"""
    <div class="insight-box">
    <h4>Respuesta:</h4>
    <p><strong>Anomalía detectada:</strong> Los pedidos refunded tienen un promedio de <strong>{refunded_avg:.2f} artículos</strong>, mucho menor que los completed (<strong>{completed_avg:.2f} artículos</strong>).</p>
    <p>Esto sugiere que muchos refunds son de pedidos que nunca se procesaron correctamente (0 artículos).</p>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta 4: Estrategia para clientes nuevos
    st.markdown("""
    <div class="question-box">
    <h4>¿Qué harías para que estos clientes nuevos vuelvan a comprar en menos de 35 días?</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h4>Estrategia de Retención para Clientes Nuevos:</h4>
    
    <p><strong>1. Welcome Series Automation (Días 1-7):</strong></p>
    <ul>
        <li>Email de bienvenida con guía de productos personalizados</li>
        <li>Contenido educativo sobre cuidado de mascotas basado en primera compra</li>
        <li>Cupón de 10% para segunda compra con urgencia (válido 7 días)</li>
        <li><em>Data Science:</em> Segmentación automática por tipo de mascota y productos comprados</li>
    </ul>
    
    <p><strong>2. Engagement Touchpoints (Días 8-21):</strong></p>
    <ul>
        <li>SMS con tips personalizados según su mascota y productos anteriores</li>
        <li>Push notifications de productos complementarios</li>
        <li>Invitación a programa de lealtad con beneficios inmediatos</li>
        <li><em>Data Science:</em> Sistema de recomendaciones basado en clientes similares</li>
    </ul>
    
    <p><strong>3. Reactivation Campaign (Días 22-35):</strong></p>
    <ul>
        <li>Email con productos recomendados basados en primera compra</li>
        <li>Oferta escalada: "Vuelve en 7 días y obtén 15% off + envío gratis"</li>
        <li>Retargeting ads en redes sociales con productos vistos</li>
        <li><em>Data Science:</em> Modelo predictivo para identificar clientes en riesgo de no volver</li>
    </ul>
    
    <p><strong>4. Monitoreo y Optimización:</strong></p>
    <ul>
        <li>A/B testing en cada fase para optimizar conversión</li>
        <li>Dashboard en tiempo real para tracking de métricas clave</li>
        <li>Análisis de cohortes para medir efectividad de cada intervención</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta 5: Análisis de clientes returning
    st.markdown("""
    <div class="question-box">
    <h4>¿Qué notas de los clientes Returning?</h4>
    </div>
    """, unsafe_allow_html=True)

    returning_metrics = analyze_returning_customers(df_sales)

    col1, col2 = st.columns(2)

    with col1:
        for metric, value in list(returning_metrics.items())[:3]:
            if "%" in metric:
                st.metric(metric, f"{value:.1f}%")
            else:
                st.metric(metric, f"₡{value:,.0f}")

    with col2:
        for metric, value in list(returning_metrics.items())[3:]:
            if "%" in metric:
                st.metric(metric, f"{value:.1f}%")
            else:
                st.metric(metric, f"{value:.2f}")

    st.markdown(f"""
    <div class="insight-box">
    <h4>Observaciones sobre Clientes Returning:</h4>
    <ul>
        <li><strong>Dominancia:</strong> Representan el {returning_metrics['% Órdenes returning']:.1f}% de las órdenes pero generan el {returning_metrics['% Ingresos returning']:.1f}% de los ingresos</li>
        <li><strong>Valor Superior:</strong> Su ticket promedio (₡{returning_metrics['Ticket promedio (returning)']:,.0f}) es {((customer_metrics.loc['returning', 'Promedio']/customer_metrics.loc['new', 'Promedio'] - 1) * 100):.1f}% mayor que nuevos</li>
        <li><strong>Uso de Cupones:</strong> {returning_metrics['% Uso de cupón (returning)']:.1f}% usa cupones vs {df_sales[df_sales['tipo_de_cliente'] == 'new']['usa_cupon'].mean()*100:.1f}% de nuevos</li>
        <li><strong>Comportamiento:</strong> Compran más artículos por orden ({returning_metrics['Artículos promedio (returning)']:.2f} vs {df_sales[df_sales['tipo_de_cliente'] == 'new']['articulos_vendidos'].mean():.2f})</li>
        <li><strong>Implicación:</strong> Son el motor del negocio - requieren estrategias de retención premium</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 3: Casos de Negocio
with tab3:
    st.markdown('<h2 class="section-header">Sección 3: Casos de Negocio</h2>',
                unsafe_allow_html=True)

    # Caso 1: Cupones y Refunds
    st.markdown('<h3 class="section-header">Caso 1: Cupones y Refunds</h3>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="question-box">
    <h4>Contexto:</h4>
    <p>El área comercial quiere entender si los cupones están ayudando a aumentar las ventas o simplemente canibalizan ingresos. También quieren reducir reembolsos.</p>
    </div>
    """, unsafe_allow_html=True)

    # Análisis de cupones
    coupon_usage_pct = df_sales['usa_cupon'].mean() * 100
    coupon_types = df_sales[df_sales['usa_cupon']]['cupones'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**% de órdenes que utilizan cupones:**")
        st.metric("Uso de Cupones", f"{coupon_usage_pct:.1f}%")

        st.markdown("**Tipo de cupón más frecuente:**")
        most_frequent = coupon_types.index[0]
        st.metric("Más usado", most_frequent)
        st.write(f"({coupon_types.iloc[0]} órdenes)")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        coupon_types.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Frecuencia por Tipo de Cupón')
        ax.set_ylabel('Número de Órdenes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Análisis de valor con/sin cupón
    st.markdown("**Valor neto promedio: con vs sin cupón**")
    with_coupon = df_sales[df_sales['usa_cupon']]['ingresos_netos'].mean()
    without_coupon = df_sales[~df_sales['usa_cupon']]['ingresos_netos'].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Con Cupón", f"₡{with_coupon:,.0f}")
    with col2:
        st.metric("Sin Cupón", f"₡{without_coupon:,.0f}")

    # Análisis de refunds
    st.markdown("**Patrones en órdenes reembolsadas:**")

    if len(df_refunds) > 0:
        refund_by_customer = df_refunds['tipo_de_cliente'].value_counts(
            normalize=True) * 100
        refund_by_coupon = df_refunds['usa_cupon'].value_counts(
            normalize=True) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Refunds", len(df_refunds))
            st.metric("% del Total", f"{len(df_refunds)/len(df)*100:.1f}%")

        with col2:
            st.write("**Por tipo de cliente:**")
            for tipo, pct in refund_by_customer.items():
                st.write(f"• {tipo}: {pct:.1f}%")

        with col3:
            st.write("**Por uso de cupón:**")
            st.write(f"• Con cupón: {refund_by_coupon.get(True, 0):.1f}%")
            st.write(f"• Sin cupón: {refund_by_coupon.get(False, 0):.1f}%")

    # Recomendaciones para el equipo comercial
    coupon_impact = "aumentan" if with_coupon > without_coupon else "reducen"
    coupon_diff = abs(with_coupon - without_coupon) / without_coupon * 100

    st.markdown(f"""
    <div class="success-box">
    <h4>2 Acciones Recomendadas para el Equipo Comercial:</h4>
    <ol>
        <li><strong>Optimizar Estrategia de Cupones:</strong>
            <ul>
                <li>Los cupones {coupon_impact} el valor promedio del pedido en {coupon_diff:.1f}%</li>
                <li>Enfocar cupon_autoship (más frecuente) en clientes returning de alto valor</li>
                <li>Implementar cupones condicionales: "Compra +$50,000 y obtén 10% off"</li>
                <li>A/B testing: cupones por porcentaje vs valor fijo</li>
            </ul>
        </li>
        <li><strong>Programa de Prevención de Refunds:</strong>
            <ul>
                <li>Implementar alertas tempranas para pedidos con 0 artículos</li>
                <li>Quality check automático antes del envío</li>
                <li>Follow-up proactivo 48h post-compra para clientes nuevos</li>
                <li>Sistema de feedback inmediato para detectar problemas</li>
            </ul>
        </li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Caso 2: AutoShip
    st.markdown('<h3 class="section-header">Caso 2: Estrategia AutoShip</h3>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="question-box">
    <h4>Contexto:</h4>
    <p>Convertir a un cliente existente (7 compras, 5 mascotas) en usuario de AutoShip.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h4>Perfil del Cliente Objetivo</h4>
    <ul>
        <li><strong>Historial:</strong> 7 compras realizadas</li>
        <li><strong>Mascotas:</strong> 5 mascotas (alta necesidad recurrente)</li>
        <li><strong>Comportamiento:</strong> Cliente leal pero manual</li>
        <li><strong>Pain Point:</strong> Reordenar constantemente los mismos productos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Argumentos de Persuasión
        
        **Comodidad Total**
        - "Olvídate de quedarte sin comida para tus 5 mascotas"
        - "Programamos tus entregas según tu cronograma histórico"
        - "Ajustes automáticos basados en tus 7 compras anteriores"
        
        **Beneficios Económicos Reales**
        - 6% descuento automático en cada envío AutoShip
        - Envío gratuito en todas las entregas
        - Si el precio baja, pagas el precio menor automáticamente
        
        **Flexibilidad y Control**
        - Modifica, pausa o cancela cuando quieras
        - Cambio de productos con un clic
        - Alertas 5 días antes de cada envío para hacer cambios
        """)

    with col2:
        st.markdown("""
        ### Proceso de Acercamiento
        
        **Semana 1: Email Personalizado**
        - Analizar sus 7 compras anteriores para identificar patrones
        - Crear plan AutoShip basado en su frecuencia histórica
        - Simulador de ahorros: mostrar ahorro proyectado anual
        
        **Semana 2: Demo Interactiva**
        - Customer Success Manager hace demo personalizada
        - Configuración en vivo basada en sus datos reales
        - Cálculo en tiempo real de frecuencia óptima por producto
        
        **Semana 3: Seguimiento Inteligente**
        - Recordatorios basados en su comportamiento de navegación
        - Ofertas ajustadas según su propensión a convertir
        - Casos de éxito de clientes con perfil similar
        
        **Optimización Continua**
        - A/B testing del proceso para maximizar conversión
        - Análisis de abandono para mejorar la experiencia
        - Tracking de métricas de adopción y satisfacción
        """)

    # Proyección ROI con datos reales
    st.markdown("### Proyección de Impacto (Basada en Datos)")

    avg_ticket = df_sales['ingresos_netos'].mean()
    monthly_frequency = 1.5
    autoship_discount = 0.06  # 6% según la información real
    retention_boost = 0.30  # 30% más retención (estimado)

    col1, col2, col3 = st.columns(3)

    with col1:
        current_ltv = avg_ticket * monthly_frequency * 12
        st.metric(
            "LTV Actual (12 meses)",
            f"₡{current_ltv:,.0f}",
            delta="Cliente típico"
        )

    with col2:
        autoship_ltv = avg_ticket * monthly_frequency * \
            (1 - autoship_discount) * (1 + retention_boost) * 12
        st.metric(
            "LTV con AutoShip",
            f"₡{autoship_ltv:,.0f}",
            delta=f"+{retention_boost*100:.0f}% retención"
        )

    with col3:
        net_gain = autoship_ltv - current_ltv
        st.metric(
            "Ganancia Neta Anual",
            f"₡{net_gain:,.0f}",
            delta="Por cliente convertido"
        )

    # Timeline de implementación
    st.markdown("### Timeline de Conversión")

    timeline_data = {
        'Día': ['Día 1', 'Día 3', 'Día 7', 'Día 14', 'Día 21'],
        'Canal': ['Email', 'Web Demo', 'Llamada', 'SMS', 'Retargeting'],
        'Mensaje Clave': [
            'Plan AutoShip basado en tus 7 compras',
            'Simulador de ahorros personalizado',
            'Demo en vivo + configuración',
            'Recordatorio: ahorros proyectados',
            'Casos de éxito + última oportunidad'
        ],
        'Métrica Objetivo': ['30% open rate', '20% click rate', '15% conversión', '10% reactivación', '5% conversión final']
    }
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)

    # Caso 3: Manejo de Reseñas Negativas
    st.markdown('<h3 class="section-header">Caso 3: Manejo de Reseñas Negativas</h3>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="question-box">
    <h4>Contexto:</h4>
    <p>Un cliente reciente tuvo una experiencia insatisfactoria, dejó reseña negativa en el sitio web y se queja en redes sociales.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h4>Estrategia de Manejo de Crisis de Reputación:</h4>
    
    <p><strong>Fase 1: Respuesta Inmediata (< 2 horas)</strong></p>
    <ul>
        <li>Respuesta pública empática agradeciendo el feedback específico</li>
        <li>Disculpa sincera sin excusas o justificaciones</li>
        <li>Invitación a contacto privado para resolución inmediata</li>
        <li>Demostración de compromiso con la mejora continua</li>
    </ul>
    
    <p><strong>Fase 2: Resolución Privada (< 24 horas)</strong></p>
    <ul>
        <li>Investigación completa del historial del pedido específico</li>
        <li>Contacto telefónico personalizado del Customer Success</li>
        <li>Oferta de solución inmediata (reemplazo express/reembolso completo)</li>
        <li>Compensación adicional apropiada (cupón 20-25%)</li>
    </ul>
    
    <p><strong>Fase 3: Seguimiento y Prevención</strong></p>
    <ul>
        <li>Implementación de mejoras en procesos internos identificados</li>
        <li>Comunicación transparente de cambios realizados</li>
        <li>Seguimiento proactivo de satisfacción del cliente</li>
        <li>Monitoreo automatizado para prevenir casos similares</li>
    </ul>
    
    <p><strong>Elementos de Data Science aplicables:</strong></p>
    <ul>
        <li><strong>Social Listening:</strong> Algoritmos para detectar menciones negativas automáticamente</li>
        <li><strong>Sentiment Analysis:</strong> Clasificación automática de severidad de quejas</li>
        <li><strong>Predictive Analytics:</strong> Identificar productos/procesos con mayor riesgo de generar quejas</li>
        <li><strong>A/B Testing:</strong> Optimizar templates de respuesta para maximizar satisfacción</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "<strong>TobiPets Analytics Dashboard</strong> | "
    f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
    "Desarrollado por Jorge"
    "</div>",
    unsafe_allow_html=True
)
