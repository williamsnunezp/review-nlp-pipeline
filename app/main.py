import os

# Silenciar avisos informativos y optimizaciones oneDNN de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Añadir la raíz al path de Python para importar src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import SentimentPredictor

# --- Configuración de la página ---
st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos CSS personalizados ---
st.markdown(
    """
    <style>
    /* Estilos globales y contenedor */
    .main {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    .title-container {
        text-align: center;
        padding: 1.5rem 0rem;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .title-container h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .title-container p {
        font-size: 1.1rem;
        color: #9ca3af;
    }
    /* Tarjetas de resultados */
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Inicialización del Predictor ---
@st.cache_resource
def get_predictor():
    predictor = SentimentPredictor()
    predictor.load_model("best_model")
    return predictor


predictor = get_predictor()

if "single_input" not in st.session_state:
    st.session_state["single_input"] = ""


# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.subheader("⚙️ Estado del Pipeline")
    model_type = predictor.model_type or "Keras Deep Learning / NLP"
    st.success(f"🟢 **Modelo Activo:** `{model_type}`")
    st.caption("Pipeline: Limpieza de texto ➔ Vectorización ➔ Inferencia")

    st.markdown("---")
    st.caption("Review NLP Pipeline • Streamlit v1.x")


# --- Encabezado Principal ---
st.markdown(
    """
    <div class="title-container">
        <h1>NLP Sentiment Analyzer</h1>
        <p>Clasificación inteligente de sentimientos para reseñas y opiniones</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- Pestañas de Navegación ---
tab1, tab2 = st.tabs(["💬 Inferencia Individual", "📁 Análisis por Lotes (Batch)"])


# ==============================================================================
# TAB 1: INFERENCIA INDIVIDUAL
# ==============================================================================
with tab1:
    st.markdown("#### Prueba una reseña individual")

    # Ejemplos rápidos en 1 clic
    st.caption("💡 O selecciona un ejemplo rápido para probar:")
    examples = {
        "⭐ Positivo Claro": "The product arrived in perfect condition, super fast shipping and exceeded all expectations!",
        "⚠️ Negativo Claro": "Terrible customer service, the package was delayed and arrived completely broken.",
    }

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⭐ Positivo Claro", use_container_width=True):
            st.session_state["single_input"] = examples["⭐ Positivo Claro"]
    with col_btn2:
        if st.button("⚠️ Negativo Claro", use_container_width=True):
            st.session_state["single_input"] = examples["⚠️ Negativo Claro"]

    # Área de texto
    user_input = st.text_area(
        "Texto de la reseña en inglés:",
        value=st.session_state["single_input"],
        height=140,
        placeholder="Type or paste customer review in English here...",
    )

    if st.button("🔮 Analizar Sentimiento", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Por favor ingresa algún texto antes de analizar.")
        else:
            with st.spinner("Procesando entrada con el modelo..."):
                res = predictor.predict(user_input)

                if "error" in res:
                    st.error(res["error"])
                else:
                    sentiment = res.get("sentiment", "N/A")
                    confidence = float(res.get("confidence", 0.0))
                    cleaned = res.get("cleaned_text", "")
                    tokens_count = len(cleaned.split())

                    st.markdown("---")
                    st.subheader("📊 Resultado del Análisis")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        icon = "😊" if sentiment == "Positivo" else "😞"
                        st.metric(
                            label="Sentimiento Predicho",
                            value=f"{icon} {sentiment}",
                            delta="Favorable"
                            if sentiment == "Positivo"
                            else "-Desfavorable",
                            delta_color="normal"
                            if sentiment == "Positivo"
                            else "inverse",
                        )
                    with col2:
                        st.metric(
                            label="Nivel de Confianza",
                            value=f"{confidence * 100:.1f}%",
                        )
                    with col3:
                        st.metric(
                            label="Tokens Procesados",
                            value=f"{tokens_count} palabras",
                        )

                    # Barra visual de confianza
                    st.progress(
                        min(confidence, 1.0),
                        text=f"Certeza del modelo: {confidence*100:.1f}%",
                    )

                    with st.expander("🔍 Inspeccionar texto preprocesado"):
                        st.markdown(f"**Texto normalizado (input al modelo):** `{cleaned}`")

                    # Componente de feedback de usuario
                    st.caption("¿Fue acertada la predicción?")
                    st.feedback("thumbs", key="single_prediction_feedback")


# ==============================================================================
# TAB 2: ANÁLISIS POR LOTES (CSV, EXCEL, TXT)
# ==============================================================================
with tab2:
    st.markdown("#### Carga masiva de opiniones")
    st.write(
        "Sube un archivo para procesar múltiples reseñas simultáneamente y generar métricas globales."
    )

    uploaded_file = st.file_uploader(
        "Formatos aceptados: CSV, Excel (.xlsx, .xls) o texto plano (.txt)",
        type=["csv", "xlsx", "xls", "txt"],
    )

    if uploaded_file is not None:
        try:
            df = None
            filename = uploaded_file.name.lower()

            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif filename.endswith(".txt"):
                raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                if lines:
                    df = pd.DataFrame({"review": lines})
                else:
                    st.error("El archivo .txt está vacío.")

            if df is not None and not df.empty:
                st.success(f"Archivo cargado exitosamente: **{len(df)} registros**")

                # Selector de columna
                text_columns = df.columns.tolist()
                default_col_idx = 0
                for i, col_name in enumerate(text_columns):
                    if any(term in str(col_name).lower() for term in ["review", "text", "texto", "comentario", "opinion"]):
                        default_col_idx = i
                        break

                selected_col = st.selectbox(
                    "Selecciona la columna que contiene el texto de la reseña:",
                    text_columns,
                    index=default_col_idx,
                )

                st.dataframe(df[[selected_col]].head(5), use_container_width=True)

                if st.button("🚀 Iniciar Análisis Masivo", type="primary", use_container_width=True):
                    progress_bar = st.progress(0, text="Iniciando inferencia por lotes...")
                    sentiments = []
                    confidences = []
                    cleaned_texts = []
                    total = len(df)

                    for i, text in enumerate(df[selected_col]):
                        res = predictor.predict(str(text))
                        sentiments.append(res.get("sentiment", "Error"))
                        confidences.append(float(res.get("confidence", 0.0)))
                        cleaned_texts.append(res.get("cleaned_text", ""))

                        # Actualizar barra cada 5 iteraciones o al final
                        if (i + 1) % 5 == 0 or (i + 1) == total:
                            pct = (i + 1) / total
                            progress_bar.progress(pct, text=f"Procesando {i + 1} de {total} reseñas ({pct*100:.0f}%)...")

                    progress_bar.empty()

                    # Guardar resultados en el DataFrame
                    result_df = df.copy()
                    result_df["pred_sentiment"] = sentiments
                    result_df["pred_confidence"] = [round(c, 4) for c in confidences]
                    result_df["cleaned_text"] = cleaned_texts

                    st.markdown("---")
                    st.subheader("📈 Resumen Ejecutivo y Distribución")

                    # Conteo y porcentajes
                    counts = result_df["pred_sentiment"].value_counts().reset_index()
                    counts.columns = ["Sentimiento", "Cantidad"]
                    counts["Porcentaje"] = (counts["Cantidad"] / total) * 100
                    counts["Etiqueta"] = counts.apply(
                        lambda r: f"{r['Cantidad']} ({r['Porcentaje']:.1f}%)", axis=1
                    )

                    pos_row = counts[counts["Sentimiento"] == "Positivo"]
                    neg_row = counts[counts["Sentimiento"] == "Negativo"]

                    pos_pct = pos_row["Porcentaje"].values[0] if not pos_row.empty else 0.0
                    neg_pct = neg_row["Porcentaje"].values[0] if not neg_row.empty else 0.0
                    pos_count = pos_row["Cantidad"].values[0] if not pos_row.empty else 0
                    neg_count = neg_row["Cantidad"].values[0] if not neg_row.empty else 0

                    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                    with col_kpi1:
                        st.metric("Total Analizado", f"{total} reseñas")
                    with col_kpi2:
                        st.metric("Positivas", f"{pos_count} ({pos_pct:.1f}%)", delta="Favorable")
                    with col_kpi3:
                        st.metric("Negativas", f"{neg_count} ({neg_pct:.1f}%)", delta="-Desfavorable", delta_color="inverse")

                    # Gráfico de Barras con Altair mostrando Cantidad y Porcentaje
                    color_scale = alt.Scale(
                        domain=["Positivo", "Negativo"],
                        range=["#10b981", "#ef4444"],
                    )

                    base_chart = (
                        alt.Chart(counts)
                        .encode(
                            x=alt.X("Sentimiento:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                            y=alt.Y("Cantidad:Q", title="Cantidad de Reseñas"),
                            color=alt.Color("Sentimiento:N", scale=color_scale, legend=None),
                            tooltip=["Sentimiento", "Cantidad", alt.Tooltip("Porcentaje:Q", format=".1f")],
                        )
                    )

                    bars = base_chart.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    labels = base_chart.mark_text(
                        align="center",
                        baseline="bottom",
                        dy=-6,
                        fontSize=14,
                        fontWeight="bold",
                        color="#f3f4f6",
                    ).encode(text="Etiqueta:N")

                    st.altair_chart(bars + labels, use_container_width=True)

                    # Tabla de detalle
                    st.subheader("📋 Vista Detallada de Predicciones")
                    st.dataframe(result_df, use_container_width=True)

                    # Botón de Descarga
                    csv_export = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Descargar Resultados en CSV",
                        data=csv_export,
                        file_name=f"analisis_sentimientos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

        except Exception as err:
            st.error(f"Error procesando el archivo: {err}")
