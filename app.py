import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import webbrowser 

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
st.set_page_config(layout="wide", page_title="Solvmetria Educativa")

# Título y Descripción Nuevos, solo si no ha seleccionado nivel
if st.session_state.get('user_level') is None:
    st.title("🌱 Solvmetria: Plataforma de Análisis y Calidad de Datos de Suelos")
    st.markdown("##### **Datos confiables para decisiones seguras.**")
    st.markdown("---")

# Inicializar el estado de sesión si no existe
if 'user_level' not in st.session_state:
    st.session_state.user_level = None

# 1. Cargar los datos (Usando caché para que cargue solo una vez)
@st.cache_data
def load_data():
    """Carga los datos limpios y asegura el formato de las columnas clave, manejando NaNs en filtros."""
    try:
        df = pd.read_csv("datos_limpios.csv", low_memory=False)
        
        df['Departamento'] = df['Departamento'].fillna('Desconocido').astype(str) 
        df['Municipio'] = df['Municipio'].fillna('Desconocido').astype(str)      
        
        df['pH_agua_suelo'] = pd.to_numeric(df['pH_agua_suelo'], errors='coerce')
        df['Aluminio intercambiable'] = pd.to_numeric(df['Aluminio intercambiable'], errors='coerce')
        df['Materia organica'] = pd.to_numeric(df['Materia organica'], errors='coerce') 
        
        df['Fecha de Análisis'] = pd.to_datetime(df['Fecha de Análisis'], errors='coerce', dayfirst=True)
        return df
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'datos_limpios.csv'. Asegúrate de que esté en la misma carpeta.")
        return pd.DataFrame()

df_original = load_data()

# --- 2. LÓGICA CENTRAL (Diagnóstico y ICD) ---

ICD_PARAMS = {
    'penalidad_nulo_ph': 20, 'penalidad_nulo_al': 20, 'penalidad_incoherente_ph': 30,
    'penalidad_anomalias': 15, 'penalidad_baja_precision': 10, 'penalidad_antiguedad': 20,
    'umbral_ph_min': 3.0, 'umbral_ph_max': 10.0, 'umbral_al_toxico': 1.0,
    'umbral_mo_bajo': 2.0, 'tasa_outliers_max': 0.10, 'año_corte_antiguedad': 2018
}

def detectar_outliers_iqr(series):
    """Detecta outliers usando el Rango Intercuartílico (IQR)."""
    data = series.dropna()
    if data.empty or len(data) < 4:
        return 0, pd.Series(dtype=series.dtype)

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = data[(data < limite_inferior) | (data > limite_superior)]
    
    return len(outliers), outliers

def obtener_diagnostico(df_muestras, params):
    """Calcula el diagnóstico del suelo y las advertencias, con manejo robusto de datos faltantes."""
    
    # 🛑 MANEJO ROBUSTO DE EXCEPCIONES PARA MUNICIPIOS CON POCOS DATOS 🛑
    if df_muestras.empty or \
       (df_muestras[['pH_agua_suelo', 'Aluminio intercambiable', 'Materia organica']].isnull().all().all()):
        
        return ["🛑 Advertencia: No hay datos completos o suficientes para generar un diagnóstico agronómico para este municipio. **Se requiere recolectar más muestras.**", "peligro", 0.0, 0.0, 0.0]

    # Contar solo las muestras donde al menos una de las 3 variables es NOT NULL
    df_validas_para_diag = df_muestras.dropna(subset=['pH_agua_suelo', 'Aluminio intercambiable', 'Materia organica'], how='all')
    
    if df_validas_para_diag.empty:
        return ["🛑 Advertencia: No hay datos completos o suficientes para generar un diagnóstico agronómico para este municipio. **Se requiere recolectar más muestras.**", "peligro", 0.0, 0.0, 0.0]


    # Continúa con el cálculo normal si hay datos
    promedio_ph = df_validas_para_diag['pH_agua_suelo'].mean()
    promedio_aluminio = df_validas_para_diag['Aluminio intercambiable'].mean()
    promedio_mo = df_validas_para_diag['Materia organica'].mean()

    advertencias = []
    estado_general = "saludable"
    
    for promedio, nombre, umbral_min, umbral_max, es_peligro_alto, msj_bajo, msj_peligro in [
        (promedio_ph, 'pH', 5.5, 7.5, False, "Ácido", "Fuera de Rango Extremo"),
        (promedio_aluminio, 'Aluminio', np.nan, params['umbral_al_toxico'], True, None, "Tóxico"),
        (promedio_mo, 'Materia Orgánica', params['umbral_mo_bajo'], np.nan, False, "Baja", None)
    ]:
        if pd.notna(promedio):
            if es_peligro_alto and promedio > umbral_max:
                advertencias.append(f"{nombre}: **{msj_peligro}** ({promedio:.2f}). 🛑")
                estado_general = "peligro"
            elif nombre == 'pH':
                if promedio < params['umbral_ph_min'] or promedio > params['umbral_ph_max']:
                    advertencias.append(f"pH: **{msj_peligro}** ({promedio:.2f}). 🛑")
                    estado_general = "peligro"
                elif promedio < umbral_min:
                    advertencias.append(f"pH: **{msj_bajo}** ({promedio:.2f}). Requiere enmiendas como cal. ⚠️")
                    if estado_general != "peligro": estado_general = "alerta"
            elif nombre == 'Materia Orgánica' and promedio < umbral_min:
                advertencias.append(f"Materia Orgánica: **{msj_bajo}** ({promedio:.2f}). Sugerimos mejorar la fertilidad. ⚠️")
                if estado_general == "saludable": estado_general = "alerta"
        else:
            advertencias.append(f"{nombre}: **Dato Ausente**. No se pudo calcular el promedio. 🚫")
            if estado_general == "saludable": estado_general = "alerta"

    if not any("🛑" in adv for adv in advertencias) and estado_general != "alerta":
        estado_general = "saludable"

    if estado_general == "saludable" and not any("🚫" in adv for adv in advertencias):
        advertencias.append("El suelo presenta condiciones **óptimas** en las variables clave. 👍")
    
    final_ph = promedio_ph if pd.notna(promedio_ph) else 0.0
    final_al = promedio_aluminio if pd.notna(promedio_aluminio) else 0.0
    final_mo = promedio_mo if pd.notna(promedio_mo) else 0.0
    
    return advertencias, estado_general, final_ph, final_al, final_mo


def calcular_icd(df_muestras, params):
    """Calcula el Índice de Calidad de Datos (ICD)."""
    
    if df_muestras.empty:
        return 0, "Baja", {}
    
    num_muestras = len(df_muestras)
    puntaje_base = 100 
    desglose = {}
    
    for col, key in [('pH_agua_suelo', 'ph'), ('Aluminio intercambiable', 'al')]:
        if df_muestras[col].isnull().all(): # Si toda la columna está nula
            penalidad = params[f'penalidad_nulo_{key}']
            puntaje_base -= penalidad
            desglose[f'Compleción ({col.split()[0]} Nulo Total)'] = -penalidad
        # Si hay valores nulos pero no toda la columna
        elif df_muestras[col].isnull().any():
            penalidad = params[f'penalidad_nulo_{key}'] * (df_muestras[col].isnull().sum() / num_muestras) # Penalidad parcial
            puntaje_base -= penalidad
            desglose[f'Compleción ({col.split()[0]} Nulo Parcial)'] = -round(penalidad)

    # Solo calcular incoherencia si hay datos de pH
    if df_muestras['pH_agua_suelo'].count() > 0:
        muestra_ph_incoherente = ((df_muestras['pH_agua_suelo'] < params['umbral_ph_min']) | (df_muestras['pH_agua_suelo'] > params['umbral_ph_max'])).any()
        if muestra_ph_incoherente:
            penalidad = params['penalidad_incoherente_ph']
            puntaje_base -= penalidad
            desglose['Coherencia (pH Imposible)'] = -penalidad
        
    # Solo calcular outliers si hay suficientes datos de pH (detectar_outliers_iqr maneja esto)
    outlier_count, _ = detectar_outliers_iqr(df_muestras['pH_agua_suelo'])
    tasa_outliers = outlier_count / num_muestras if num_muestras > 0 else 0
    
    if tasa_outliers > params['tasa_outliers_max']:
        penalizacion = params['penalidad_anomalias']
        puntaje_base -= penalizacion
        desglose['Anomalías (Outliers pH IQR)'] = -penalizacion
        
    if (df_muestras['Aluminio intercambiable'] < 0.01).any():
        penalidad = params['penalidad_baja_precision']
        puntaje_base -= penalidad
        desglose['Precisión (Al bajo)'] = -penalidad
        
    if 'Fecha de Análisis' in df_muestras.columns and (df_muestras['Fecha de Análisis'].dt.year < params['año_corte_antiguedad']).any():
        penalidad = params['penalidad_antiguedad']
        puntaje_base -= penalidad
        desglose['Actualidad (Fecha Antigüa)'] = -penalidad
        
    icd_puntaje = max(0, round(puntaje_base)) # Asegurar que el puntaje es un entero no negativo
    
    if icd_puntaje >= 80: calificacion = "Alta"
    elif icd_puntaje >= 50: calificacion = "Media"
    else: calificacion = "Baja"
        
    return icd_puntaje, calificacion, desglose


# --- Funciones de Utilidad y Componentes de Interfaz ---

def draw_layer(label, value, color, margin_top="10px"):
    """Dibuja una capa simulada con iluminación."""
    if color == "red": bg_color = "#FF4B4B" 
    elif color == "#FFD700": bg_color = "#FFD700" 
    elif color == "#6C7A89": bg_color = "#6C7A89" # Color para "N/A"
    else: bg_color = "#196F3D" 
    
    formatted_value = f"{value:.2f}" if isinstance(value, float) and label != "ICD" else str(value)
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color}; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 0 10px 3px {bg_color} ;
            text-align: center; 
            margin-top: {margin_top};
            border: 2px solid {'#FFFFFF' if color != 'green' and color != '#6C7A89' else 'transparent'};
            ">
            <p style="color: white; font-weight: bold; font-size: 1.1em; margin: 0;">{label}</p>
            <p style="color: white; margin: 0;">Promedio: {formatted_value}</p>
        </div>
        """, unsafe_allow_html=True
    )

def show_educational_card(emoji, title, description):
    """Dibuja una tarjeta informativa simple."""
    st.markdown(
        f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; height: 100%;">
            <p style="font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">{emoji} {title}</p>
            <p style="font-size: 0.9em; margin: 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True
    )

def show_map_tab(selected_municipio, selected_departamento):
    """Implementa el mapa usando un enlace a Google Maps."""
    st.subheader(f"📍 Ubicación de **{selected_municipio}** en el Mapa")
    st.info("Para ver la ubicación precisa, presiona el botón. Se abrirá Google Maps en una nueva pestaña.")
    
    # Crea la URL de Google Maps para buscar el municipio y departamento
    map_query = f"{selected_municipio}, {selected_departamento}, Colombia"
    map_url = f"https://www.google.com/maps/search/?api=1&query={map_query.replace(' ', '+')}"
    
    st.link_button("Abrir Mapa en Google Maps", map_url, help=f"Buscar: {map_query}")


# --- 3. FUNCIONES DE INTERFAZ POR NIVEL ---

def show_easy_level(df_diagnostico, selected_municipio, selected_departamento, estado_general, ph_val, al_val, mo_val, icd_puntaje, icd_calificacion, diagnostico):
    """Interfaz para el usuario Novato/Fácil (Diagnóstico Rápido y Educación)."""
    
    st.header("🟢 Nivel Principiante: Diagnóstico Rápido")
    st.markdown("##### **Interfaz educativa con explicaciones básicas. Ideal para usuarios nuevos en análisis de suelos.**")
    
    # --- Tarjetas Educativas ---
    st.markdown("---")
    st.subheader("📚 Conceptos Clave del Suelo")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    with col_e1: show_educational_card("🧪", "pH (Acidez)", "Mide qué tan ácido o alcalino es el suelo. Afecta cómo la planta absorbe nutrientes. Valores de 6.0 a 7.5 son ideales.")
    with col_e2: show_educational_card("🛡️", "**ICD**", "El Índice de Calidad de Datos te dice qué tan confiables son las muestras que estás usando. ¡Un ICD alto es crucial!") # FIX: Título ICD
    with col_e3: show_educational_card("☣️", "Aluminio Intercambiable", "En suelos ácidos, este elemento puede volverse **tóxico** para las raíces. Un valor alto es una señal de peligro.")
    with col_e4: show_educational_card("🌿", "Materia Orgánica (MO)", "El 'alimento' del suelo. Mejora la fertilidad, retención de agua y estructura. Un valor bajo indica que la tierra necesita nutrientes.")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Diagnóstico Rápido", "Sugerencias", "Ubicación en el Mapa"]) 

    # --- VERIFICACIÓN DE DATOS INSUFICIENTES (CORRECCIÓN ABEJORRAL) ---
    datos_insuficientes = estado_general == "peligro" and "Advertencia: No hay datos completos" in diagnostico[0]
    num_muestras_validas = len(df_diagnostico.dropna(subset=['pH_agua_suelo', 'Aluminio intercambiable']))
    if df_diagnostico.empty or num_muestras_validas == 0:
        datos_insuficientes = True

    with tab1:
        st.subheader(f"✅ La Salud del Suelo en **{selected_municipio}**")
        
        col_icd_big, col_general_big = st.columns(2)
        with col_icd_big:
            # FIX: Asegurar que el valor del ICD se maneje bien si es N/A
            icd_display = f"{icd_puntaje}%" if not datos_insuficientes else "N/A"
            st.metric(label="ICD Total (Fiabilidad de los Datos)", value=icd_display, delta=f"Nivel {icd_calificacion}" if not datos_insuficientes else "Baja")
        with col_general_big:
            if datos_insuficientes:
                 st.error(f"🚨 ESTADO: **CRÍTICO** - No hay datos suficientes para un análisis fiable en {selected_municipio}.")
            elif estado_general == "peligro": st.error(f"🚨 ESTADO: **CRÍTICO** - Se requiere acción inmediata en {selected_municipio}")
            elif estado_general == "alerta": st.warning(f"⚠️ ESTADO: **ALERTA** - Se requiere monitoreo y atención en {selected_municipio}")
            else: st.success(f"✅ ESTADO: **ÓPTIMO** - El suelo está en buen estado en {selected_municipio}")
        
        st.markdown("---")
        
        col_visual, col_summary = st.columns([1, 1])
        
        with col_visual:
            st.header("🧱 Análisis de Variables Clave")
            st.caption("Los colores indican el riesgo: Verde (Bajo), Amarillo (Alerta), Gris (Sin Datos).")
            
            # Si los datos son insuficientes, mostramos los visuales en gris
            if datos_insuficientes:
                draw_layer("Nivel de Acidez (pH)", "N/A", "#6C7A89", margin_top="0px")
                draw_layer("Nivel de Toxicidad (Aluminio)", "N/A", "#6C7A89", margin_top="15px")
                draw_layer("Nivel de Fertilidad (Materia Orgánica)", "N/A", "#6C7A89", margin_top="15px")
                
                with col_summary:
                    st.subheader("Resumen Ejecutivo")
                    st.warning("No se puede generar un resumen. La calidad de los datos es CRÍTICA. Vea la pestaña Sugerencias.")
            else:
                # Lógica de color normal
                color_ph = "red" if ph_val < 5.5 or ph_val > 7.5 else "#FFD700" if ph_val < 6.0 else "green" 
                color_al = "red" if al_val > 1.0 else "green"
                color_mo = "#FFD700" if mo_val < 2.0 else "green" 
                
                draw_layer("Nivel de Acidez (pH)", ph_val, color_ph, margin_top="0px")
                draw_layer("Nivel de Toxicidad (Aluminio)", al_val, color_al, margin_top="15px")
                draw_layer("Nivel de Fertilidad (Materia Orgánica)", mo_val, color_mo, margin_top="15px")
                
                with col_summary:
                    st.subheader("Resumen Ejecutivo")
                    st.markdown(f"""
                        El suelo de **{selected_municipio}** presenta un estado general **{estado_general.upper()}**. 
                        Nuestra confianza en los datos (ICD) es **{icd_calificacion.upper()}** ({icd_puntaje}%).
                        
                        **Advertencias Principales:**
                        """)
                    
                    with st.container():
                        advertencias_text = "\n".join([f"- {adv}" for adv in diagnostico])
                        st.markdown(advertencias_text)


    with tab2:
        st.subheader(f"📝 Recomendaciones Simples para **{selected_municipio}**")
        
        if datos_insuficientes:
            st.error(f"🛑 **IMPOSIBLE SUGERIR:** No hay datos de muestras de suelo válidos en **{selected_municipio}** para hacer una recomendación. **La acción inmediata es recolectar muestras.**")
        else:
            st.info(f"Estas sugerencias se basan en **{num_muestras_validas}** muestras de suelo válidas.")
            st.markdown("---")
            
            if estado_general == "peligro":
                st.error("**ACCIÓN INMEDIATA:** Aplique **cal** o enmiendas para corregir la acidez y reducir el aluminio tóxico. Consulte a un agrónomo.")
            elif estado_general == "alerta":
                st.warning("**MONITOREO:** Considere agregar abono orgánico para subir la Materia Orgánica y revise el pH en el próximo ciclo.")
            else:
                st.success("**MANTENIMIENTO:** Las condiciones son favorables. Continúe con las prácticas agrícolas actuales.")
            
    with tab3:
        show_map_tab(selected_municipio, selected_departamento)


def show_intermediate_level(df_diagnostico, selected_municipio, selected_departamento, icd_puntaje, icd_calificacion, desglose):
    """Interfaz para el usuario Intermedio (Desglose del ICD)."""
    
    st.header("🟡 Nivel Intermedio: Detalle Analítico del ICD")
    st.markdown("##### **Ideal para agrónomos y técnicos que necesitan justificar la fiabilidad de las muestras.**")
    
    tab1, tab2 = st.tabs(["Análisis de Calidad", "Ubicación en el Mapa"])
    
    with tab1:
        st.subheader("Desglose Analítico del Índice de Calidad de Datos (ICD)")
        
        st.info(f"El puntaje total para **{selected_municipio}** es **{icd_puntaje}%** ({icd_calificacion}).")

        col_icd, col_penalizaciones = st.columns([1, 1.5])
        
        with col_icd:
            st.metric("Puntaje ICD Total", f"{icd_puntaje}%", icd_calificacion)
            st.markdown("---")
            st.subheader("Causas Principales de Baja Calidad")
            
            penalidades_df = pd.DataFrame(
                {'Dimensión': list(desglose.keys()), 'Penalidad': [abs(v) for v in desglose.values()]}
            ).sort_values(by='Penalidad', ascending=False)
            
            if not penalidades_df.empty:
                st.bar_chart(penalidades_df.set_index('Dimensión'), height=300)
            else:
                st.success("¡Excelente! No se encontraron penalizaciones importantes en estas muestras.")


        with col_penalizaciones:
            st.subheader("Puntos Restados (Detalle)")
            
            with st.container():
                if desglose:
                    for criterio, penalidad in desglose.items():
                        st.markdown(f"- **{criterio}:** Se restaron {abs(penalidad)} puntos. ")
                else:
                    st.info("No hay puntos restados. La calidad de los datos es óptima.")
                
            st.markdown("---")
            st.subheader("Muestras Individuales con Errores Detectados")
            
            _, outliers_ph = detectar_outliers_iqr(df_diagnostico['pH_agua_suelo'])
            
            filas_con_problemas = df_diagnostico[
                (df_diagnostico['pH_agua_suelo'].isnull()) | 
                (df_diagnostico['Aluminio intercambiable'].isnull()) |
                (df_diagnostico.index.isin(outliers_ph.index)) | 
                (df_diagnostico['Fecha de Análisis'].dt.year < ICD_PARAMS['año_corte_antiguedad'])
            ]
            
            with st.container():
                if not filas_con_problemas.empty:
                    st.warning(f"Se encontraron **{len(filas_con_problemas)}** muestras individuales con baja calidad:")
                    st.dataframe(filas_con_problemas[['Cultivo', 'Fecha de Análisis', 'pH_agua_suelo', 'Aluminio intercambiable']].head(5))
                else:
                    st.success("No hay muestras que reporten problemas de fiabilidad individuales.")
                    
    with tab2:
        selected_departamento = df_diagnostico['Departamento'].iloc[0] if not df_diagnostico.empty else "Departamento Desconocido"
        show_map_tab(selected_municipio, selected_departamento)


def show_advanced_level(df_diagnostico, selected_municipio, selected_departamento, icd_puntaje, icd_calificacion, desglose):
    """Interfaz para el usuario Avanzado (Ajuste de Reglas)."""
    
    st.header("🔴 Nivel Experto: Ajuste de Reglas del ICD")
    st.markdown("##### **Para usuarios que desean personalizar los criterios de calidad de datos.**")
    st.info(f"Modifica los parámetros de penalización y los umbrales para recalcular el ICD en **{selected_municipio}**.")
    
    if 'current_icd_params' not in st.session_state:
        st.session_state.current_icd_params = ICD_PARAMS.copy()

    tab1, tab2 = st.tabs(["Ajuste de Reglas", "Ubicación en el Mapa"])
    
    with tab1:
        col_rules, col_results = st.columns([1, 1.5])
        
        with col_rules:
            st.subheader("Parámetros de Penalización (Puntos restados)")
            
            st.session_state.current_icd_params['penalidad_nulo_ph'] = st.slider("Penalidad por pH Nulo:", 0, 50, st.session_state.current_icd_params['penalidad_nulo_ph'], key='adv_p1')
            st.session_state.current_icd_params['penalidad_incoherente_ph'] = st.slider("Penalidad por pH Imposible (<3 o >10):", 0, 50, st.session_state.current_icd_params['penalidad_incoherente_ph'], key='adv_p2')
            st.session_state.current_icd_params['penalidad_anomalias'] = st.slider("Penalidad por Outliers (IQR):", 0, 50, st.session_state.current_icd_params['penalidad_anomalias'], key='adv_p3')
            st.session_state.current_icd_params['penalidad_antiguedad'] = st.slider("Penalidad por Dato Antiguo (> 2018):", 0, 50, st.session_state.current_icd_params['penalidad_antiguedad'], key='adv_p4')
            
            st.subheader("Umbrales Físicos y Estadísticos")
            
            st.session_state.current_icd_params['umbral_ph_min'] = st.number_input("Umbral Mínimo de pH Coherente:", 2.0, 5.0, st.session_state.current_icd_params['umbral_ph_min'], step=0.1, key='adv_u1')
            st.session_state.current_icd_params['tasa_outliers_max'] = st.slider("Tasa Máxima de Outliers antes de penalizar:", 0.01, 0.20, st.session_state.current_icd_params['tasa_outliers_max'], step=0.01, key='adv_u2')

            new_icd_puntaje, new_icd_calificacion, new_desglose = calcular_icd(df_diagnostico, st.session_state.current_icd_params)

        with col_results:
            st.subheader("Resultados del ICD Recalculado")
            
            st.metric("ICD con Reglas Ajustadas", f"{new_icd_puntaje}%", new_icd_calificacion)
            
            st.markdown("---")
            st.subheader("Penalizaciones con sus Reglas:")
            with st.container():
                if new_desglose:
                    for criterio, penalidad in new_desglose.items():
                        st.markdown(f"- **{criterio}:** Se penalizó con {abs(penalidad)} puntos.")
                else:
                    st.success("No hay penalizaciones con estas reglas.")
            
            st.markdown("---")
            st.caption("Nota: El ICD se recalcula automáticamente al mover los deslizadores.")
            st.download_button(
                label="Descargar Reporte de Reglas",
                data=pd.DataFrame(st.session_state.current_icd_params.items(), columns=['Parámetro', 'Valor']).to_csv(index=False),
                file_name=f"ICD_Reglas_{selected_municipio}.csv",
                mime="text/csv"
            )

    with tab2:
        show_map_tab(selected_municipio, selected_departamento)


# --- 4. CONTROLADOR DE VISTA PRINCIPAL ---

def show_level_selector():
    """Pantalla inicial para seleccionar el nivel de experiencia."""
    st.header("Selecciona tu Nivel de Experiencia:")
    st.info("Esto ajustará la complejidad de la interfaz para tu comodidad.")
    
    col_f, col_i, col_a = st.columns(3)
    
    with col_f:
        if st.button("Novato (Fácil) 🟢", use_container_width=True):
            st.session_state.user_level = 'easy'
            st.rerun()
    with col_i:
        if st.button("Intermedio 🟡", use_container_width=True):
            st.session_state.user_level = 'intermediate'
            st.rerun()
    with col_a:
        if st.button("Experto (Avanzado) 🔴", use_container_width=True):
            st.session_state.user_level = 'advanced'
            st.rerun()
            
    st.markdown("---")
    st.caption("El ICD se calcula a partir del archivo 'datos_limpios.csv'.")


def show_main_app():
    """Interfaz principal con filtros y pestañas basadas en el nivel de usuario."""
    
    if df_original.empty: return
    
    # --- FILTROS GLOBALES (Barra Lateral) ---
    st.sidebar.header("🔍 Filtra tu Ubicación")
    st.sidebar.markdown("---")

    departamento_list = sorted(df_original['Departamento'].unique())
    if not departamento_list:
        st.sidebar.warning("No hay datos de departamentos para filtrar.")
        return
        
    selected_departamento = st.sidebar.selectbox("1. Selecciona el Departamento:", departamento_list)

    df_filtrado = df_original[df_original['Departamento'] == selected_departamento]
    municipio_list = sorted(df_filtrado['Municipio'].unique())
    
    if not municipio_list or df_filtrado.empty:
        st.sidebar.warning(f"No hay municipios con datos válidos en **{selected_departamento}**.")
        st.info("No hay datos disponibles para el análisis en este departamento.")
        
        if st.sidebar.button("⬅️ Cambiar Nivel de Experiencia"):
            st.session_state.user_level = None
            st.rerun()
        return

    selected_municipio = st.sidebar.selectbox("2. Selecciona el Municipio:", municipio_list)

    # --- DATAFRAME DE DIAGNÓSTICO (FILTRADO FINAL) ---
    df_diagnostico = df_filtrado[df_filtrado['Municipio'] == selected_municipio]
    
    # --- CÁLCULO CENTRAL ---
    diagnostico, estado_general, ph_val, al_val, mo_val = obtener_diagnostico(df_diagnostico, ICD_PARAMS)
    icd_puntaje, icd_calificacion, desglose = calcular_icd(df_diagnostico, ICD_PARAMS)
    
    
    # --- RENDERIZADO DE PESTAÑAS BASADO EN EL NIVEL ---
    
    if st.session_state.user_level == 'easy':
        show_easy_level(df_diagnostico, selected_municipio, selected_departamento, estado_general, ph_val, al_val, mo_val, icd_puntaje, icd_calificacion, diagnostico)
    elif st.session_state.user_level == 'intermediate':
        show_intermediate_level(df_diagnostico, selected_municipio, selected_departamento, icd_puntaje, icd_calificacion, desglose)
    elif st.session_state.user_level == 'advanced':
        show_advanced_level(df_diagnostico, selected_municipio, selected_departamento, icd_puntaje, icd_calificacion, desglose)
    
    # Botón para volver al selector de nivel
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Cambiar Nivel de Experiencia", use_container_width=True):
        st.session_state.user_level = None
        st.rerun()
    

# --- 5. LÓGICA DE ARRANQUE ---

if st.session_state.user_level is None:
    show_level_selector()
else:
    show_main_app()