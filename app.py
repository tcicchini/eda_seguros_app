"""
Análisis de Cartera — Seguros de Vida
======================================

Aplicación Streamlit para explorar de forma interactiva la cartera de seguros
de vida analizada en `notebooks/eda_seguros.ipynb`. Reutiliza la misma lógica
de limpieza de datos documentada allí y las decisiones de visualización del
reporte `outputs/reporte_figuras.md` (qué figuras incluir, mejorar o
descartar) más los hallazgos nuevos generados en ese reporte.

Esta app NO copia los datos crudos: los lee directamente desde la carpeta
`datos/` del proyecto (ver resolución de `DATA_DIR` más abajo).

"""

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Análisis de Cartera — Seguros de Vida",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paleta de colores (consistente en toda la app)
# ---------------------------------------------------------------------------
# Paleta categórica validada para accesibilidad (contraste y daltonismo):
# slot 1 = azul, slot 2 = naranja (acento). Se mantiene además la asociación
# de género usada en el notebook original (M=azul, F=naranja).
COLOR_PRIMARY = "#2a78d6"      # azul — color principal / serie 1
COLOR_ACCENT = "#eb6834"       # naranja — acento / serie 2
COLOR_MUTED = "#9a9a94"        # gris claro — ejes, etiquetas secundarias (visible sobre fondo oscuro)
COLOR_GRID = "#3a3a38"         # gris oscuro — líneas de grilla sobre fondo oscuro
COLOR_REFERENCE = "#c3c2b7"    # gris claro — líneas de referencia/mediana global
COLOR_TEXT_SECONDARY = "#e8e7e2"  # texto secundario claro (anotaciones sobre fondo oscuro)
COLOR_BG = "#1a1a19"           # fondo oscuro de los gráficos

GENDER_COLOR_MAP = {"M": COLOR_PRIMARY, "F": COLOR_ACCENT}

# Rampa secuencial de azules (magnitud), de más clara a más oscura.
BLUE_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Tema visual: fondo oscuro en todos los gráficos plotly, con texto blanco explícito.
PLOTLY_TEMPLATE = "plotly_dark"

CHART_FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="white")


def style_fig(fig, height=420, showlegend=True):
    """Aplica un estilo oscuro consistente (fondo oscuro, texto blanco, grilla tenue) a cada figura."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="white"),
        height=height,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="white")),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zeroline=False, linecolor=COLOR_MUTED, color="white")
    fig.update_yaxes(gridcolor=COLOR_GRID, zeroline=False, linecolor=COLOR_MUTED, color="white")
    return fig


# ---------------------------------------------------------------------------
# Resolución de la carpeta de datos
# ---------------------------------------------------------------------------
# La app NO incluye datos propios: los busca en una carpeta `datos/` ubicada
# junto al proyecto. Se contemplan dos layouts posibles y una variable de
# entorno de override explícito, en ese orden de prioridad:
#   1) SEGUROS_EDA_DATA_DIR      -> override explícito (variable de entorno)
#   2) <raíz del proyecto>/seguros_eda_app/../seguros_eda/datos
#      (estructura solicitada: seguros_eda_app y seguros_eda como carpetas
#      hermanas dentro del home del usuario)
#   3) <raíz del proyecto>/datos
#      (esta app vive DENTRO de la carpeta del proyecto, junto a datos/,
#      notebooks/ y outputs/ — es el layout usado para probar esta app)
APP_DIR = Path(__file__).resolve().parent

def _resolve_data_dir() -> Path:
    env_override = os.environ.get("SEGUROS_EDA_DATA_DIR", "").strip()
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(APP_DIR.parent / "seguros_eda" / "datos")
    candidates.append(APP_DIR.parent / "datos")

    for c in candidates:
        if (c / "Super_reducido_Prod1.parquet").exists() or (c / "Super_reducido_Prod1.xlsx").exists():
            return c
    # Si ninguna existe, devolvemos la más probable para que el mensaje de
    # error en pantalla indique una ruta concreta y accionable.
    return candidates[-1]


DATA_DIR = _resolve_data_dir()
# Preferimos el Parquet (carga mucho más rápida); si no existe, caemos al
# Excel original como fallback.
_DATA_FILE_PARQUET = DATA_DIR / "Super_reducido_Prod1.parquet"
DATA_FILE = _DATA_FILE_PARQUET if _DATA_FILE_PARQUET.exists() else DATA_DIR / "Super_reducido_Prod1.xlsx"

# ---------------------------------------------------------------------------
# Carga y limpieza de datos (misma lógica que notebooks/eda_seguros.ipynb,
# Sección 1.4 — "Aplicamos las correcciones documentadas")
# ---------------------------------------------------------------------------
AGE_BINS = [0, 30, 40, 50, 60, 150]
AGE_LABELS = ["<30", "30-40", "40-50", "50-60", "60+"]


@st.cache_data(show_spinner="Cargando y limpiando la base de datos (puede tardar 1-3 minutos la primera vez)...")
def load_clean_data(data_file: str) -> pd.DataFrame:
    if str(data_file).endswith(".parquet"):
        df = pd.read_parquet(data_file)
        # pandas 3 infiere las columnas de texto leídas de un Parquet con su
        # dtype "str" (arrow), más estricto que el "object" que devuelve
        # read_excel: no acepta que más abajo se le asignen valores float
        # (edad recalculada) a esta columna. Lo restauramos a "object" para
        # que el resto de la función se comporte igual con ambos formatos.
        df["Edad"] = df["Edad"].astype(object)
        df["Cliente Asegurado[Fecha Nacimiento]"] = df["Cliente Asegurado[Fecha Nacimiento]"].astype(object)
    else:
        df = pd.read_excel(data_file, sheet_name="Hoja1")
    d = df.copy()

    # (1) y (2): recalcular Edad para los registros con "(nulo)"
    mask_nulo = d["Edad"] == "(nulo)"
    fecha_nac = pd.to_datetime(d["Cliente Asegurado[Fecha Nacimiento]"], errors="coerce")
    edad_recalculada = (d.loc[mask_nulo, "occurDate"] - fecha_nac.loc[mask_nulo]).dt.days / 365.25
    d.loc[mask_nulo, "Edad"] = edad_recalculada
    d["Edad"] = pd.to_numeric(d["Edad"], errors="coerce")
    d["Cliente Asegurado[Fecha Nacimiento]"] = fecha_nac

    # Variable de estatus
    d["Status"] = np.where(d["Siniestro"] == 1, "Fallecido", "No Fallecido")
    d["is_siniestro"] = d["Siniestro"]

    # (3) Remover Edad_ingreso > 100 (fechas de nacimiento centinela)
    mask_edad_valida = d["Edad_ingreso"] <= 100
    d = d[mask_edad_valida].copy()

    # (4) Marcar (sin eliminar) montos negativos
    d["suma_valida"] = d["Suma_asegurada"] >= 0

    # Monto en dólares (reemplaza a Suma_UMS y Suma_asegurada en toda la UI).
    # 1 UMS = 228.700 ARS; 1 USD = 1.500 ARS. Se calcula acá (y no se elimina
    # Suma_UMS/Suma_asegurada) porque el segmento por cuartil, más abajo, ya
    # se calcula sobre este monto en dólares.
    d["suma_usd"] = d["Suma_UMS"] * 228700 / 1500

    # Deciles etarios (se mantienen por compatibilidad, no se usan en la app)
    d["age_decile"] = pd.qcut(d["Edad_ingreso"], 10, duplicates="drop")

    # Rangos etarios "de negocio" (Sección 3 de la app)
    d["rango_edad"] = pd.cut(d["Edad_ingreso"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    # Segmento por cuartil de Suma Asegurada, en USD (Sección 2) — se calcula
    # una única vez sobre TODA la base para que los cortes de cuartil no
    # cambien con los filtros (comparabilidad entre selecciones).
    d["segmento_suma"] = pd.Series(pd.NA, index=d.index, dtype="object")
    valid_idx = d.index[d["suma_valida"]]
    d.loc[valid_idx, "segmento_suma"] = pd.qcut(
        d.loc[valid_idx, "suma_usd"], 4, labels=["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"]
    ).astype(str)

    for col in ["sexo", "Unidad de Negocio[Unidad Negocios]", "Status", "segmento_suma", "rango_edad"]:
        if col in d.columns:
            d[col] = d[col].astype("category")

    return d


# ---------------------------------------------------------------------------
# Carga alternativa vía uploader manual (fallback cuando no existe DATA_FILE)
# ---------------------------------------------------------------------------
# Columnas crudas esperadas de la base de siniestros de vida (mismo esquema
# que `Super_reducido_Prod1.xlsx`, hoja "Hoja1").
EXPECTED_COLUMNS = [
    "Cliente Asegurado[Fecha Nacimiento]",
    "Edad_ingreso",
    "Poliza/Certificado[Fecha Origen Certificado]",
    "Intermediario[Codigo Organizador - Organizador]",
    "Unidad de Negocio[Unidad Negocios]",
    "Refcert",
    "Suma_asegurada",
    "Suma_UMS",
    "Siniestro",
    "InicioVigencia",
    "occurDate",
    "dias_hasta_sin",
    "lossNoticeDate",
    "demora_den",
    "Edad",
    "sexo",
]


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Misma lógica de limpieza que `load_clean_data` (ver esa función para el
    detalle de cada paso), duplicada acá para poder aplicarla también sobre un
    archivo cargado manualmente por el usuario sin tocar `load_clean_data`."""
    d = df.copy()

    mask_nulo = d["Edad"] == "(nulo)"
    fecha_nac = pd.to_datetime(d["Cliente Asegurado[Fecha Nacimiento]"], errors="coerce")
    edad_recalculada = (d.loc[mask_nulo, "occurDate"] - fecha_nac.loc[mask_nulo]).dt.days / 365.25
    d.loc[mask_nulo, "Edad"] = edad_recalculada
    d["Edad"] = pd.to_numeric(d["Edad"], errors="coerce")
    d["Cliente Asegurado[Fecha Nacimiento]"] = fecha_nac

    d["Status"] = np.where(d["Siniestro"] == 1, "Fallecido", "No Fallecido")
    d["is_siniestro"] = d["Siniestro"]

    mask_edad_valida = d["Edad_ingreso"] <= 100
    d = d[mask_edad_valida].copy()

    d["suma_valida"] = d["Suma_asegurada"] >= 0

    # Monto en dólares (reemplaza a Suma_UMS y Suma_asegurada en toda la UI).
    # 1 UMS = 228.700 ARS; 1 USD = 1.500 ARS.
    d["suma_usd"] = d["Suma_UMS"] * 228700 / 1500

    d["age_decile"] = pd.qcut(d["Edad_ingreso"], 10, duplicates="drop")
    d["rango_edad"] = pd.cut(d["Edad_ingreso"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    d["segmento_suma"] = pd.Series(pd.NA, index=d.index, dtype="object")
    valid_idx = d.index[d["suma_valida"]]
    d.loc[valid_idx, "segmento_suma"] = pd.qcut(
        d.loc[valid_idx, "suma_usd"], 4, labels=["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"]
    ).astype(str)

    for col in ["sexo", "Unidad de Negocio[Unidad Negocios]", "Status", "segmento_suma", "rango_edad"]:
        if col in d.columns:
            d[col] = d[col].astype("category")

    return d


@st.cache_data(show_spinner="Procesando el archivo cargado...")
def process_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Lee y limpia un archivo subido manualmente (.csv, .xlsx o .parquet).

    Cachea por contenido del archivo (`file_bytes` como parte de la clave de
    cache) para no reprocesar si el usuario no cambió el archivo. Lanza
    `ValueError` si al archivo le faltan columnas esperadas del esquema
    original de la base de siniestros de vida.
    """
    name_lower = file_name.lower()
    if name_lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif name_lower.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Hoja1")
        except ValueError:
            df = pd.read_excel(io.BytesIO(file_bytes))

    faltantes = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(
            "El archivo no tiene las columnas esperadas de la base de siniestros de vida. "
            "Faltan las siguientes columnas: " + ", ".join(faltantes)
        )

    if name_lower.endswith(".csv"):
        # `_clean_dataframe` opera sobre `occurDate` como fecha (necesita restar
        # fechas para recalcular la edad de los registros "(nulo)"). Un CSV no
        # tiene tipos de columna propios como un .xlsx, así que lo normalizamos acá.
        df["occurDate"] = pd.to_datetime(df["occurDate"], errors="coerce")
    elif name_lower.endswith(".parquet"):
        # Mismo motivo que en `load_clean_data`: Parquet devuelve estas
        # columnas con el dtype estricto "str" de pandas 3.
        df["Edad"] = df["Edad"].astype(object)
        df["Cliente Asegurado[Fecha Nacimiento]"] = df["Cliente Asegurado[Fecha Nacimiento]"].astype(object)

    return _clean_dataframe(df)


def hist_binned(serie: pd.Series, bins: int = 60):
    """Histograma pre-agregado con numpy (evita mandarle al navegador arrays
    de cientos de miles de filas). Devuelve centros de bin y conteos."""
    s = serie.dropna()
    if s.empty:
        return np.array([]), np.array([])
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


# ---------------------------------------------------------------------------
# Filtros reutilizables
# ---------------------------------------------------------------------------
UNIDAD_COL = "Unidad de Negocio[Unidad Negocios]"


def unidad_filter(d: pd.DataFrame, key: str):
    unidades = sorted(d[UNIDAD_COL].dropna().unique().tolist())
    seleccion = st.multiselect(
        "Unidad de negocio", options=unidades, default=unidades, key=f"unidad_{key}"
    )
    return seleccion


def genero_filter(key: str):
    seleccion = st.multiselect(
        "Género", options=["F", "M"], default=["F", "M"], key=f"genero_{key}"
    )
    return seleccion


def sin_datos():
    st.warning("No hay datos para la selección actual")


# ---------------------------------------------------------------------------
# Panel de KPIs (header, siempre visible)
# ---------------------------------------------------------------------------
def compute_kpis(d: pd.DataFrame) -> dict:
    n_total = len(d)
    n_siniestros = int(d["is_siniestro"].sum())
    incidencia = n_siniestros / n_total * 100 if n_total else 0.0
    monto_total_usd = d.loc[d["suma_valida"], "suma_usd"].sum()
    unidad_counts = d[UNIDAD_COL].value_counts()
    top3_pct = unidad_counts.head(3).sum() / unidad_counts.sum() * 100 if len(unidad_counts) else 0.0
    edad_mediana_ingreso = d["Edad_ingreso"].median()
    edad_p25_ingreso = d["Edad_ingreso"].quantile(0.25)
    edad_p75_ingreso = d["Edad_ingreso"].quantile(0.75)
    return dict(
        n_total=n_total,
        incidencia=incidencia,
        monto_total_usd=monto_total_usd,
        top3_pct=top3_pct,
        edad_mediana_ingreso=edad_mediana_ingreso,
        edad_p25_ingreso=edad_p25_ingreso,
        edad_p75_ingreso=edad_p75_ingreso,
    )


def fmt_usd(valor: float) -> str:
    """Formatea un valor en USD de forma adaptativa según su magnitud.
    Separador decimal: punto. Separador de miles: punto.
    < 1.000          → 'USD X'
    1.000–999.999    → 'USD X.Xk'
    1M–999M          → 'USD X.X M'
    >= 1.000M        → 'USD X.XXX M'
    """
    if abs(valor) < 1_000:
        return f"USD {valor:,.0f}".replace(",", ".")
    elif abs(valor) < 1_000_000:
        return f"USD {valor/1_000:,.1f}k".replace(",", ".")
    elif abs(valor) < 1_000_000_000:
        return f"USD {valor/1_000_000:,.1f} M".replace(",", ".")
    else:
        return f"USD {valor/1_000_000:,.0f} M".replace(",", ".")


def render_kpi_header(d: pd.DataFrame):
    st.title("Análisis de Cartera — Seguros de Vida")
    kpis = compute_kpis(d)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de asegurados", f"{kpis['n_total']:,.0f}".replace(",", "."))
    c2.metric(
        "Incidencia global", f"{kpis['incidencia']:.2f}%",
        help="Incidencia de muerte global: siniestros sobre el total de asegurados de la cartera.",
    )
    c3.metric(
        "Monto expuesto",
        fmt_usd(kpis['monto_total_usd']),
        help="Monto total expuesto en Suma Asegurada (USD). Formato adaptativo: k = miles, M = millones.",
    )
    c4.metric(
        "Top-3 unidades", f"{kpis['top3_pct']:.1f}%",
        help="Concentración de cartera: % de asegurados en las 3 unidades de negocio con mayor cantidad de clientes.",
    )
    c5.metric(
        "Edad mediana de ingreso", f"{kpis['edad_mediana_ingreso']:.0f} años",
        help=(
            f"P25: {kpis['edad_p25_ingreso']:.0f} años — "
            f"Mediana: {kpis['edad_mediana_ingreso']:.0f} años — "
            f"P75: {kpis['edad_p75_ingreso']:.0f} años"
        ),
    )

    st.caption(
        "El monto expuesto representa el capital máximo potencial, "
        "no el costo esperado de siniestros."
    )
    st.divider()


@st.cache_data(show_spinner=False)
def compute_aggregados_globales(df_hash: int, _d: pd.DataFrame) -> dict:
    """Pre-computa todos los agregados que no dependen de filtros."""
    incidencia_global = _d["is_siniestro"].sum() / len(_d) * 100

    riesgo_unidad = _d.groupby(UNIDAD_COL, observed=True)["is_siniestro"].agg(["sum", "count"])
    riesgo_unidad.columns = ["siniestros", "total_asegurados"]
    riesgo_unidad["pct_siniestros"] = riesgo_unidad["siniestros"] / riesgo_unidad["total_asegurados"] * 100
    riesgo_unidad = riesgo_unidad.sort_values("pct_siniestros", ascending=True)

    monto_total_unidad = _d[_d["suma_valida"]].groupby(UNIDAD_COL, observed=True)["suma_usd"].sum()

    base_unidad = _d.groupby(UNIDAD_COL, observed=True).agg(
        n_polizas=("Refcert", "count"), siniestros=("is_siniestro", "sum")
    )
    base_unidad["monto_total_usd"] = monto_total_unidad
    base_unidad["pct_siniestros"] = base_unidad["siniestros"] / base_unidad["n_polizas"] * 100
    base_unidad = base_unidad.dropna(subset=["monto_total_usd"])

    orden_unidad_clientes = _d[UNIDAD_COL].value_counts().sort_values(ascending=True)
    orden_unidad_monto = _d[_d["suma_valida"]].groupby(
        UNIDAD_COL, observed=True)["suma_usd"].sum().sort_values(ascending=True)

    sexo_counts = _d["sexo"].value_counts()

    bounds = _d.dropna(subset=["segmento_suma"]).groupby(
        "segmento_suma", observed=True)["suma_usd"].agg(["min", "max"])

    return dict(
        incidencia_global=incidencia_global,
        riesgo_unidad=riesgo_unidad,
        base_unidad=base_unidad,
        orden_unidad_clientes=orden_unidad_clientes,
        orden_unidad_monto=orden_unidad_monto,
        sexo_counts=sexo_counts,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# Sección 1 — Composición de la cartera
# ---------------------------------------------------------------------------
def section_composicion(d: pd.DataFrame, agregados: dict):
    st.header("1. Composición de la cartera")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        unidades_sel = unidad_filter(d, "s1")
    with fcol2:
        generos_sel = genero_filter("s1")

    dd = d[d[UNIDAD_COL].isin(unidades_sel) & d["sexo"].isin(generos_sel)]
    if dd.empty:
        sin_datos()
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de edad de ingreso")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dd["Edad_ingreso"], xbins=dict(size=5), marker_color=COLOR_PRIMARY))
        style_fig(fig, showlegend=False)
        fig.update_xaxes(title="Edad de ingreso")
        fig.update_yaxes(title="Cantidad de asegurados")
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Muestra en qué edades se concentra el ingreso a la cartera (bins de 5 años). "
            "La mayoría de los asegurados ingresa entre los 30 y los 55 años."
        )

    with col2:
        st.subheader("Composición por sexo")
        st.caption("Vista global de la cartera — no se ve afectada por los filtros")
        sexo_counts = agregados["sexo_counts"]
        fig = px.pie(
            values=sexo_counts.values, names=sexo_counts.index, hole=0.45,
            color=sexo_counts.index, color_discrete_map=GENDER_COLOR_MAP,
        )
        style_fig(fig, height=420)
        st.plotly_chart(fig, width='stretch')
        st.caption("Proporción de hombres y mujeres en el total de la cartera.")

    st.subheader("Distribución de cartera por unidad de negocio")
    st.caption("Vista global de la cartera — no se ve afectada por los filtros")
    vista_unidad = st.radio(
        "Mostrar distribución por:",
        options=["Clientes", "Monto asegurado (USD)"],
        horizontal=True,
        key="radio_dist_unidad",
    )
    if vista_unidad == "Clientes":
        orden = agregados["orden_unidad_clientes"]
        fig = px.bar(
            x=orden.values, y=orden.index, orientation="h",
            labels={"x": "Cantidad de asegurados", "y": ""},
        )
        fig.update_traces(marker_color=COLOR_PRIMARY, marker_line_width=0)
        style_fig(fig, height=max(380, 28 * len(orden)), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Tamaño relativo de cada unidad de negocio. La cartera está fuertemente "
            "concentrada en pocas unidades (ver KPI de concentración top-3)."
        )
    else:
        monto_unidad = agregados["orden_unidad_monto"]
        fig = px.bar(
            x=monto_unidad.values, y=monto_unidad.index, orientation="h",
            labels={"x": "Capital asegurado (USD)", "y": ""},
        )
        fig.update_traces(marker_color=COLOR_PRIMARY, marker_line_width=0)
        style_fig(fig, height=max(380, 28 * len(monto_unidad)), showlegend=False)
        x_max = monto_unidad.max()
        magnitud = 10 ** int(np.floor(np.log10(x_max))) if x_max > 0 else 1
        tick_step = magnitud / 2
        tickvals = np.arange(0, x_max * 1.15, tick_step)
        ticktext = [fmt_usd(v) for v in tickvals]
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
        st.plotly_chart(fig, width='stretch')
        st.caption("Capital total asegurado por unidad de negocio, sobre la cartera completa.")

    st.subheader("Distribución de edad de ingreso por sexo")
    fig = go.Figure()
    for genero in ["M", "F"]:
        sub = dd.loc[dd["sexo"] == genero, "Edad_ingreso"]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(x=sub, xbins=dict(size=5), name=genero, marker_color=GENDER_COLOR_MAP[genero], opacity=0.75))
    fig.update_layout(barmode="overlay")
    style_fig(fig)
    fig.update_xaxes(title="Edad de ingreso")
    fig.update_yaxes(title="Cantidad de asegurados")
    st.plotly_chart(fig, width='stretch')
    st.caption("Compara la forma de la distribución etaria entre hombres y mujeres en la selección actual (bins de 5 años).")
    st.caption("La distribución etaria es similar entre hombres y mujeres, sin diferencias relevantes entre géneros.")


# ---------------------------------------------------------------------------
# Sección 2 — Riesgo y siniestralidad
# ---------------------------------------------------------------------------
def section_riesgo(d: pd.DataFrame, agregados: dict):
    st.header("2. Riesgo y siniestralidad")

    unidades_sel = unidad_filter(d, "s2")
    dd = d[d[UNIDAD_COL].isin(unidades_sel)]
    if dd.empty:
        sin_datos()
        return

    incidencia_global_filtrada = dd["is_siniestro"].sum() / len(dd) * 100
    incidencia_global_total = agregados["incidencia_global"]

    st.subheader("Incidencia de siniestros por unidad de negocio")
    st.caption("Vista de cartera total — no se ve afectada por los filtros")
    riesgo_unidad = agregados["riesgo_unidad"]
    fig = px.bar(
        riesgo_unidad.reset_index(), x="pct_siniestros", y=UNIDAD_COL, orientation="h",
        color="total_asegurados", color_continuous_scale=BLUE_SEQUENTIAL,
        labels={"pct_siniestros": "% de siniestros", UNIDAD_COL: "", "total_asegurados": "Total asegurados"},
    )
    fig.add_vline(x=incidencia_global_total, line_dash="dash", line_color=COLOR_REFERENCE,
                  annotation_text=f"Incidencia global ({incidencia_global_total:.2f}%)",
                  annotation_font_color=COLOR_TEXT_SECONDARY, annotation_position="bottom")
    style_fig(fig, height=max(380, 28 * len(riesgo_unidad)), showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "El volumen de siniestros y el riesgo relativo cuentan historias distintas: unidades chicas "
        "pueden tener una proporción de siniestros mucho mayor que unidades grandes."
    )

    st.subheader("Exposición, siniestralidad y capital asegurado por unidad")
    st.caption("Vista de cartera total — no se ve afectada por los filtros")
    base_unidad = agregados["base_unidad"]
    if base_unidad.empty:
        st.info("No hay datos suficientes por unidad.")
    else:
        fig = px.scatter(
            base_unidad.reset_index(), x="n_polizas", y="pct_siniestros",
            size="monto_total_usd", color="monto_total_usd", color_continuous_scale=BLUE_SEQUENTIAL,
            text=UNIDAD_COL, size_max=55, log_x=True,
            labels={
                "n_polizas": "N° de asegurados (escala log)",
                "pct_siniestros": "% de siniestros",
                "monto_total_usd": "Capital asegurado (USD)",
            },
        )
        fig.update_traces(textposition="top center", marker_line_color="white", marker_line_width=0.6)
        cbar_max = base_unidad["monto_total_usd"].max()
        cbar_magnitud = 10 ** int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 1
        cbar_step = cbar_magnitud / 2
        cbar_tickvals = np.arange(0, cbar_max * 1.05, cbar_step)
        cbar_ticktext = [fmt_usd(v) for v in cbar_tickvals]
        fig.update_layout(coloraxis_colorbar=dict(title="Capital (USD)", tickvals=cbar_tickvals, ticktext=cbar_ticktext))
        fig.add_hline(y=incidencia_global_total, line_dash="dash", line_color=COLOR_REFERENCE,
                      annotation_text=f"incidencia global ({incidencia_global_total:.2f}%)", annotation_font_color=COLOR_TEXT_SECONDARY)
        x_min = base_unidad["n_polizas"].min()
        x_max = base_unidad["n_polizas"].max()
        fig.update_xaxes(range=[np.log10(x_min) - 0.35, np.log10(x_max) + 0.55])
        style_fig(fig, height=520, showlegend=False)
        fig.update_layout(margin=dict(l=10, r=40, t=60, b=10))
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Cada burbuja es una unidad de negocio. Su posición muestra cuántos clientes tiene y qué proporción "
            "siniestró. El tamaño y el color indican el capital total asegurado en dólares: burbujas más grandes "
            "y más oscuras concentran mayor volumen de capital."
        )

    st.subheader("Incidencia de siniestros por segmento de Suma Asegurada")
    seg = dd.dropna(subset=["segmento_suma"]).groupby("segmento_suma", observed=True).agg(
        n_polizas=("Refcert", "count"), siniestros=("is_siniestro", "sum")
    )
    orden_seg = ["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"]
    seg = seg.reindex([s for s in orden_seg if s in seg.index])
    if seg.empty:
        st.info("No hay registros con Suma Asegurada válida en la selección actual.")
    else:
        seg["pct_siniestros"] = seg["siniestros"] / seg["n_polizas"] * 100

        # Límites reales de cada cuartil (min/max de suma_usd por grupo, en USD),
        # pre-calculados sobre toda la cartera en compute_aggregados_globales
        # para que las etiquetas no cambien con los filtros.
        bounds = agregados["bounds"].reindex(orden_seg)
        q1_max = bounds.loc["Q1 (más bajo)", "max"]
        q2_max = bounds.loc["Q2", "max"]
        q3_max = bounds.loc["Q3", "max"]
        etiquetas = {
            "Q1 (más bajo)": f"Q1 (0 – {fmt_usd(q1_max)})",
            "Q2": f"Q2 ({fmt_usd(q1_max)} – {fmt_usd(q2_max)})",
            "Q3": f"Q3 ({fmt_usd(q2_max)} – {fmt_usd(q3_max)})",
            "Q4 (más alto)": f"Q4 ({fmt_usd(q3_max)}+)",
        }
        seg = seg.reset_index()
        seg["segmento_suma"] = seg["segmento_suma"].map(etiquetas)

        fig = px.bar(seg, x="segmento_suma", y="pct_siniestros",
                     labels={"segmento_suma": "Segmento por Suma Asegurada", "pct_siniestros": "% de siniestros"})
        fig.update_traces(marker_color=COLOR_PRIMARY, marker_line_width=0)
        fig.add_hline(y=incidencia_global_filtrada, line_dash="dash", line_color=COLOR_REFERENCE,
                      annotation_text=f"incidencia de la selección ({incidencia_global_filtrada:.2f}%)", annotation_font_color=COLOR_TEXT_SECONDARY)
        style_fig(fig, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "La suma asegurada se divide en cuatro grupos de igual tamaño (cuartiles). "
            "Q1 agrupa las pólizas de menor cobertura y Q4 las de mayor cobertura. Se muestra qué proporción de "
            "asegurados siniestró en cada grupo."
        )

    st.subheader("Días hasta el siniestro")
    siniestros_dd = dd[dd["is_siniestro"] == 1]
    if siniestros_dd.empty:
        sin_datos()
    else:
        mediana_dias = siniestros_dd["dias_hasta_sin"].median()
        centers, counts = hist_binned(siniestros_dd["dias_hasta_sin"], bins=60)
        fig = px.bar(x=centers, y=counts, labels={"x": "Días hasta el siniestro", "y": "Cantidad de siniestros"})
        fig.update_traces(marker_color=COLOR_PRIMARY, marker_line_width=0)
        fig.add_vline(x=mediana_dias, line_dash="dash", line_color=COLOR_REFERENCE,
                      annotation_text=f"Mediana: {mediana_dias:.0f} días", annotation_font_color=COLOR_TEXT_SECONDARY)
        style_fig(fig, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Indica en qué etapa de la vida de la póliza ocurren los siniestros. La mayor concentración "
            "sucede dentro de los primeros años de vigencia."
        )

    st.subheader("Suma asegurada por estatus (fallecido vs. no fallecido)")
    d_valid = dd[dd["suma_valida"]].copy()
    if d_valid.empty:
        sin_datos()
    else:
        fig = px.box(
            d_valid, x="Status", y="suma_usd", color="Status",
            category_orders={"Status": ["No Fallecido", "Fallecido"]},
            color_discrete_map={"No Fallecido": COLOR_PRIMARY, "Fallecido": COLOR_ACCENT},
            log_y=True,
            labels={"suma_usd": "Suma Asegurada (USD, escala log)", "Status": ""},
        )
        style_fig(fig, showlegend=False)
        y_min = d_valid["suma_usd"].replace(0, np.nan).dropna().min()
        y_max = d_valid["suma_usd"].max()
        exp_min = int(np.floor(np.log10(y_min))) if y_min > 0 else 0
        exp_max = int(np.ceil(np.log10(y_max))) if y_max > 0 else 6
        tickvals = [10**e for e in range(exp_min, exp_max + 1)]
        ticktext = [fmt_usd(v) for v in tickvals]
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Se usa escala logarítmica porque la Suma Asegurada tiene valores extremos que aplastarían "
            "la comparación en escala lineal. Permite ver si los siniestros ocurren en pólizas de mayor o menor cobertura."
        )


# ---------------------------------------------------------------------------
# Sección 3 — Perfil etario y cobertura
# ---------------------------------------------------------------------------
def section_etario(d: pd.DataFrame):
    st.header("3. Perfil etario y cobertura")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        generos_sel = genero_filter("s3")
    with fcol2:
        unidades_sel = unidad_filter(d, "s3")

    dd = d[d["sexo"].isin(generos_sel) & d[UNIDAD_COL].isin(unidades_sel)]
    if dd.empty:
        sin_datos()
        return

    incidencia_seleccion = dd["is_siniestro"].sum() / len(dd) * 100
    incidencia_global_total = d["is_siniestro"].sum() / len(d) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Incidencia de siniestros por rango etario")
        by_edad = dd.dropna(subset=["rango_edad"]).groupby("rango_edad", observed=True).agg(
            n=("is_siniestro", "count"), eventos=("is_siniestro", "sum")
        ).reindex(AGE_LABELS)
        by_edad["incidencia"] = by_edad["eventos"] / by_edad["n"] * 100
        fig = px.bar(by_edad.reset_index(), x="rango_edad", y="incidencia",
                     labels={"rango_edad": "Rango etario (edad de ingreso)", "incidencia": "% de siniestros"})
        fig.update_traces(marker_color=COLOR_PRIMARY, marker_line_width=0)
        fig.add_hline(y=incidencia_seleccion, line_dash="dash", line_color=COLOR_REFERENCE,
                      annotation_text=f"Incidencia de la selección ({incidencia_seleccion:.2f}%)",
                      annotation_font_color=COLOR_TEXT_SECONDARY, annotation_position="top left")
        fig.add_hline(y=incidencia_global_total, line_dash="dot", line_color=COLOR_ACCENT,
                      annotation_text=f"Incidencia global (cartera completa) ({incidencia_global_total:.2f}%)",
                      annotation_font_color=COLOR_TEXT_SECONDARY, annotation_position="bottom left")
        style_fig(fig, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "El riesgo crece con la edad de ingreso, pero de forma no lineal: el salto más marcado "
            "ocurre a partir de los 50 años. La línea punteada muestra la incidencia de la selección actual; "
            "la línea de puntos, la incidencia global fija de toda la cartera."
        )

    with col2:
        st.subheader("Suma asegurada promedio por rango etario")
        d_valid = dd[dd["suma_valida"]].dropna(subset=["rango_edad"])
        if d_valid.empty:
            st.info("No hay registros con Suma Asegurada válida en la selección actual.")
        else:
            by_edad_suma = (
                d_valid.groupby("rango_edad", observed=True)["suma_usd"].mean().reindex(AGE_LABELS)
            )
            fig = px.bar(x=by_edad_suma.index, y=by_edad_suma.values,
                         labels={"x": "Rango etario (edad de ingreso)", "y": "Suma Asegurada promedio (USD)"})
            fig.update_traces(marker_color=COLOR_ACCENT, marker_line_width=0)
            style_fig(fig, showlegend=False)
            y_vals = by_edad_suma.dropna().values
            if len(y_vals):
                y_max = y_vals.max()
                magnitud = 10 ** int(np.floor(np.log10(y_max))) if y_max > 0 else 1
                tick_step = magnitud / 2
                tickvals = np.arange(0, y_max * 1.15, tick_step)
                ticktext = [fmt_usd(v) for v in tickvals]
                fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Muestra si los asegurados de distintas edades contratan, en promedio, coberturas de "
                "distinto tamaño."
            )

    st.subheader("Edad de ingreso por estatus (fallecido vs. no fallecido)")
    fig = px.box(
        dd, x="Status", y="Edad_ingreso", color="Status",
        category_orders={"Status": ["No Fallecido", "Fallecido"]},
        color_discrete_map={"No Fallecido": COLOR_PRIMARY, "Fallecido": COLOR_ACCENT},
        labels={"Edad_ingreso": "Edad de ingreso", "Status": ""},
    )
    style_fig(fig, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Los asegurados que siniestran ingresaron a edades significativamente mayores que quienes no siniestran."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _render_navegacion(d: pd.DataFrame, fuente_caption: str):
    st.sidebar.title("Navegación")
    seccion = st.sidebar.radio(
        "Ir a la sección:",
        [
            "1. Composición de la cartera",
            "2. Riesgo y siniestralidad",
            "3. Perfil etario y cobertura",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(fuente_caption)

    if seccion.startswith("1."):
        agregados = compute_aggregados_globales(df_hash=len(d), _d=d)
        section_composicion(d, agregados)
    elif seccion.startswith("2."):
        agregados = compute_aggregados_globales(df_hash=len(d), _d=d)
        section_riesgo(d, agregados)
    elif seccion.startswith("3."):
        section_etario(d)


def main():
    # 1) Intentar cargar automáticamente desde el path local ya definido.
    if DATA_FILE.exists():
        d = load_clean_data(str(DATA_FILE))
        render_kpi_header(d)
        _render_navegacion(d, "Datos leídos desde:\n\n" f"`{DATA_DIR}`")
        return

    # 2) El archivo no existe en el path local -> pantalla de carga manual,
    # mostrando únicamente el uploader (sin contenido del dashboard).
    upload_container = st.empty()
    with upload_container.container():
        st.title("Análisis de Cartera — Seguros de Vida")
        _, col_centro, _ = st.columns([1, 2, 1])
        with col_centro:
            st.markdown("### Cargá el archivo de datos para iniciar el dashboard")
            archivo = st.file_uploader("Archivo de datos", type=["csv", "xlsx", "parquet"])
            st.caption("El archivo debe contener las columnas originales de la base de siniestros de vida")

    if archivo is None:
        st.stop()

    try:
        d = process_uploaded_file(archivo.getvalue(), archivo.name)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    upload_container.empty()

    render_kpi_header(d)
    _render_navegacion(
        d,
        f"Datos cargados manualmente:\n\n`{archivo.name}`\n\n{len(d):,} registros.".replace(",", "."),
    )


if __name__ == "__main__":
    main()
