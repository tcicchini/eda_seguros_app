# Análisis de Cartera — Seguros de Vida

Aplicación Streamlit para explorar de forma interactiva la cartera de seguros de vida
analizada en `notebooks/eda_seguros.ipynb`. Traduce a un tablero navegable las decisiones
tomadas en `outputs/reporte_figuras.md` (qué figuras incluir, mejorar o descartar) y los
hallazgos nuevos de ese reporte, pensada para una audiencia **no técnica, de perfil
empresarial**.

La app está organizada en un panel de KPIs siempre visible y 5 secciones navegables desde
la barra lateral:

1. **Composición de la cartera** — perfil demográfico y distribución por unidad de negocio.
2. **Riesgo y siniestralidad** — incidencia por unidad, exposición monetaria vs. riesgo,
   incidencia por segmento de cobertura, días hasta el siniestro, suma asegurada por estatus.
3. **Perfil etario y cobertura** — incidencia y suma asegurada promedio por rango etario de negocio.
4. **Análisis temporal** — fecha de alta de póliza, con una advertencia sobre un patrón de
   carga de datos detectado (ver Limitaciones).
5. **Demora de denuncia** — qué tan rápido se reportan los siniestros, por unidad de negocio.

Todas las figuras son interactivas (Plotly) y responden a filtros por unidad de negocio y/o
género, definidos una sola vez por sección (no por figura individual).

## Datos: dónde deben estar ubicados

**Esta carpeta no incluye datos.** La app lee el archivo `Super_reducido_Prod1.xlsx`
directamente desde una carpeta `datos/` externa a `seguros_eda_app/`. Se probó con el
siguiente layout (carpetas hermanas dentro del home del usuario, tal como se pidió):

```
~/seguros_eda/
├── datos/
│   └── Super_reducido_Prod1.xlsx
├── notebooks/
│   └── eda_seguros.ipynb
└── outputs/
    └── reporte_figuras.md

~/seguros_eda_app/          <- esta carpeta
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

La app busca los datos, en este orden:

1. Variable de entorno `SEGUROS_EDA_DATA_DIR` (si está definida, apunta directo a la
   carpeta `datos/`).
2. `../seguros_eda/datos/` relativo a esta carpeta (el layout de arriba).
3. `../datos/` relativo a esta carpeta (por si `seguros_eda_app/` se coloca **dentro** de la
   carpeta del proyecto, junto a `datos/`, `notebooks/` y `outputs/`).

Si ninguna de las dos rutas relativas existe y no se definió la variable de entorno, la app
muestra un mensaje de error indicando la ruta exacta que buscó.

Para forzar una ubicación distinta:

```bash
export SEGUROS_EDA_DATA_DIR=/ruta/a/tu/carpeta/datos
streamlit run app.py
```

## Cómo correrla localmente

```bash
conda activate eda_seguros
streamlit run app.py
```

La primera carga puede tardar **1 a 3 minutos**: el archivo Excel tiene ~873.000 filas y se
limpia en memoria siguiendo la misma lógica del notebook. Streamlit cachea ese resultado
(`@st.cache_data`), así que las cargas siguientes dentro de la misma sesión son instantáneas;
cambiar de sección o de filtro no vuelve a leer el Excel.

## Cómo desplegarla en Streamlit Community Cloud

1. Subí esta carpeta (`seguros_eda_app/`) a un repositorio de GitHub. **No subas la carpeta
   `datos/` ni el archivo `.xlsx`** — la base de datos de siniestros no debería publicarse en
   un repositorio, y esta app está pensada para leerla desde afuera del repo.
2. Para que la app tenga datos en la nube, hay dos opciones:
   - Subir el Excel a un storage privado (ej. un bucket S3/GCS o un Google Drive
     compartido) y adaptar `load_clean_data()` para descargarlo al iniciar, o
   - Usar **Streamlit secrets** para inyectar una URL de descarga segura y guardar el
     archivo en un directorio temporal antes de llamar a `pd.read_excel`.
   Esta versión de la app asume que los datos ya están accesibles en el sistema de archivos
   (uso previsto: ejecución local o en un servidor propio con la carpeta `datos/` montada).
3. En [share.streamlit.io](https://share.streamlit.io):
   - Conectá el repositorio de GitHub.
   - Seleccioná `app.py` como *entry point*.
   - Cargá `requirements.txt` (se detecta automáticamente si está en la raíz del repo).
   - Si corresponde, configurá `SEGUROS_EDA_DATA_DIR` u otras variables en
     **App settings → Secrets**.

## Limitaciones conocidas del análisis

- **No hay monto pagado por siniestro.** La base sólo tiene el flag binario `Siniestro`
  (ocurrió / no ocurrió), no un monto indemnizado. Por lo tanto, el KPI "Monto total
  expuesto" es el **capital máximo potencial** de la cartera, no un costo esperado ni una
  pérdida incurrida — no debe interpretarse como un *loss ratio* monetario real.
- **El patrón temporal de altas de póliza no es confiable para análisis de estacionalidad.**
  La Sección 4 detectó que ~65% de los registros tiene como fecha de alta el día 1 de un
  mes, y una sola fecha concentra más del 13% de toda la cartera — un patrón típico de
  cargas masivas / migraciones de sistema, no de altas comerciales orgánicas. La app muestra
  esta advertencia explícitamente en esa sección.
- **Los rangos etarios de negocio (`<30`, `30-40`, `40-50`, `50-60`, `60+`) usan `Edad_ingreso`**,
  no la edad al momento del siniestro. Esto es intencional (perfila el riesgo según el perfil
  con el que el asegurado ingresó a la cartera), pero conviene tenerlo presente al leer la
  Sección 3.
- **Los cuartiles de Suma Asegurada (Sección 2) se calculan una única vez sobre toda la
  cartera** y no se recalculan al filtrar por unidad de negocio, para que los segmentos
  Q1–Q4 signifiquen lo mismo en cualquier selección. Esto significa que, al filtrar a pocas
  unidades, algún segmento puede quedar con muy pocos registros (o vacío).
- **Existen anomalías de datos ya documentadas en el notebook** (fechas de nacimiento
  centinela, 2 registros con suma asegurada negativa, 58 casos con `dias_hasta_sin`
  negativo) que se tratan igual que en el notebook original (se excluyen o se marcan, según
  el caso) pero no se vuelven a explicar en el detalle en esta app — para el detalle completo,
  ver `outputs/reporte_figuras.md` y la Sección 1 de `notebooks/eda_seguros.ipynb`.
- **Ausencia de variables de producto, provincia/localidad de calidad suficiente y ocupación**
  del asegurado limita la profundidad de segmentación de riesgo posible en esta versión de
  la app (misma limitación señalada en el notebook).
