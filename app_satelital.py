import streamlit as st
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
import contextily as cx
import tempfile
import os
import glob
from datetime import datetime
import warnings

# Silenciar advertencias matemáticas al promediar celdas NaN
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="Mapas Satelitales - Los Mochis", layout="wide")
st.title("🛰️ Generador de Mapas Satelitales (Landsat 8/9)")

# --- 1. CARGA AUTOMÁTICA DEL SHAPEFILE LOCAL ---
@st.cache_data
def cargar_shapefile_local():
    archivos_shp = glob.glob("*.shp") + glob.glob("**/*.shp", recursive=True)
    if archivos_shp:
        return gpd.read_file(archivos_shp[0]), archivos_shp[0]
    return None, None

gdf_mascara, nombre_shp = cargar_shapefile_local()

# --- BARRA LATERAL ---
st.sidebar.header("1. Límite Urbano (Máscara)")
if gdf_mascara is not None:
    st.sidebar.success(f"Shapefile activo: `{os.path.basename(nombre_shp)}`")
else:
    st.sidebar.warning("No se detectó un .shp en la carpeta.")

st.sidebar.header("2. Subir Bandas Satelitales (.TIF)")
archivos_tif = st.sidebar.file_uploader(
    "Arrastra las bandas (de un solo mes o de varios meses para promediar)",
    type=["tif", "TIF"],
    accept_multiple_files=True
)

# --- 2. EXTRACCIÓN DE FECHAS Y AGRUPAMIENTO TEMPORAL ---
meses_es = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}

def obtener_info_fecha(filename):
    partes = filename.split("_")
    if len(partes) > 3 and partes[3].isdigit() and len(partes[3]) == 8:
        try:
            dt = datetime.strptime(partes[3], "%Y%m%d")
            return dt, f"{meses_es[dt.month]} {dt.year}"
        except: pass
    return datetime(1900, 1, 1), "Sin Fecha"

coleccion_temporal = {}
orden_fechas = {}

if archivos_tif:
    for archivo in archivos_tif:
        dt, label_mes = obtener_info_fecha(archivo.name)
        if label_mes not in coleccion_temporal:
            coleccion_temporal[label_mes] = {"ST_B10": [], "SR_B4": [], "SR_B5": [], "Otras": []}
            orden_fechas[label_mes] = dt

        nombre_upper = archivo.name.upper()
        if "ST_B10" in nombre_upper: coleccion_temporal[label_mes]["ST_B10"].append(archivo)
        elif "SR_B4" in nombre_upper: coleccion_temporal[label_mes]["SR_B4"].append(archivo)
        elif "SR_B5" in nombre_upper: coleccion_temporal[label_mes]["SR_B5"].append(archivo)
        else: coleccion_temporal[label_mes]["Otras"].append(archivo)

meses_disponibles = sorted(coleccion_temporal.keys(), key=lambda x: orden_fechas[x])

# --- 3. SELECCIÓN DE PERÍODO DE ANÁLISIS ---
meses_seleccionados = []
if len(meses_disponibles) == 1:
    st.sidebar.header("3. Período de Análisis")
    st.sidebar.info(f"📅 Fecha detectada: **{meses_disponibles[0]}**")
    meses_seleccionados = meses_disponibles
elif len(meses_disponibles) > 1:
    st.sidebar.header("3. Período de Análisis (Promedios)")
    meses_seleccionados = st.sidebar.multiselect(
        "Selecciona los meses a analizar (se promediarán espacialmente):",
        meses_disponibles,
        default=meses_disponibles
    )

# --- 4. FILTRADO DINÁMICO DE MAPAS ---
mapas_disponibles = []
if meses_seleccionados:
    tiene_b10 = any(len(coleccion_temporal[m]["ST_B10"]) > 0 for m in meses_seleccionados)
    tiene_b4 = any(len(coleccion_temporal[m]["SR_B4"]) > 0 for m in meses_seleccionados)
    tiene_b5 = any(len(coleccion_temporal[m]["SR_B5"]) > 0 for m in meses_seleccionados)

    if tiene_b10: mapas_disponibles.extend(["LST - Temperatura Superficial Terrestre (°C)", "UTFVI - Urban Thermal Field Variance Index", "STD (Z_thermal) - Índice Térmico Estandarizado"])
    if tiene_b4 and tiene_b5: mapas_disponibles.append("NDVI - Índice de Vegetación de Diferencia Normalizada")
    if tiene_b10 and tiene_b4 and tiene_b5: mapas_disponibles.append("UTVI - Índice de Vulnerabilidad Térmica Urbana (LST + NDVI)")
    if tiene_b4 and not tiene_b5: mapas_disponibles.append("Reflectancia Superficial Banda 4 (Rojo)")
    if tiene_b5 and not tiene_b4: mapas_disponibles.append("Reflectancia Superficial Banda 5 (NIR)")
    
    # Agregar las bandas "Otras" al menú
    nombres_otras = set()
    for m in meses_seleccionados:
        for f in coleccion_temporal[m]["Otras"]:
            nombre_corto = f.name.split("_")[-1].replace(".TIF", "")
            nombres_otras.add(f"Banda {nombre_corto} ({f.name})")
    mapas_disponibles.extend(list(nombres_otras))

# --- 5. FUNCIONES CORE ---
def extraer_por_mascara(uploaded_tif, gdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_tif.getbuffer())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        crs_str = src.crs.to_string() if src.crs else "EPSG:32612"
        gdf_aligned = gdf.to_crs(crs_str) if str(gdf.crs) != crs_str else gdf
        out_img, out_transform = mask(src, gdf_aligned.geometry.values, crop=True, nodata=0)
        bounds = rasterio.transform.array_bounds(out_img.shape[1], out_img.shape[2], out_transform)

    os.remove(tmp_path)
    arr = out_img[0].astype(np.float64)
    arr[arr == 0] = np.nan
    return arr, bounds, gdf_aligned, crs_str

def procesar_promedio(archivos_list, factor, offset, gdf):
    matrices = []
    bounds_ref, crs_ref, gdf_utm_ref = None, None, None
    for f in archivos_list:
        arr, bnds, gdf_utm, crs = extraer_por_mascara(f, gdf)
        matrices.append((arr * factor) + offset)
        if bounds_ref is None:
            bounds_ref, crs_ref, gdf_utm_ref = bnds, crs, gdf_utm
    if matrices:
        promedio = np.nanmean(matrices, axis=0)
        return promedio, bounds_ref, crs_ref, gdf_utm_ref
    return None, None, None, None

def clasificar_raster(matriz, tipo_mapa, n_clases_lst=5):
    mask_valida = ~np.isnan(matriz)
    total_pixeles = np.sum(mask_valida)
    matriz_clases = np.full(matriz.shape, np.nan)

    if tipo_mapa.startswith("LST"):
        vmin, vmax = np.nanmin(matriz), np.nanmax(matriz)
        cortes = np.linspace(vmin, vmax, n_clases_lst + 1)
        limites_internos = cortes[1:-1]
        if n_clases_lst == 5:
            nombres = ["Muy Baja", "Baja", "Moderada", "Alta", "Muy Alta"]
            colores = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
            etiquetas_base = [f"{nombres[i]} ({cortes[i]:.1f} - {cortes[i+1]:.1f} °C)" for i in range(5)]
        else:
            cmap_9 = plt.get_cmap("Spectral_r", 9)
            colores = [ cmap_9(i) for i in range(9) ]
            etiquetas_base = [f"Clase {i+1} ({cortes[i]:.1f} - {cortes[i+1]:.1f} °C)" for i in range(9)]
    elif tipo_mapa.startswith("UTFVI"):
        limites_internos = [-0.15, -0.05, 0.05, 0.15]
        colores = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
        etiquetas_base = ["Muy Bajo (≤ -0.15)", "Bajo (-0.15 a -0.05)", "Moderado (-0.05 a 0.05)", "Alto (0.05 a 0.15)", "Muy Alto (> 0.15)"]
    elif tipo_mapa.startswith("STD"):
        limites_internos = [-1.5, -0.5, 0.5, 1.5]
        colores = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        etiquetas_base = ["Muy Baja (< -1.5)", "Baja (-1.5 a -0.5)", "Moderada (-0.5 a 0.5)", "Alta (0.5 a 1.5)", "Muy Alta (≥ 1.5)"]
    elif tipo_mapa.startswith("NDVI"):
        limites_internos = [0.0, 0.10, 0.30, 0.50]
        colores = ["#2b83ba", "#d7191c", "#fdae61", "#a6d96a", "#1a9641"]
        etiquetas_base = ["< 0.00 (Agua/Sombra)", "0.00 - 0.10 (Nula/Suelo)", "0.10 - 0.30 (Veg. escasa)", "0.30 - 0.50 (Veg. moderada)", "> 0.50 (Veg. densa)"]
    elif tipo_mapa.startswith("UTVI"):
        limites_internos = [5.5, 9.5, 12.5]
        colores = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]
        etiquetas_base = ["Very Low (2 - 5)", "Low (6 - 9)", "High (10 - 12)", "Very High (13 - 18)"]
    else:
        vmin, vmax = np.nanmin(matriz), np.nanmax(matriz)
        cortes = np.linspace(vmin, vmax, 6)
        limites_internos = cortes[1:-1]
        colores = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
        etiquetas_base = [f"Nivel {i+1} ({cortes[i]:.2f} - {cortes[i+1]:.2f})" for i in range(5)]

    indices = np.digitize(matriz[mask_valida], limites_internos)
    matriz_clases[mask_valida] = indices

    etiquetas_finales = []
    for idx, texto in enumerate(etiquetas_base):
        conteo = np.sum(indices == idx)
        porcentaje = (conteo / total_pixeles) * 100 if total_pixeles > 0 else 0
        etiquetas_finales.append(f"{texto} [{porcentaje:.1f}%]")

    return matriz_clases, colores, etiquetas_finales

def formatear_ejes_coordenadas(ax, bounds, crs_origen, tipo_coord):
    ax.ticklabel_format(style="plain", useOffset=False)
    if tipo_coord == "UTM (Metros - Zona 12N)":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,} m E"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y):,} m N"))
    else:
        transformer = Transformer.from_crs(crs_origen, "EPSG:4326", always_xy=True)
        y_centro = (bounds[1] + bounds[3]) / 2.0
        x_centro = (bounds[0] + bounds[2]) / 2.0

        def fmt_lon(x, pos):
            lon, _ = transformer.transform(x, y_centro)
            if tipo_coord == "Geográficas (Grados° Minutos')":
                deg = int(abs(lon))
                minutes = (abs(lon) - deg) * 60
                hemi = "W" if lon < 0 else "E"
                return f"{deg}°{minutes:04.1f}'{hemi}"
            return f"{lon:.3f}°"

        def fmt_lat(y, pos):
            _, lat = transformer.transform(x_centro, y)
            if tipo_coord == "Geográficas (Grados° Minutos')":
                deg = int(abs(lat))
                minutes = (abs(lat) - deg) * 60
                hemi = "N" if lat >= 0 else "S"
                return f"{deg}°{minutes:04.1f}'{hemi}"
            return f"{lat:.3f}°"

        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")

# --- 6. RENDERIZADO PRINCIPAL ---
if gdf_mascara is not None and mapas_disponibles and meses_seleccionados:
    st.sidebar.header("4. Configuración del Mapa")
    mapa_seleccionado = st.sidebar.selectbox("Elige el mapa a generar:", mapas_disponibles)
    
    modo_visualizacion = st.sidebar.radio("Tipo de representación:", ["Continuo (Rampa suave)", "Clasificado por Rangos (Normado PDF)"])

    n_clases_lst = 5
    if modo_visualizacion == "Clasificado por Rangos (Normado PDF)" and mapa_seleccionado.startswith("LST"):
        tipo_reclass_lst = st.sidebar.selectbox("Escala LST:", ["5 Categorías (Muy Baja a Muy Alta)", "9 Clases (Escala 1 a 9 de ArcMap)"])
        n_clases_lst = 9 if "9" in tipo_reclass_lst else 5

    formato_coord = st.sidebar.selectbox("Formato de coordenadas en bordes:", ["Geográficas (Grados° Minutos')", "Geográficas (Grados Decimales)", "UTM (Metros - Zona 12N)"])

    st.sidebar.header("5. Capa Base y Diseño")
    mapa_base = st.sidebar.selectbox("Fondo de mapa (Contextily):", ["Esri Topográfico (Calles)", "Esri Satelital", "CartoDB Positron (Claro)", "Ninguno"])
    transparencia = st.sidebar.slider("Transparencia del raster generado (Alpha):", 0.1, 1.0, 0.85)

    texto_accion = "Procesando escena de" if len(meses_seleccionados) == 1 else "Calculando promedio de"
    
    with st.spinner(f"{texto_accion} {len(meses_seleccionados)} mes(es) y generando {mapa_seleccionado}..."):
        archivos_b10 = [f for m in meses_seleccionados for f in coleccion_temporal[m]["ST_B10"]]
        archivos_b4 = [f for m in meses_seleccionados for f in coleccion_temporal[m]["SR_B4"]]
        archivos_b5 = [f for m in meses_seleccionados for f in coleccion_temporal[m]["SR_B5"]]

        bounds, crs_actual, gdf_utm = None, None, None

        # Bloque de pre-cálculo condicional
        if any(m in mapa_seleccionado for m in ["LST", "UTFVI", "STD", "UTVI"]):
            lst_c, bounds, crs_actual, gdf_utm = procesar_promedio(archivos_b10, 0.00341802, 149.0 - 273.15, gdf_mascara)
            lst_mean = float(np.nanmean(lst_c))
            lst_std = float(np.nanstd(lst_c))

        if any(m in mapa_seleccionado for m in ["NDVI", "UTVI", "Banda 4"]):
            if archivos_b4:
                rho_b4, b_b4, c_b4, g_b4 = procesar_promedio(archivos_b4, 0.0000275, -0.2, gdf_mascara)
                if bounds is None: bounds, crs_actual, gdf_utm = b_b4, c_b4, g_b4

        if any(m in mapa_seleccionado for m in ["NDVI", "UTVI", "Banda 5"]):
            if archivos_b5:
                rho_b5, b_b5, c_b5, g_b5 = procesar_promedio(archivos_b5, 0.0000275, -0.2, gdf_mascara)
                if bounds is None: bounds, crs_actual, gdf_utm = b_b5, c_b5, g_b5

        if any(m in mapa_seleccionado for m in ["NDVI", "UTVI"]):
            denom = rho_b5 + rho_b4
            ndvi = np.clip(np.where(denom != 0, (rho_b5 - rho_b4) / denom, np.nan), -1.0, 1.0)

        # Asignación final a las variables de dibujado (ESTO PREVIENE EL ERROR)
        if mapa_seleccionado.startswith("LST"):
            matriz_final, cmap, unidad = lst_c, "Spectral_r", "°C"
        elif mapa_seleccionado.startswith("UTFVI"):
            matriz_final, cmap, unidad = (lst_c - lst_mean) / lst_mean, "RdYlBu_r", "UTFVI"
        elif mapa_seleccionado.startswith("STD"):
            matriz_final, cmap, unidad = (lst_c - lst_mean) / lst_std, "coolwarm", "Z-Score (STD)"
        elif mapa_seleccionado.startswith("NDVI"):
            matriz_final, cmap, unidad = ndvi, "RdYlGn", "NDVI"
        elif mapa_seleccionado.startswith("UTVI"):
            lst_edges = np.linspace(np.nanmin(lst_c), np.nanmax(lst_c), 10)
            ndvi_edges = np.linspace(np.nanmin(ndvi), np.nanmax(ndvi), 10)
            lst_reclass = np.clip(np.digitize(lst_c, lst_edges[:-1]), 1, 9).astype(float)
            ndvi_reclass_inv = 10.0 - np.clip(np.digitize(ndvi, ndvi_edges[:-1]), 1, 9).astype(float)
            lst_reclass[np.isnan(lst_c)] = np.nan
            ndvi_reclass_inv[np.isnan(ndvi)] = np.nan
            matriz_final, cmap, unidad = lst_reclass + ndvi_reclass_inv, "YlOrRd", "UTVI (2-18)"
        elif "Banda 4" in mapa_seleccionado:
            matriz_final, cmap, unidad = rho_b4, "Reds", "Reflectancia"
        elif "Banda 5" in mapa_seleccionado:
            matriz_final, cmap, unidad = rho_b5, "Greens", "Reflectancia"
        else:
            # Para cuando seleccionas cualquier otra banda TIF que subiste
            archivos_extra = [f for m in meses_seleccionados for f in coleccion_temporal[m]["Otras"] if f.name in mapa_seleccionado]
            arr_extra, b_ex, c_ex, g_ex = procesar_promedio(archivos_extra, 1.0, 0.0, gdf_mascara)
            matriz_final, cmap, unidad = arr_extra, "viridis", "DN"
            if bounds is None: bounds, crs_actual, gdf_utm = b_ex, c_ex, g_ex

        # --- MÉTRICAS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media Urbana (MEAN)", f"{np.nanmean(matriz_final):.3f} {unidad}")
        c2.metric("Desviación Estándar (STD)", f"{np.nanstd(matriz_final):.3f}")
        c3.metric("Mínimo", f"{np.nanmin(matriz_final):.3f}")
        c4.metric("Máximo", f"{np.nanmax(matriz_final):.3f}")

        # --- GRAFICADO ---
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        matriz_masked = np.ma.masked_invalid(matriz_final)

        if modo_visualizacion == "Clasificado por Rangos (Normado PDF)":
            matriz_clases, colores, etiquetas = clasificar_raster(matriz_final, mapa_seleccionado, n_clases_lst)
            clases_masked = np.ma.masked_invalid(matriz_clases)
            cmap_disc = ListedColormap(colores)
            cmap_disc.set_bad(color="none", alpha=0)
            norm_disc = BoundaryNorm(np.arange(-0.5, len(colores) + 0.5, 1), cmap_disc.N)
            
            im = ax.imshow(clases_masked, cmap=cmap_disc, norm=norm_disc, extent=extent, interpolation="nearest", alpha=transparencia, zorder=2)
            parches = [Patch(facecolor=c, edgecolor="black", label=l) for c, l in zip(colores, etiquetas)]
            ax.legend(handles=parches, loc="lower right", title="Clasificación [Área %]", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
        else:
            cmap_cont = plt.get_cmap(cmap).copy()
            cmap_cont.set_bad(color="none", alpha=0)
            im = ax.imshow(matriz_masked, cmap=cmap_cont, extent=extent, vmin=np.nanmin(matriz_final), vmax=np.nanmax(matriz_final), alpha=transparencia, zorder=2)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unidad)

        gdf_utm.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.8, zorder=3)
        
        if mapa_base != "Ninguno":
            proveedor = cx.providers.Esri.WorldTopoMap if mapa_base == "Esri Topográfico (Calles)" else (cx.providers.Esri.WorldImagery if mapa_base == "Esri Satelital" else cx.providers.CartoDB.Positron)
            try: cx.add_basemap(ax, crs=crs_actual, source=proveedor, attribution=False, zorder=1)
            except: st.warning("No se pudo cargar la capa base de internet.")

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        # --- TÍTULO DINÁMICO DE RANGO DE FECHAS ---
        if len(meses_seleccionados) == 1:
            texto_fecha = meses_seleccionados[0]
        else:
            seleccion_ordenada = sorted(meses_seleccionados, key=lambda x: orden_fechas[x])
            texto_fecha = f"Promedio ({seleccion_ordenada[0]} - {seleccion_ordenada[-1]})"
            
        ax.set_title(f"{mapa_seleccionado}\n{texto_fecha}", fontsize=14, fontweight="bold", pad=12)
        formatear_ejes_coordenadas(ax, bounds, crs_actual, formato_coord)

        st.pyplot(fig)

elif gdf_mascara is not None:
    st.info("👈 Sube las imágenes satelitales .TIF para comenzar el análisis.")
