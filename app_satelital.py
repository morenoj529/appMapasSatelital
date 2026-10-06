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
import tempfile
import os
import glob

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
    "Arrastra una o varias bandas de tu carpeta Landsat",
    type=["tif", "TIF"],
    accept_multiple_files=True
)

# --- 2. IDENTIFICADOR AUTOMÁTICO DE BANDAS ---
bandas_detectadas = {}
otras_bandas = {}

if archivos_tif:
    for archivo in archivos_tif:
        nombre_upper = archivo.name.upper()
        if "ST_B10" in nombre_upper:
            bandas_detectadas["ST_B10"] = archivo
        elif "SR_B4" in nombre_upper:
            bandas_detectadas["SR_B4"] = archivo
        elif "SR_B5" in nombre_upper:
            bandas_detectadas["SR_B5"] = archivo
        else:
            nombre_corto = archivo.name.split("_")[-1].replace(".TIF", "")
            otras_bandas[f"Banda {nombre_corto} ({archivo.name})"] = archivo

st.sidebar.subheader("Bandas Principales Detectadas:")
col_b1, col_b2, col_b3 = st.sidebar.columns(3)
col_b1.markdown(f"**ST_B10:** {'✅' if 'ST_B10' in bandas_detectadas else '⚪'}")
col_b2.markdown(f"**SR_B4:** {'✅' if 'SR_B4' in bandas_detectadas else '⚪'}")
col_b3.markdown(f"**SR_B5:** {'✅' if 'SR_B5' in bandas_detectadas else '⚪'}")

# --- 3. MAPAS DISPONIBLES ---
mapas_disponibles = []
if "ST_B10" in bandas_detectadas:
    mapas_disponibles.extend([
        "LST - Temperatura Superficial Terrestre (°C)",
        "UTFVI - Urban Thermal Field Variance Index",
        "STD (Z_thermal) - Índice Térmico Estandarizado"
    ])
if "SR_B4" in bandas_detectadas and "SR_B5" in bandas_detectadas:
    mapas_disponibles.append("NDVI - Índice de Vegetación de Diferencia Normalizada")
if all(k in bandas_detectadas for k in ("ST_B10", "SR_B4", "SR_B5")):
    mapas_disponibles.append("UTVI - Índice de Vulnerabilidad Térmica Urbana (LST + NDVI)")
if "SR_B4" in bandas_detectadas and "SR_B5" not in bandas_detectadas:
    mapas_disponibles.append("Reflectancia Superficial Banda 4 (Rojo)")
if "SR_B5" in bandas_detectadas and "SR_B4" not in bandas_detectadas:
    mapas_disponibles.append("Reflectancia Superficial Banda 5 (NIR)")
for nombre_extra in otras_bandas.keys():
    mapas_disponibles.append(nombre_extra)

# --- 4. FUNCIÓN DE RECORTE (EXTRACT BY MASK) ---
def extraer_por_mascara(uploaded_tif, gdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_tif.getbuffer())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        crs_raster = src.crs
        gdf_aligned = gdf.to_crs(crs_raster) if gdf.crs != crs_raster else gdf
        out_img, out_transform = mask(src, gdf_aligned.geometry.values, crop=True, nodata=0)
        bounds = rasterio.transform.array_bounds(out_img.shape[1], out_img.shape[2], out_transform)

    os.remove(tmp_path)
    arr = out_img[0].astype(np.float64)
    arr[arr == 0] = np.nan
    return arr, bounds, gdf_aligned, crs_raster

# --- 5. FUNCIÓN DE RECLASIFICACIÓN EXACTA (ESTILO RECLASSIFY DE ARCMAP) ---
def clasificar_raster(matriz, tipo_mapa, n_clases_lst=5):
    """Convierte valores continuos en clases discretas (0 a N-1) respetando NaN."""
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
            # 9 clases ordinales (1 a 9) según sección 6.1 del procedimiento
            cmap_9 = plt.get_cmap("Spectral_r", 9)
            colores = [ cmap_9(i) for i in range(9) ]
            etiquetas_base = [f"Clase {i+1} ({cortes[i]:.1f} - {cortes[i+1]:.1f} °C)" for i in range(9)]

    elif tipo_mapa.startswith("UTFVI"):
        limites_internos = [-0.15, -0.05, 0.05, 0.15]
        colores = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
        etiquetas_base = [
            "Muy Bajo (≤ -0.15)",
            "Bajo (-0.15 a -0.05)",
            "Moderado (-0.05 a 0.05)",
            "Alto (0.05 a 0.15)",
            "Muy Alto (> 0.15)"
        ]

    elif tipo_mapa.startswith("STD"):
        limites_internos = [-1.5, -0.5, 0.5, 1.5]
        colores = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        etiquetas_base = [
            "Muy Baja (< -1.5 STD)",
            "Baja (-1.5 a -0.5 STD)",
            "Moderada (-0.5 a 0.5 STD)",
            "Alta (0.5 a 1.5 STD)",
            "Muy Alta (≥ 1.5 STD)"
        ]

    elif tipo_mapa.startswith("NDVI"):
        limites_internos = [0.0, 0.10, 0.30, 0.50]
        colores = ["#2b83ba", "#d7191c", "#fdae61", "#a6d96a", "#1a9641"]
        etiquetas_base = [
            "< 0.00 (Agua / Sombra)",
            "0.00 - 0.10 (Nula / Suelo desnudo)",
            "0.10 - 0.30 (Vegetación escasa)",
            "0.30 - 0.50 (Vegetación moderada)",
            "> 0.50 (Vegetación densa)"
        ]

    elif tipo_mapa.startswith("UTVI"):
        limites_internos = [5.5, 9.5, 12.5]
        colores = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]
        etiquetas_base = [
            "Very Low (2 - 5)",
            "Low (6 - 9)",
            "High (10 - 12)",
            "Very High (13 - 18)"
        ]

    else:
        # Cualquier otra banda individual en 5 intervalos iguales
        vmin, vmax = np.nanmin(matriz), np.nanmax(matriz)
        cortes = np.linspace(vmin, vmax, 6)
        limites_internos = cortes[1:-1]
        colores = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
        etiquetas_base = [f"Nivel {i+1} ({cortes[i]:.2f} - {cortes[i+1]:.2f})" for i in range(5)]

    # Asignar clase entera (0, 1, ..., N-1) únicamente a los píxeles dentro de Los Mochis
    indices = np.digitize(matriz[mask_valida], limites_internos)
    matriz_clases[mask_valida] = indices

    # Calcular porcentaje de cobertura urbana por cada clase para la leyenda
    etiquetas_finales = []
    for idx, texto in enumerate(etiquetas_base):
        conteo = np.sum(indices == idx)
        porcentaje = (conteo / total_pixeles) * 100 if total_pixeles > 0 else 0
        etiquetas_finales.append(f"{texto} [{porcentaje:.1f}%]")

    return matriz_clases, colores, etiquetas_finales

# --- 6. FORMATEADOR DE COORDENADAS EN EJES ---
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

# --- 7. SELECCIÓN Y RENDERIZADO ---
if gdf_mascara is not None and mapas_disponibles:
    st.sidebar.header("3. Configuración del Mapa")
    mapa_seleccionado = st.sidebar.selectbox("Elige el mapa a generar:", mapas_disponibles)
    
    modo_visualizacion = st.sidebar.radio(
        "Tipo de representación:",
        ["Continuo (Rampa suave)", "Clasificado por Rangos (Normado PDF)"]
    )

    # Si elige LST clasificado, permitir alternar entre 5 categorías o las 9 clases de Reclassify
    n_clases_lst = 5
    if modo_visualizacion == "Clasificado por Rangos (Normado PDF)" and mapa_seleccionado.startswith("LST"):
        tipo_reclass_lst = st.sidebar.selectbox(
            "Escala de Reclasificación LST:",
            ["5 Categorías (Muy Baja a Muy Alta)", "9 Clases (Escala 1 a 9 de ArcMap)"]
        )
        n_clases_lst = 9 if "9" in tipo_reclass_lst else 5

    formato_coord = st.sidebar.selectbox(
        "Formato de coordenadas en bordes:",
        ["Geográficas (Grados° Minutos')", "Geográficas (Grados Decimales)", "UTM (Metros - Zona 12N)"]
    )

    with st.spinner(f"Generando {mapa_seleccionado}..."):
        if any(m in mapa_seleccionado for m in ["LST", "UTFVI", "STD", "UTVI"]):
            dn_b10, bounds, gdf_utm, crs_actual = extraer_por_mascara(bandas_detectadas["ST_B10"], gdf_mascara)
            lst_c = (dn_b10 * 0.00341802 + 149.0) - 273.15
            lst_mean = float(np.nanmean(lst_c))
            lst_std = float(np.nanstd(lst_c))

        if any(m in mapa_seleccionado for m in ["NDVI", "UTVI"]):
            dn_b4, bounds, gdf_utm, crs_actual = extraer_por_mascara(bandas_detectadas["SR_B4"], gdf_mascara)
            dn_b5, _, _, _ = extraer_por_mascara(bandas_detectadas["SR_B5"], gdf_mascara)
            rho_b4 = (dn_b4 * 0.0000275) - 0.2
            rho_b5 = (dn_b5 * 0.0000275) - 0.2
            denom = rho_b5 + rho_b4
            ndvi = np.clip(np.where(denom != 0, (rho_b5 - rho_b4) / denom, np.nan), -1.0, 1.0)

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
            dn_b4, bounds, gdf_utm, crs_actual = extraer_por_mascara(bandas_detectadas["SR_B4"], gdf_mascara)
            matriz_final, cmap, unidad = (dn_b4 * 0.0000275) - 0.2, "Reds", "Reflectancia"
        elif "Banda 5" in mapa_seleccionado:
            dn_b5, bounds, gdf_utm, crs_actual = extraer_por_mascara(bandas_detectadas["SR_B5"], gdf_mascara)
            matriz_final, cmap, unidad = (dn_b5 * 0.0000275) - 0.2, "Greens", "Reflectancia"
        else:
            arr_extra, bounds, gdf_utm, crs_actual = extraer_por_mascara(otras_bandas[mapa_seleccionado], gdf_mascara)
            matriz_final, cmap, unidad = arr_extra, "viridis", "DN"

        # --- MÉTRICAS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media Urbana (MEAN)", f"{np.nanmean(matriz_final):.3f} {unidad}")
        c2.metric("Desviación Estándar (STD)", f"{np.nanstd(matriz_final):.3f}")
        c3.metric("Mínimo", f"{np.nanmin(matriz_final):.3f}")
        c4.metric("Máximo", f"{np.nanmax(matriz_final):.3f}")

        # --- GRAFICADO ---
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        if modo_visualizacion == "Clasificado por Rangos (Normado PDF)":
            matriz_clases, colores, etiquetas = clasificar_raster(matriz_final, mapa_seleccionado, n_clases_lst)
            cmap_disc = ListedColormap(colores)
            cmap_disc.set_bad(color="white", alpha=0)
            norm_disc = BoundaryNorm(np.arange(-0.5, len(colores) + 0.5, 1), cmap_disc.N)
            
            im = ax.imshow(matriz_clases, cmap=cmap_disc, norm=norm_disc, extent=extent, interpolation="nearest")
            parches = [Patch(facecolor=c, edgecolor="black", label=l) for c, l in zip(colores, etiquetas)]
            ax.legend(
                handles=parches,
                loc="lower right",
                title="Clasificación [Área %]",
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                fontsize=8.5
            )
        else:
            cmap_cont = plt.get_cmap(cmap).copy()
            cmap_cont.set_bad(color="white", alpha=0)
            im = ax.imshow(matriz_final, cmap=cmap_cont, extent=extent)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unidad)

        gdf_utm.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.5)
        ax.set_title(mapa_seleccionado, fontsize=13, fontweight="bold", pad=12)
        formatear_ejes_coordenadas(ax, bounds, crs_actual, formato_coord)

        st.pyplot(fig)

elif gdf_mascara is not None:
    st.info("👈 Sube al menos una banda `.TIF` para ver los mapas disponibles.")