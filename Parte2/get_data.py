import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv(find_dotenv())
HERE_API_KEY = os.getenv("HERE_API_KEY")

# ---------------------------------------------------------------------------
# Lista de las 47 capitales peninsulares de España
# ---------------------------------------------------------------------------
capitales_info: list[dict] = [
    {"ciudad": "Madrid", "provincia": "Madrid"},
    {"ciudad": "Barcelona", "provincia": "Barcelona"},
    {"ciudad": "Valencia", "provincia": "Valencia"},
    {"ciudad": "Sevilla", "provincia": "Sevilla"},
    {"ciudad": "Zaragoza", "provincia": "Zaragoza"},
    {"ciudad": "Málaga", "provincia": "Málaga"},
    {"ciudad": "Murcia", "provincia": "Murcia"},
    {"ciudad": "Bilbao", "provincia": "Bizkaia"},
    {"ciudad": "Alicante", "provincia": "Alicante"},
    {"ciudad": "Valladolid", "provincia": "Valladolid"},
    {"ciudad": "Córdoba", "provincia": "Córdoba"},
    {"ciudad": "Granada", "provincia": "Granada"},
    {"ciudad": "Vitoria-Gasteiz", "provincia": "Álava"},
    {"ciudad": "A Coruña", "provincia": "A Coruña"},
    {"ciudad": "Oviedo", "provincia": "Asturias"},
    {"ciudad": "Pamplona", "provincia": "Navarra"},
    {"ciudad": "Almería", "provincia": "Almería"},
    {"ciudad": "San Sebastián", "provincia": "Gipuzkoa"},
    {"ciudad": "Burgos", "provincia": "Burgos"},
    {"ciudad": "Santander", "provincia": "Cantabria"},
    {"ciudad": "Castellón de la Plana", "provincia": "Castellón"},
    {"ciudad": "Albacete", "provincia": "Albacete"},
    {"ciudad": "Logroño", "provincia": "La Rioja"},
    {"ciudad": "Badajoz", "provincia": "Badajoz"},
    {"ciudad": "Salamanca", "provincia": "Salamanca"},
    {"ciudad": "Huelva", "provincia": "Huelva"},
    {"ciudad": "Lleida", "provincia": "Lleida"},
    {"ciudad": "Tarragona", "provincia": "Tarragona"},
    {"ciudad": "León", "provincia": "León"},
    {"ciudad": "Cádiz", "provincia": "Cádiz"},
    {"ciudad": "Jaén", "provincia": "Jaén"},
    {"ciudad": "Ourense", "provincia": "Ourense"},
    {"ciudad": "Girona", "provincia": "Girona"},
    {"ciudad": "Lugo", "provincia": "Lugo"},
    {"ciudad": "Cáceres", "provincia": "Cáceres"},
    {"ciudad": "Guadalajara", "provincia": "Guadalajara"},
    {"ciudad": "Toledo", "provincia": "Toledo"},
    {"ciudad": "Ávila", "provincia": "Ávila"},
    {"ciudad": "Ciudad Real", "provincia": "Ciudad Real"},
    {"ciudad": "Cuenca", "provincia": "Cuenca"},
    {"ciudad": "Huesca", "provincia": "Huesca"},
    {"ciudad": "Palencia", "provincia": "Palencia"},
    {"ciudad": "Pontevedra", "provincia": "Pontevedra"},
    {"ciudad": "Segovia", "provincia": "Segovia"},
    {"ciudad": "Soria", "provincia": "Soria"},
    {"ciudad": "Teruel", "provincia": "Teruel"},
    {"ciudad": "Zamora", "provincia": "Zamora"}
]


def obtener_peaje_eur(lat1, lon1, lat2, lon2, api_key):
    """Consulta la API de HERE v8 para obtener el coste total del peaje en EUR."""
    url = (
        f"https://router.hereapi.com/v8/routes"
        f"?transportMode=car"
        f"&origin={lat1},{lon1}"
        f"&destination={lat2},{lon2}"
        f"&return=tolls"
        f"&currency=EUR"
        f"&apikey={api_key}"
    )
    
    try:
        res = requests.get(url).json()
        total_eur = 0.0
        sections = res.get('routes', [{}])[0].get('sections', [])
        
        for section in sections:
            if 'tolls' in section:
                for toll in section['tolls']:
                    for fare in toll.get('fares', []):
                        total_eur += float(fare.get('price', {}).get('value', 0.0))
                        
        return round(total_eur, 2)
    except Exception:
        return 0.0


def main():
    # ---------------------------------------------------------------------------
    # PASO 1: Coordenadas (Nominatim API)
    # ---------------------------------------------------------------------------
    file_coords = "coordenadas_47_ciudades.csv"
    if os.path.exists(file_coords):
        print(f"✔ [{file_coords}] ya existe en local. Cargando datos...")
        df_coordenadas = pd.read_csv(file_coords)
    else:
        print("⏳ Consultando Nominatim API para obtener coordenadas de las 47 ciudades...")
        headers = {'User-Agent': 'ProyectoRedesNeuronalesACO/1.0'}
        for c in capitales_info:
            url = f"https://nominatim.openstreetmap.org/search?city={c['ciudad']}&country=Spain&format=json"
            response = requests.get(url, headers=headers).json()
            if response:
                c['latitud'] = float(response[0]['lat'])
                c['longitud'] = float(response[0]['lon'])
            time.sleep(1)

        df_coordenadas = pd.DataFrame(capitales_info)
        df_coordenadas.to_csv(file_coords, index=False)
        print(f"✔ Archivo '{file_coords}' generado con éxito.\n")

    ciudades = df_coordenadas['ciudad'].tolist()
    n = len(ciudades)

    # ---------------------------------------------------------------------------
    # PASO 2 & 3: Matrices de Tiempos (Horas) y Distancias (Km) (OSRM Table API)
    # ---------------------------------------------------------------------------
    file_tiempos = "matriz_tiempos_47x47.csv"
    file_distancias = "matriz_distancias_47x47.csv"

    if os.path.exists(file_tiempos) and os.path.exists(file_distancias):
        print(f"✔ [{file_tiempos}] y [{file_distancias}] ya existen en local. Cargando datos...")
        df_tiempos = pd.read_csv(file_tiempos, index_col=0)
        df_distancias = pd.read_csv(file_distancias, index_col=0)
    else:
        print("⏳ Consultando OSRM Table API para generar matrices de tiempos y distancias...")
        coords_str = ";".join([f"{row['longitud']},{row['latitud']}" for _, row in df_coordenadas.iterrows()])
        
        # Pedimos simultáneamente 'duration' y 'distance'
        url_osrm = f"https://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=duration,distance"
        
        res = requests.get(url_osrm).json()
        if res.get('code') == 'Ok':
            # --- Matriz de Tiempos (convertir segundos a horas) ---
            matriz_horas = np.round(np.array(res['durations']) / 3600.0, 2)
            df_matriz_t = pd.DataFrame(matriz_horas, index=ciudades, columns=ciudades)
            matriz_t_simetrica = (df_matriz_t.values + df_matriz_t.values.T) / 2.0
            df_tiempos = pd.DataFrame(np.round(matriz_t_simetrica, 2), index=ciudades, columns=ciudades)
            df_tiempos.to_csv(file_tiempos)
            print(f"✔ Archivo '{file_tiempos}' (en horas) generado con éxito.")

            # --- Matriz de Distancias (convertir metros a kilómetros) ---
            matriz_km = np.round(np.array(res['distances']) / 1000.0, 1)
            df_matriz_d = pd.DataFrame(matriz_km, index=ciudades, columns=ciudades)
            matriz_d_simetrica = (df_matriz_d.values + df_matriz_d.values.T) / 2.0
            df_distancias = pd.DataFrame(np.round(matriz_d_simetrica, 1), index=ciudades, columns=ciudades)
            df_distancias.to_csv(file_distancias)
            print(f"✔ Archivo '{file_distancias}' (en km) generado con éxito.\n")
        else:
            print("❌ Error al consultar la API de OSRM.")

    # ---------------------------------------------------------------------------
    # PASO 4: Matriz de Peajes (HERE Routing API)
    # ---------------------------------------------------------------------------
    file_peajes = "matriz_peajes_47x47.csv"
    if os.path.exists(file_peajes):
        print(f"✔ [{file_peajes}] ya existe en local. Cargando datos...")
        df_peajes = pd.read_csv(file_peajes, index_col=0)
    else:
        if not HERE_API_KEY or HERE_API_KEY == "tu_clave_secreta_aqui_sin_comillas":
            print("⚠️ No se encontró una HERE_API_KEY válida en el archivo .env.")
            print("   Configura tu API Key en el .env antes de generar la matriz de peajes.")
            return

        total_peticiones = (n * (n - 1)) // 2  # 1,081 peticiones para 47 ciudades
        print(f"⏳ Consultando HERE Routing API para generar matriz de peajes ({total_peticiones} trayectos)...")
        matriz_peajes = np.zeros((n, n))

        # Recorrer la mitad superior por simetría A -> B = B -> A
        for i in range(n):
            print(f"   --> Procesando trayectos desde {ciudades[i]} ({i + 1}/{n})...")
            for j in range(i + 1, n):
                lat_o, lon_o = df_coordenadas.loc[i, 'latitud'], df_coordenadas.loc[i, 'longitud']
                lat_d, lon_d = df_coordenadas.loc[j, 'latitud'], df_coordenadas.loc[j, 'longitud']
                
                costo = obtener_peaje_eur(lat_o, lon_o, lat_d, lon_d, HERE_API_KEY)
                
                matriz_peajes[i][j] = costo
                matriz_peajes[j][i] = costo

        df_peajes = pd.DataFrame(matriz_peajes, index=ciudades, columns=ciudades)
        df_peajes.to_csv(file_peajes)
        print(f"✔ Archivo '{file_peajes}' generado con éxito.\n")

    print("\n🚀 Todos los datasets (47x47) están listos para el algoritmo ACO.")


if __name__ == "__main__":
    main()