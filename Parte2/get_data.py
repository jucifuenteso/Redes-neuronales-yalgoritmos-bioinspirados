import requests
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------------------------
# 1. Lista de las 37 capitales peninsulares de España
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
    {"ciudad": "Toledo", "provincia": "Toledo"}
]

def main():
    print("1/2 Obtención de coordenadas mediante Nominatim API...")
    headers = {'User-Agent': 'ProyectoRedesNeuronalesACO/1.0'}

    for c in capitales_info:
        url = f"https://nominatim.openstreetmap.org/search?city={c['ciudad']}&country=Spain&format=json"
        response = requests.get(url, headers=headers).json()
        if response:
            c['latitud'] = float(response[0]['lat'])
            c['longitud'] = float(response[0]['lon'])
        time.sleep(1) # Respetar límite de uso de 1 solicitud/segundo

    # Guardar Tabla 1: Coordenadas
    df_coordenadas = pd.DataFrame(capitales_info)
    df_coordenadas.to_csv("coordenadas_37_ciudades.csv", index=False)
    print("✔ 'coordenadas_37_ciudades.csv' generado con éxito.\n")

    # ---------------------------------------------------------------------------
    # 2. Obtener la Matriz de Tiempos 37x37 mediante OSRM Table API
    # ---------------------------------------------------------------------------
    print("2/2 Generando la matriz de tiempos 37x37 con OSRM Table API...")
    coords_str = ";".join([f"{row['longitud']},{row['latitud']}" for _, row in df_coordenadas.iterrows()])
    url_osrm = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=duration"
    
    res = requests.get(url_osrm).json()

    if res['code'] == 'Ok':
        # La API devuelve los tiempos en segundos -> Convertir a minutos
        matriz_minutos = np.round(np.array(res['durations']) / 60.0, 1)
        ciudades = df_coordenadas['ciudad'].tolist()
        df_matriz = pd.DataFrame(matriz_minutos, index=ciudades, columns=ciudades)
        
        # Simetrizar matriz para garantizar T(A,B) == T(B,A)
        matriz_simetrica = (df_matriz.values + df_matriz.values.T) / 2.0
        df_matriz_simetrica = pd.DataFrame(np.round(matriz_simetrica, 1), index=ciudades, columns=ciudades)
        
        # Guardar Tabla 2: Matriz 37x37
        df_matriz_simetrica.to_csv("matriz_tiempos_37x37.csv")
        print("✔ 'matriz_tiempos_37x37.csv' generado con éxito.\n")
        print("Proceso finalizado. Archivos generados correctamente.")
    else:
        print("❌ Ocurrió un error al consultar la API de OSRM.")

if __name__ == "__main__":
    main()