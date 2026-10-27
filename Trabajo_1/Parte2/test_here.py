import os
import requests
from dotenv import load_dotenv, find_dotenv

# 1. Cargar la API Key desde el .env
load_dotenv(find_dotenv())
API_KEY = os.getenv("HERE_API_KEY")

if not API_KEY or API_KEY == "tu_clave_secreta_aqui_sin_comillas":
    print(" Error: No se encontró la HERE_API_KEY en el archivo .env")
    exit()

print(f" Probando con API Key: {API_KEY[:6]}...{API_KEY[-4:]}\n")

# 2. Definir dos rutas de prueba (Una con peaje real y otra gratuita)
pruebas = [
    {
        "nombre": "Prueba 1: Bilbao -> San Sebastián (Autopista AP-8 con peaje)",
        "origen": "43.2630,-2.9350",
        "destino": "43.3183,-1.9812"
    },
    {
        "nombre": "Prueba 2: Madrid -> Valencia (Autovía A-3 gratuita)",
        "origen": "40.4168,-3.7038",
        "destino": "39.4699,-0.3763"
    }
]

# 3. Realizar las peticiones
for p in pruebas:
    print(f"==================================================")
    print(f" {p['nombre']}")
    print(f"==================================================")
    
    url = (
        f"https://router.hereapi.com/v8/routes"
        f"?transportMode=car"
        f"&origin={p['origen']}"
        f"&destination={p['destino']}"
        f"&return=tolls"
        f"&currency=EUR"
        f"&apikey={API_KEY}"
    )
    
    res = requests.get(url)
    print(f"HTTP Status Code: {res.status_code}")
    
    if res.status_code != 200:
        print(f" Error en la API:")
        print(res.text)
        continue
        
    data = res.json()
    sections = data.get("routes", [{}])[0].get("sections", [])
    
    total_eur = 0.0
    tramos_peaje = 0
    
    for section in sections:
        if "tolls" in section:
            for toll in section["tolls"]:
                tramos_peaje += 1
                nombre_peaje = toll.get("name", "TRAMO DE PEAJE")
                for fare in toll.get("fares", []):
                    precio = float(fare.get("price", {}).get("value", 0.0))
                    total_eur += precio
                    print(f"  📌 {nombre_peaje}: {precio:.2f} EUR")
                    
    if tramos_peaje == 0:
        print("  ℹ Ruta 100% gratuita (sin peajes).")
        
    print(f"\n TOTAL RUTAS: {total_eur:.2f} EUR\n")