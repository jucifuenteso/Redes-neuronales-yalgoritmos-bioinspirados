"""
Visualizacion de ciudades y rutas, en el mismo estilo de ant_colony.ipynb:

    plt.scatter(Ciudades[:,0], Ciudades[:,1])
    plt.plot(Ciudades[optimizer.best_path,0], Ciudades[optimizer.best_path,1], '-r')
    plt.grid()
    plt.title("Ciudades a visitar")
    plt.show()

Aqui se usan latitud/longitud (en vez de coordenadas sinteticas) y se
agrega una animacion simple para ver como evoluciona la mejor ruta.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def graficar_ruta(coords, ruta, costo=None, titulo="Ciudades a visitar"):
    """coords: array (n, 2) con columnas [latitud, longitud]."""
    lat, lon = coords[:, 0], coords[:, 1]
    ruta_cerrada = np.append(ruta, ruta[0])

    plt.scatter(lon, lat)
    plt.plot(lon[ruta_cerrada], lat[ruta_cerrada], "-r")
    plt.grid()
    if costo is not None:
        titulo = f"{titulo} (costo = {costo:,.1f} EUR)"
    plt.title(titulo)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")


def animar_evolucion(coords, historial_rutas, historial_costos, out_path, cada_n=1, fps=8):
    """Genera un GIF mostrando como cambia la mejor ruta a lo largo de
    las iteraciones (ACO) o generaciones (GA)."""
    rutas = historial_rutas[::cada_n]
    costos = historial_costos[::cada_n]

    fig = plt.figure(figsize=(7, 8))

    def actualizar(i):
        plt.clf()
        graficar_ruta(coords, rutas[i], costo=costos[i], titulo=f"Evolucion - paso {i * cada_n}")
        return []

    anim = FuncAnimation(fig, actualizar, frames=len(rutas))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def graficar_convergencia(historiales, etiquetas, out_path=None):
    """historiales: lista de listas (una por algoritmo) con el mejor
    costo encontrado hasta cada iteracion/generacion."""
    for hist, etq in zip(historiales, etiquetas):
        plt.plot(hist, label=etq)
    plt.xlabel("Iteracion / generacion")
    plt.ylabel("Mejor costo encontrado (EUR)")
    plt.title("Convergencia de los algoritmos")
    plt.legend()
    plt.grid()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")