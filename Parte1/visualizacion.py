"""
Figuras y animaciones de la Parte 1.

  - graficar_convergencia:    mediana de f - f* contra evaluaciones equivalentes.
  - graficar_valores_finales: valor final de cada una de las corridas.
  - animar_optimizacion:      GIF con el descenso por gradiente y los tres
                              heuristicos avanzando al mismo ritmo de
                              evaluaciones equivalentes (funciones en 2D).

En todas las figuras cada metodo conserva el mismo color.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

COLORES = {
    "GD paso fijo": "#2a78d6",
    "GD backtracking": "#eb6834",
    "GA": "#1baf7a",
    "PSO": "#eda100",
    "DE": "#e87ba4",
}
# Los dos descensos por gradiente van con linea discontinua: asi no se
# distinguen solo por el color.
ESTILOS = {"GD paso fijo": (0, (5, 2)), "GD backtracking": (0, (1, 1.5)), "GA": "-", "PSO": "-", "DE": "-"}

FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e4e3df"
PISO = 1e-16   # los valores por debajo se dibujan aqui (escala logaritmica)

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "text.color": TEXTO,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
})


def _titulo(funcion, n):
    return f"{funcion.capitalize()} {n}D"


def _linea_umbral(ax, umbral, x_texto):
    ax.axhline(umbral, color=TEXTO_2, linewidth=0.8)
    ax.text(x_texto, umbral * 2.5, "umbral de exito", color=TEXTO_2, fontsize=7.5, va="bottom")


def graficar_convergencia(curvas, rejillas, cfg, metodos, out_path):
    """Mediana (linea) y rango intercuartil (banda) de f - f* sobre las
    corridas, despues de gastar cada presupuesto de evaluaciones equivalentes."""
    funciones, dims = cfg["funciones"], cfg["dimensiones"]
    fig, axes = plt.subplots(len(funciones), len(dims), figsize=(11, 7.6), sharey="row", squeeze=False)

    for i, funcion in enumerate(funciones):
        for j, n in enumerate(dims):
            ax = axes[i, j]
            rejilla = rejillas[(funcion, n)]
            for metodo in metodos:
                c = np.maximum(curvas[(funcion, n, metodo)], PISO)
                # Un heuristico no tiene valor hasta evaluar su poblacion inicial
                x, c = rejilla[~np.isnan(c[0])], c[:, ~np.isnan(c[0])]
                q1, med, q3 = np.percentile(c, [25, 50, 75], axis=0)
                ax.fill_between(x, q1, q3, color=COLORES[metodo], alpha=0.15, linewidth=0)
                ax.plot(x, med, color=COLORES[metodo], linestyle=ESTILOS[metodo], linewidth=2, label=metodo)
            _linea_umbral(ax, cfg["umbral_exito"], rejilla[0] * 1.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(_titulo(funcion, n))
            if i == len(funciones) - 1:
                ax.set_xlabel("Evaluaciones equivalentes de f   ($n_f + 2n \\cdot n_{\\nabla f}$)")
            if j == 0:
                ax.set_ylabel("$f - f^*$  (mediana y rango intercuartil)")

    asas, etiquetas = axes[0, 0].get_legend_handles_labels()
    fig.legend(asas, etiquetas, loc="upper center", ncol=len(metodos), bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Convergencia con el mismo presupuesto ({cfg['n_corridas']} corridas por metodo)",
                 fontsize=12, fontweight="bold", x=0.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def graficar_valores_finales(corridas, cfg, metodos, out_path):
    """Un punto por corrida: valor final de f - f* de cada metodo. La raya
    horizontal marca la mediana."""
    funciones, dims = cfg["funciones"], cfg["dimensiones"]
    rng = np.random.default_rng(0)   # solo para separar horizontalmente los puntos
    fig, axes = plt.subplots(len(funciones), len(dims), figsize=(11, 7.6), sharey="row", squeeze=False)

    for i, funcion in enumerate(funciones):
        for j, n in enumerate(dims):
            ax = axes[i, j]
            for k, metodo in enumerate(metodos):
                sel = (corridas["funcion"] == funcion) & (corridas["n"] == n) & (corridas["metodo"] == metodo)
                valores = np.maximum(corridas.loc[sel, "f_final"].to_numpy(), PISO)
                x = k + rng.uniform(-0.22, 0.22, size=len(valores))
                ax.scatter(x, valores, s=26, color=COLORES[metodo], edgecolor=FONDO, linewidth=0.8, zorder=3)
                ax.plot([k - 0.32, k + 0.32], [np.median(valores)] * 2, color=TEXTO, linewidth=1.6, zorder=2)
            _linea_umbral(ax, cfg["umbral_exito"], -0.45)
            ax.set_yscale("log")
            ax.set_xticks(range(len(metodos)))
            ax.set_xticklabels([m.replace("GD ", "GD\n") for m in metodos], fontsize=9, color=TEXTO)
            ax.set_xlim(-0.5, len(metodos) - 0.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(_titulo(funcion, n))
            if j == 0:
                ax.set_ylabel(f"$f - f^*$ final  (piso en {PISO:.0e})")

    fig.suptitle(f"Valor final de cada corrida ({cfg['n_corridas']} por metodo; raya = mediana)",
                 fontsize=12, fontweight="bold", x=0.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def animar_optimizacion(problema, resultados, umbral, out_path, n_cuadros=80, fps=10):
    """GIF para una funcion en 2D.

    resultados: dict metodo -> resultado de experimentos.correr(...,
    guardar_trayectoria=True). Los metodos se dibujan en el orden dado:
    primero los dos descensos por gradiente y luego los tres heuristicos.

    Todos los paneles avanzan al mismo ritmo de evaluaciones equivalentes:
    en cada cuadro cada metodo aparece como estaba despues de gastar esa
    cantidad. Los cuadros estan espaciados en escala logaritmica.
    """
    metodos = list(resultados)
    lo, hi = problema.limites
    nombre = _titulo(problema.nombre, 2)

    # Paisaje de fondo (no cuenta como evaluaciones de ningun metodo)
    eje = np.linspace(lo, hi, 300)
    XX, YY = np.meshgrid(eje, eje)
    ZZ = problema.f(np.column_stack([XX.ravel(), YY.ravel()]), contar=False).reshape(XX.shape)
    ZZ = np.log10(ZZ - problema.f_opt + 1.0)

    e_max = max(res["evals_hist"][-1] for res in resultados.values())
    cuadros = np.unique(np.round(np.geomspace(problema.costo_grad, e_max, n_cuadros)))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    ax_metodo = dict(zip(metodos, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]))
    ax_conv = axes[0, 2]

    puntos, rastros = {}, {}
    for metodo, ax in ax_metodo.items():
        # Un solo tono, mas oscuro donde f es mayor; se deja claro para que resalten los puntos
        ax.contourf(XX, YY, ZZ, levels=8, cmap="Greys", vmin=ZZ.min(), vmax=2.2 * ZZ.max())
        ax.scatter(*problema.x_opt, marker="*", s=220, color=TEXTO, edgecolor=FONDO, linewidth=1.2, zorder=5)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.grid(False)
        if resultados[metodo]["trayectoria"][0].ndim == 1:   # descenso por gradiente: un solo punto
            x0 = resultados[metodo]["trayectoria"][0]
            ax.scatter(*x0, s=60, facecolor="none", edgecolor=TEXTO, linewidth=1.4, zorder=4)
            rastros[metodo], = ax.plot([], [], color=COLORES[metodo], linewidth=1.6, zorder=3)
            puntos[metodo] = ax.scatter([], [], s=70, color=COLORES[metodo], edgecolor=FONDO, linewidth=1.5, zorder=6)
        else:
            puntos[metodo] = ax.scatter([], [], s=30, color=COLORES[metodo], edgecolor=FONDO, linewidth=0.8, zorder=6)

    curvas = {}
    for metodo in metodos:
        curvas[metodo], = ax_conv.plot([], [], color=COLORES[metodo], linestyle=ESTILOS[metodo], linewidth=2, label=metodo)
    y_max = max(res["error_hist"].max() for res in resultados.values())
    ax_conv.set(xscale="log", yscale="log", xlim=(cuadros[0], e_max * 1.2), ylim=(PISO / 3, y_max * 5),
                xlabel="Evaluaciones equivalentes de f", ylabel=f"$f - f^*$  (piso en {PISO:.0e})")
    ax_conv.set_title("Convergencia")
    _linea_umbral(ax_conv, umbral, cuadros[0] * 1.2)
    ax_conv.legend(loc="lower left", fontsize=8)

    encabezado = fig.suptitle("", fontsize=13, fontweight="bold", x=0.5, y=0.985)
    fig.text(0.5, 0.012, "Estrella = optimo global.   Circulo vacio = punto inicial del descenso por gradiente.   "
             "Fondo = curvas de nivel de f.", ha="center", fontsize=9, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955), h_pad=2.5)

    def actualizar(k):
        e = cuadros[k]
        encabezado.set_text(f"{nombre}  -  {int(e):,} evaluaciones equivalentes de f".replace(",", " "))
        for metodo, ax in ax_metodo.items():
            res = resultados[metodo]
            idx = np.searchsorted(res["evals_hist"], e, side="right") - 1
            tray = res["trayectoria"]
            if idx < 0:
                # Poblacion inicial a medio evaluar: se muestran los individuos ya evaluados
                puntos[metodo].set_offsets(tray[0][: int(e)])
                ax.set_title(f"{metodo}\nevaluando la poblacion inicial")
                continue
            fin = "   (detenido)" if e > res["evals_hist"][-1] else ""
            if metodo in rastros:
                recorrido = np.array(tray[: idx + 1])
                rastros[metodo].set_data(recorrido[:, 0], recorrido[:, 1])
                puntos[metodo].set_offsets(recorrido[-1:])
                ax.set_title(f"{metodo}{fin}\niteracion {idx}   f = {res['error_hist'][idx]:.2e}")
            else:
                puntos[metodo].set_offsets(tray[idx])
                ax.set_title(f"{metodo}{fin}\ngeneracion {idx}   mejor f = {res['error_hist'][idx]:.2e}")
            curvas[metodo].set_data(res["evals_hist"][: idx + 1], np.maximum(res["error_hist"][: idx + 1], PISO))
        return []

    pausa_final = [len(cuadros) - 1] * (2 * fps)
    anim = FuncAnimation(fig, actualizar, frames=list(range(len(cuadros))) + pausa_final)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
