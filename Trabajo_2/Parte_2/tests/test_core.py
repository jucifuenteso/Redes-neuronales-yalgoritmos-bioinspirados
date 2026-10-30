"""Pruebas unitarias de la Parte 2 (unittest):  python -m unittest discover -s tests -v   (desde Parte_2)"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgdmodel import RAIZ_PARTE1, RAIZ_PARTE2  # noqa: E402  (agrega la Parte 1 al path)
from lgdmodel.config import cargar_configuracion  # noqa: E402
from lgdmodel.data import construir_lgd, construir_tabla_lgd, verificar_consistencia_con_parte1  # noqa: E402
from lgdmodel.experiments import configuracion_final_efectiva  # noqa: E402
from lgdmodel.logit_fraccional import LogitFraccional  # noqa: E402
from lgdmodel.metrics import comparar_distribuciones, metricas_regresion  # noqa: E402
from lgdmodel.mlp_lgd import MLPLGD  # noqa: E402
from pdmodel.mlp import MLPClassifier  # noqa: E402

CONFIG_DEMO = RAIZ_PARTE2 / "configs" / "demo.yaml"
CONFIG_COMPLETA = RAIZ_PARTE2 / "configs" / "default.yaml"


class PruebaObjetivoLGD(unittest.TestCase):
    def test_formula_de_la_lgd(self):
        prestamos = pd.DataFrame({"funded_amnt": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
                                  "total_rec_prncp": [0.0, 400.0, 1000.0, 0.0, 1200.0],
                                  "recoveries": [0.0, 100.0, 50.0, 50.0, 0.0],
                                  "collection_recovery_fee": [0.0, 10.0, 0.0, 80.0, 0.0]})
        # 1 - (capital pagado + max(recuperación - comisión, 0)) / desembolsado, recortada a [0, 1]
        esperado = [1.0, 1 - (400 + 90) / 1000, 0.0, 1.0, 0.0]
        np.testing.assert_allclose(construir_lgd(prestamos).to_numpy(), esperado)

    def test_lgd_solo_en_estados_con_perdida_definitiva(self):
        from lgdmodel.synthetic import generar
        configuracion = cargar_configuracion(CONFIG_DEMO)
        tabla = construir_tabla_lgd(generar(3000, 7), configuracion)
        con_lgd = tabla[tabla["lgd"].notna()]
        self.assertTrue(set(con_lgd["loan_status"]) <= set(configuracion["lgd"]["estados_perdida_definitiva"]))
        self.assertTrue((con_lgd["y"] == 1).all())
        malos_sin_castigo = tabla[tabla["loan_status"].isin(["Default", "Late (31-120 days)"])]
        self.assertGreater(len(malos_sin_castigo), 0)
        self.assertTrue(malos_sin_castigo["lgd"].isna().all())
        self.assertTrue(((con_lgd["lgd"] >= 0) & (con_lgd["lgd"] <= 1)).all())


def _gradiente_numerico_coincide(prueba, modelo, variables, objetivo, decimales=5):
    _, gradientes = modelo._loss_grads(variables, objetivo, np.random.RandomState(0))
    generador = np.random.RandomState(1)
    paso = 1e-6
    for parametro, gradiente in zip(modelo._params(), gradientes):
        for indice_plano in generador.choice(parametro.size, size=min(5, parametro.size), replace=False):
            indice = np.unravel_index(indice_plano, parametro.shape)
            valor_original = parametro[indice]
            parametro[indice] = valor_original + paso
            perdida_mas = modelo._loss_grads(variables, objetivo, np.random.RandomState(0))[0]
            parametro[indice] = valor_original - paso
            perdida_menos = modelo._loss_grads(variables, objetivo, np.random.RandomState(0))[0]
            parametro[indice] = valor_original
            prueba.assertAlmostEqual(gradiente[indice], (perdida_mas - perdida_menos) / (2 * paso), places=decimales)


class PruebaRedLGD(unittest.TestCase):
    def setUp(self):
        generador = np.random.RandomState(0)
        self.variables = generador.randn(40, 5)
        self.objetivo_fraccional = generador.rand(40)                     # LGD continua en [0, 1]

    def test_retropropagacion_mse_coincide_con_diferencias_finitas(self):
        for activacion in ["relu", "tanh", "leaky_relu"]:
            red = MLPLGD(perdida="mse", hidden=(6, 4), activation=activacion, dropout=0.0, l2=0.01,
                         dtype=np.float64, seed=3)
            red._init_params(5, 0.5, np.random.RandomState(3))
            _gradiente_numerico_coincide(self, red, self.variables, self.objetivo_fraccional)

    def test_retropropagacion_bce_con_objetivo_fraccional_coincide_con_diferencias_finitas(self):
        red = MLPLGD(perdida="bce", hidden=(6,), activation="tanh", l2=0.01, dtype=np.float64, seed=3)
        red._init_params(5, 0.5, np.random.RandomState(3))
        _gradiente_numerico_coincide(self, red, self.variables, self.objetivo_fraccional)

    def test_con_bce_es_exactamente_la_red_de_la_parte1(self):
        generador = np.random.RandomState(2)
        variables = generador.randn(600, 4)
        objetivo = generador.rand(600)
        hiperparametros = dict(hidden=(8,), activation="relu", dropout=0.1, l2=1e-4, lr=0.01, batch_size=64, seed=5)
        red_parte2 = MLPLGD(perdida="bce", **hiperparametros).fit(variables[:400], objetivo[:400],
                                                                  variables[400:], objetivo[400:], 15, 3)
        red_parte1 = MLPClassifier(**hiperparametros).fit(variables[:400], objetivo[:400],
                                                          variables[400:], objetivo[400:], 15, 3)
        np.testing.assert_allclose(red_parte2.predict_proba(variables), red_parte1.predict_proba(variables), rtol=1e-6)

    def test_bce_fraccional_se_minimiza_en_la_media_condicional(self):
        """Condición necesaria (no suficiente) para usar la BCE con y fraccional: el mínimo está en p = y."""
        objetivo = 0.3
        probabilidades = np.linspace(0.01, 0.99, 981)
        perdida = -(objetivo * np.log(probabilidades) + (1 - objetivo) * np.log(1 - probabilidades))
        self.assertAlmostEqual(float(probabilidades[np.argmin(perdida)]), objetivo, places=3)

    def test_aprende_una_senal_simple(self):
        generador = np.random.RandomState(0)
        variables = generador.randn(3000, 3)
        objetivo = 1 / (1 + np.exp(-(1.5 * variables[:, 0] - np.abs(variables[:, 1]))))
        red = MLPLGD(perdida="bce", hidden=(16,), lr=0.01, seed=0).fit(variables[:2000], objetivo[:2000],
                                                                       variables[2000:], objetivo[2000:], 60, 8)
        error_red = metricas_regresion(objetivo[2000:], red.predict_proba(variables[2000:]))["rmse"]
        error_constante = metricas_regresion(objetivo[2000:], np.full(1000, objetivo[:2000].mean()))["rmse"]
        self.assertLess(error_red, 0.5 * error_constante)


class PruebaLogitFraccional(unittest.TestCase):
    def test_con_objetivo_binario_coincide_con_regresion_logistica(self):
        from sklearn.linear_model import LogisticRegression
        generador = np.random.RandomState(0)
        variables = generador.randn(2000, 3)
        objetivo = (generador.rand(2000) < 1 / (1 + np.exp(-(variables @ [1.0, -0.5, 0.2] + 0.3)))).astype(float)
        glm = LogitFraccional(l2=0.0).fit(variables, objetivo)
        sklearn_sin_penalizacion = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(variables, objetivo)
        self.assertTrue(glm.convergio_)
        np.testing.assert_allclose(glm.coef_, sklearn_sin_penalizacion.coef_.ravel(), atol=1e-4)
        self.assertAlmostEqual(glm.intercept_, float(sklearn_sin_penalizacion.intercept_[0]), places=4)

    def test_gradiente_analitico(self):
        generador = np.random.RandomState(1)
        variables, objetivo = generador.randn(50, 4), generador.rand(50)
        glm = LogitFraccional(l2=0.05)
        parametros = generador.randn(5)
        _, gradiente = glm._perdida_y_gradiente(parametros, variables, objetivo)
        for indice in range(5):
            desplazamiento = np.zeros(5)
            desplazamiento[indice] = 1e-6
            numerico = (glm._perdida_y_gradiente(parametros + desplazamiento, variables, objetivo)[0]
                        - glm._perdida_y_gradiente(parametros - desplazamiento, variables, objetivo)[0]) / 2e-6
            self.assertAlmostEqual(gradiente[indice], numerico, places=6)


class PruebaConfiguracion(unittest.TestCase):
    def test_sobreescritura_parcial_de_la_red_final(self):
        mejor = {"hidden": [64], "activation": "relu", "dropout": 0.2, "l2": 1e-5, "optimizer": "adam",
                 "lr": 1e-3, "batch_size": 512}
        efectiva = configuracion_final_efectiva(mejor, {"hidden": [8], "activation": "tanh"})
        self.assertEqual(efectiva["hidden"], [8])
        self.assertEqual(efectiva["activation"], "tanh")
        self.assertEqual(efectiva["lr"], 1e-3)                     # lo no pedido se conserva
        with self.assertRaises(ValueError):
            configuracion_final_efectiva(mejor, {"capas": [8]})    # clave mal escrita: error, no se ignora

    def test_set_se_aplica_despues_del_yaml_de_demostracion(self):
        configuracion = cargar_configuracion(CONFIG_DEMO, ["final.n_semillas=10", "final.max_epocas=7"])
        self.assertEqual(configuracion["final"]["n_semillas"], 10)
        self.assertEqual(configuracion["final"]["max_epocas"], 7)

    def test_set_anidado_en_sobreescribir(self):
        configuracion = cargar_configuracion(CONFIG_COMPLETA, ["final.sobreescribir.hidden=[8]"])
        self.assertEqual(configuracion["final"]["sobreescribir"], {"hidden": [8]})

    def test_hereda_datos_y_semilla_de_la_parte1(self):
        from pdmodel.config import load_config
        configuracion = cargar_configuracion(CONFIG_COMPLETA)
        configuracion_parte1 = load_config(RAIZ_PARTE1 / "configs" / "default.yaml")
        self.assertEqual(configuracion["datos"], configuracion_parte1["data"])
        self.assertEqual(configuracion["semilla"], configuracion_parte1["seed"])
        modificada = cargar_configuracion(CONFIG_COMPLETA, sobreescrituras_parte1=["data.split.val_end=2013-06-30"])
        self.assertEqual(str(modificada["datos"]["split"]["val_end"]), "2013-06-30")


class PruebaConsistenciaConParte1(unittest.TestCase):
    """Con el mismo archivo, la Parte 2 reproduce exactamente créditos y particiones de la Parte 1."""

    def test_mismas_particiones_y_deteccion_de_diferencias(self):
        from lgdmodel.synthetic import escribir
        from pdmodel.data import prepare_dataset
        from lgdmodel.data import leer_archivo_original
        with tempfile.TemporaryDirectory() as carpeta_temporal:
            carpeta = Path(carpeta_temporal).as_posix()
            escribir(Path(carpeta) / "loan.csv", 4000, 11)
            sobreescrituras_parte1 = [f"paths.raw={carpeta}/loan.csv", f"paths.processed={carpeta}/p1",
                                      f"paths.tables={carpeta}/p1t", f"paths.models={carpeta}/p1m"]
            configuracion = cargar_configuracion(CONFIG_DEMO, sobreescrituras_parte1=sobreescrituras_parte1)
            datos_parte1, _ = prepare_dataset(configuracion["configuracion_parte1"])
            tabla_parte2 = construir_tabla_lgd(leer_archivo_original(Path(carpeta) / "loan.csv"), configuracion)
            pd.testing.assert_series_equal(datos_parte1["split"], tabla_parte2["split"])
            pd.testing.assert_series_equal(datos_parte1["y"], tabla_parte2["y"])
            self.assertTrue(verificar_consistencia_con_parte1(tabla_parte2, configuracion)["coincide"])

            otra_particion = cargar_configuracion(
                CONFIG_DEMO, sobreescrituras_parte1=sobreescrituras_parte1 + ["data.split.train_end=2011-12-31"])
            tabla_otra_particion = construir_tabla_lgd(leer_archivo_original(Path(carpeta) / "loan.csv"),
                                                       otra_particion)
            # data_summary.json de la Parte 1 se generó con la partición original: debe detectarse la diferencia
            self.assertFalse(verificar_consistencia_con_parte1(tabla_otra_particion, otra_particion)["coincide"])


class PruebaMetricas(unittest.TestCase):
    def test_mae_y_rmse(self):
        resultado = metricas_regresion([0.0, 0.5, 1.0], [0.1, 0.5, 0.7])
        self.assertAlmostEqual(resultado["mae"], 0.4 / 3)
        self.assertAlmostEqual(resultado["rmse"], np.sqrt(0.1 / 3))

    def test_distribuciones_identicas(self):
        valores = np.random.RandomState(0).rand(500)
        comparacion = comparar_distribuciones(valores, {"copia": valores.copy()})
        self.assertEqual(comparacion["copia"]["ks_dos_muestras_vs_observada"], 0.0)
        self.assertAlmostEqual(comparacion["copia"]["wasserstein_vs_observada"], 0.0)


if __name__ == "__main__":
    unittest.main()
