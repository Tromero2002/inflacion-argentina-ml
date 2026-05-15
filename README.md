# Inflación en Argentina: Análisis y Pronóstico con Machine Learning

Modelo de Random Forest para el análisis, *nowcasting* y *forecasting* de la inflación mensual argentina (IPC), integrando 30+ variables macro-financieras y un análisis de rupturas estructurales por regímenes inflacionarios.

---

## Resumen ejecutivo

La inflación argentina presenta una dinámica significativamente distinta a la de economías estables: múltiples rupturas estructurales, cambios de régimen frecuentes y un peso desproporcionado de las expectativas. Este proyecto:

1. **Caracteriza** la dinámica inflacionaria local mediante detección de quiebres estructurales (algoritmo PELT) y la contrasta con la de EE.UU.
2. **Identifica** los determinantes inflacionarios dominantes en cada régimen (estable, moderado, alta inflación).
3. **Construye** dos modelos de Random Forest:
   - **Nowcasting**: estimación del IPC del mes en curso (MAE = 0.34 pp).
   - **Forecasting**: predicción a un paso (t+1) usando rezagos (MAE = 1.00 pp en validación cruzada).

El enfoque no busca probar causalidad, sino aportar una **base metodológica robusta** para anticipar la dinámica inflacionaria en un contexto de alta volatilidad.

---

## Tabla de contenidos

- [Motivación](#motivación)
- [Datos](#datos)
- [Metodología](#metodología)
- [Modelos](#modelos)
- [Resultados](#resultados)
- [Hallazgos sobre los determinantes inflacionarios](#hallazgos-sobre-los-determinantes-inflacionarios)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo reproducirlo](#cómo-reproducirlo)
- [Stack técnico](#stack-técnico)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Referencias académicas](#referencias-académicas)
- [Autores](#autores)

---

## Motivación

A diferencia de economías con regímenes monetarios estables —donde la curva de Phillips conserva poder explicativo razonable (Galí y Gertler, 1999)—, en Argentina el proceso inflacionario se redefine cada pocos años: la relación entre la inflación y sus determinantes tradicionales (agregados monetarios, tipo de cambio real, tasa de interés) cambia con el régimen macroeconómico vigente.

Esto plantea dos problemas concretos para cualquier modelo predictivo:

- Un modelo entrenado sobre toda la serie histórica mezcla regímenes con dinámicas heterogéneas.
- En regímenes de alta inflación, la dinámica se vuelve **autorreferencial**: las expectativas alimentan la inflación futura con relativa independencia de los fundamentos (Cagan, 1956; Heymann y Leijonhufvud, 1995).

El proyecto aborda ambos problemas: primero caracteriza los regímenes, luego construye modelos sensibles a esa estructura.

---

## Datos

**Período cubierto:** julio 2004 – marzo 2025 (frecuencia mensual).

**Variable objetivo:** `ipc_var_mensual` — variación mensual del IPC argentino.

### Variables explicativas (30+)

| Grupo | Variables |
|---|---|
| **Índices de precios** | `inflacion_t-1`, `inflacion_t-2`, `expectativas_infla` (REM-BCRA y Encuesta Di Tella) |
| **Agregados monetarios** | `BM_varm` y rezagos, `M2_varm` y rezagos, `tasa_i_promedio` (depósitos a 30 días) |
| **Actividad** | `emae_varm_b2004` (Estimador Mensual de Actividad Económica), `RIPTE_varm` y rezago (salarios) |
| **Tipo de cambio** | `brecha` (CCL vs. oficial), `Varm_ccl_unificado`, `Varm_ccl_implicito`, `tcr_multilateral_varm`, `tcr_dolar_varm`, `tcr_bra_varm` |
| **Precios internacionales** | `ipc_usa_varm` (CPI EE.UU.), variación mensual de soja, maíz, trigo, petróleo, aceite vegetal |
| **Volatilidad** | `volatilidad_6m`, `cv_6m` (coeficiente de variación móvil) |
| **Político-electorales** | `año_mandato`, `es_año_electoral` (dummy) |

### Fuentes

- **Argentina:** BCRA, INDEC, SSPM, datos.gob.ar, Centro de Estudios Sociales y Financieros (Bolsa de Comercio de Santa Fe).
- **Internacional:** FRED (Federal Reserve of St. Louis), IMF Primary Commodity Price System.
- **Mercado financiero local:** Rava, Investing.com (CCL histórico).

Para el período de intervención del INDEC (2007–2015), se utilizaron índices alternativos (IPC San Luis, IPC Congreso) como reemplazo del IPC oficial. Las expectativas de inflación se imputaron con un valor de `-99` entre 2007-02 y 2016-05, permitiendo al modelo identificar el período como dato faltante diferenciado.

---

## Metodología

### 1. Detección de rupturas estructurales

Se aplicó el algoritmo **PELT** (`ruptures`) sobre la serie de inflación mensual para detectar puntos de cambio estructural en media y tendencia. El mismo procedimiento se aplicó sobre el CPI de EE.UU. como contraste:

- **Argentina:** múltiples quiebres detectados (con penalización = 3) y un número manejable de regímenes (con penalización = 10).
- **EE.UU.:** sin quiebres significativos desde 2004 — coherente con un régimen de *inflation targeting* y expectativas ancladas.

### 2. Análisis por régimen

Se entrenó un Random Forest exploratorio en cada régimen para comparar la importancia relativa de las variables:

| Régimen | Período | Hallazgo principal |
|---|---|---|
| Estable | 2004-07 → 2007-02 | Sin variable dominante; conjunto balanceado de determinantes. |
| Moderado | 2018-05 → 2022-02 | Importancia compartida entre fundamentos monetarios y expectativas. |
| Alta inflación | 2022-03 → 2024-03 | **Expectativas dominan**; M2 y TCR pasan a roles marginales. |

Este resultado motiva el diseño de los modelos predictivos: en un contexto de alta inflación los modelos deben dar peso central a expectativas y rezagos.

---

## Modelos

Se construyeron dos modelos sobre **Random Forest Regressor** (criterio `absolute_error`, validación con `TimeSeriesSplit`).

### Nowcasting

Predicción del IPC del **mes en curso** usando información de alta frecuencia disponible antes del dato oficial.

**Hiperparámetros óptimos (GridSearchCV):**

```
n_estimators       = 100
max_depth          = 15
min_samples_split  = 2
min_samples_leaf   = 2
criterion          = 'absolute_error'
TimeSeriesSplit    = n_splits=20
```

Se desarrolló también una **versión simplificada** filtrando variables con `feature_importance < 0.01` para evaluar el trade-off complejidad–performance.

### Forecasting

Modelo a un paso adelante (`π_{t+1} = f(π_t, π_{t-1}, i_t, ...)`) que aprende a partir de rezagos. Permite predecir el mes siguiente sin necesidad de información contemporánea.

**Grilla de hiperparámetros explorada:**

```
n_estimators       : [100, 200]
max_depth          : [3, 5, 10]
min_samples_split  : [6, 10]
min_samples_leaf   : [3, 4, 8]
max_features       : [0.5, 0.7]
criterion          : ['absolute_error']
```

La grilla fue diseñada explícitamente para **prevenir overfitting** —prioridad en una serie tan volátil como la argentina—, sacrificando flexibilidad por estabilidad.

### Nota sobre la métrica: MAE en lugar de RMSE

Se eligió el **Error Absoluto Medio** como métrica principal porque:

1. **Interpretabilidad directa:** un MAE de 0.7 se lee como "en promedio, el modelo se equivoca en 0.7 puntos porcentuales".
2. **Robustez a outliers:** el RMSE penaliza desproporcionadamente errores grandes al elevarlos al cuadrado. En un contexto con shocks impredecibles (devaluaciones, salida del cepo, etc.) un solo mes atípico distorsionaría la evaluación de un modelo que, en el resto del año, predice razonablemente bien.

Por coherencia interna, los árboles del modelo se optimizaron también con `criterion='absolute_error'`.

---

## Resultados

| Modelo | MAE (validación cruzada) | MAE (in-sample) |
|---|---|---|
| Nowcasting general | — | **0.3418 pp** |
| Nowcasting simplificado | — | 0.3503 pp |
| Forecasting (Random Forest) | **1.0017 pp** | 0.5341 pp |

La brecha entre validación cruzada e in-sample en el modelo de forecasting (1.00 vs. 0.53) refleja un grado moderado de sobreajuste — el modelo captura parte del ruido de la serie, esperable dada la alta volatilidad.

### Test out-of-sample: predicción mayo 2025

Con datos hasta marzo 2025, el modelo de forecasting predijo mayo 2025 con un error de **+1.45 pp** respecto al valor real (1.5%). Si bien excede el MAE promedio de validación cruzada, se mantiene dentro de un orden de magnitud razonable. El sesgo se explica por la alta volatilidad de marzo 2025 y por un cambio estructural no capturado por el modelo: la salida del cepo cambiario el 14 de abril de 2025.

---

## Hallazgos sobre los determinantes inflacionarios

Los hallazgos del modelo de forecasting (basado en *feature importance*) son consistentes con la literatura sobre regímenes de alta inflación:

- **Bajo peso de agregados monetarios:** solo `M2_varm_lag_2` aparece en el top de variables. La tesis ortodoxa de "la inflación es siempre y en todo lugar un fenómeno monetario" no se verifica nítidamente en este período.
- **Predominio de variables nominales:** las tres más importantes son nominales (excepto `tcr_bra_varm`, dato llamativo que sugiere una dependencia subestimada respecto del socio comercial principal).
- **Rol central de la tasa de interés:** consistente con la paridad de Fisher (`i = r + π`), bajo supuesto de `r` razonablemente estable.
- **Expectativas con peso relevante:** indicio de que el modelo capta el desanclaje típico de regímenes de alta inflación.
- **Inercia inflacionaria:** los rezagos de la propia inflación tienen una contribución no menor.
- **Brecha cambiaria entre el top:** plausibilidad de la hipótesis cambiaria como mecanismo subyacente.

> **Nota:** estos son hallazgos predictivos, no causales. El objetivo del proyecto es el pronóstico, no la inferencia estructural.

---

## Estructura del repositorio

```
.
├── README.md                   # Este archivo
├── inflacion.ipynb             # Notebook principal (análisis + modelos)
├── data/
│   └── DATASET_V5.csv          # Dataset consolidado (variables explicativas + IPC)
└── requirements.txt            # Dependencias
```

---

## Cómo reproducirlo

### Opción 1 — Google Colab (recomendado)

El notebook está diseñado para ejecutarse en Colab. Pasos:

1. Abrir `inflacion.ipynb` en Colab.
2. Cargar `DATASET_V5.csv` al almacenamiento de la sesión cuando lo solicite la celda correspondiente.
3. Ejecutar las celdas en orden.

### Opción 2 — Entorno local

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repo>.git
cd <repo>

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Abrir el notebook
jupyter notebook inflacion.ipynb
```

**Tiempo de ejecución:** la grilla completa del GridSearch puede tardar varios minutos. Los hiperparámetros óptimos están hardcodeados en el notebook para reproducibilidad sin necesidad de re-correr la búsqueda.

---

## Stack técnico

- **Lenguaje:** Python 3.x
- **Análisis de datos:** `pandas`, `numpy`
- **Modelado:** `scikit-learn` (RandomForestRegressor, GridSearchCV, TimeSeriesSplit), `xgboost`
- **Detección de rupturas estructurales:** `ruptures` (algoritmo PELT)
- **Visualización:** `matplotlib`, `seaborn`
- **Entorno:** Google Colab / Jupyter

---

## Limitaciones y trabajo futuro

- **Frecuencia de los datos:** mensual. Incorporar datos de alta frecuencia (precios online, indicadores diarios de mercado) podría mejorar el nowcasting (cf. Modugno, 2013).
- **Rupturas estructurales recientes:** la salida del cepo en abril 2025 y eventos de política económica posteriores no están en el set de entrenamiento. Un modelo con reentrenamiento periódico (rolling window) capturaría mejor estos cambios.
- **Comparación con benchmarks:** falta contrastar performance contra modelos econométricos clásicos (VAR, ARIMA, curva de Phillips) y un benchmark trivial tipo Random Walk.
- **Selección de variables:** se podría aplicar una selección más rigurosa (LASSO, Boruta) en lugar del filtro por feature importance > 0.01.
- **Modelos alternativos:** evaluar XGBoost (ya importado en el notebook), Gradient Boosting y redes recurrentes (LSTM) sobre la misma base.

---

## Referencias académicas

- Bañbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). *Now-casting and the real-time data flow*. In *Handbook of Economic Forecasting*, Vol. 2A.
- Cagan, P. (1956). *The Monetary Dynamics of Hyperinflation*. In M. Friedman (Ed.), *Studies in the Quantity Theory of Money*. University of Chicago Press.
- Galí, J., & Gertler, M. (1999). *Inflation dynamics: A structural econometric analysis*. *Journal of Monetary Economics*, 44(2), 195–222.
- Giannone, D., Reichlin, L., & Small, D. (2008). *Nowcasting: The real-time informational content of macroeconomic data*. *Journal of Monetary Economics*, 55(4), 665–676.
- Heymann, D., & Leijonhufvud, A. (1995). *High Inflation*. Oxford University Press.
- Knotek, E. S., & Zaman, S. (2017). *Nowcasting U.S. headline and core inflation*. *Journal of Money, Credit and Banking*, 49(5), 931–968.
- Modugno, M. (2013). *Now-casting inflation using high frequency data*. *International Journal of Forecasting*, 29(4), 664–675.

---

## Autores

- **Tomás Lucas Romero** — [LinkedIn](https://www.linkedin.com/in/tomas-lucas-romero/)
- **Jorge Iván Juárez A.**
- **Gonzalo Alderete**

Proyecto desarrollado en el marco de cursada universitaria, Universidad de Buenos Aires.

---

## Licencia

Proyecto académico de uso libre con fines educativos y de investigación. Para uso comercial o reproducción extensiva, contactar a los autores.
