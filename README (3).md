# Ingeniería de Clasificación de Riesgo en Neurofibromatosis Tipo 1 (NF1) basada en Características Clínicas: mediante Ensemble Learning

**Autora:** Jessica Melani Romero Lora

---

## 1. Introducción

La Neurofibromatosis Tipo 1 (NF1) es un trastorno genético autosómico dominante causado por mutaciones en el gen *NF1*, que actúa como supresor tumoral mediante la regulación de la proteína neurofibromina. Esta condición se caracteriza por una expresividad clínica altamente variable, que abarca desde manifestaciones cutáneas benignas —como las manchas café con leche (*Café-au-Lait Spots*)— hasta complicaciones tumorales severas, incluyendo neurofibromas plexiformes y, en un subconjunto de casos, transformación maligna hacia tumores malignos de la vaina de los nervios periféricos (MPNST).

Dado que la evolución clínica de la NF1 no sigue un patrón determinístico único, la estratificación temprana del riesgo de complicación tumoral severa (variable `Tumour_Case`) representa un problema de clasificación binaria de alto valor clínico. El presente proyecto aborda este problema mediante técnicas de aprendizaje automático supervisado, integrando métodos de selección de características y un enfoque de *ensemble learning* comparativo (Regresión Logística, Random Forest y XGBoost) con el fin de identificar el modelo con mejor desempeño y estabilidad predictiva.

## 2. Justificación (Protocolo de Investigación)

### 2.1 Planteamiento del problema

La identificación oportuna de pacientes con NF1 en riesgo de desarrollar complicaciones tumorales severas es crítica para el seguimiento clínico, la toma de decisiones terapéuticas y la asignación eficiente de recursos de salud (por ejemplo, priorización de estudios de imagen o interconsultas a oncología). Sin embargo, el diagnóstico y pronóstico de la NF1 se basa predominantemente en criterios clínicos cualitativos, lo cual introduce variabilidad interobservador y limita la capacidad de anticipar el curso de la enfermedad en etapas tempranas.

### 2.2 Pregunta de investigación

¿Es posible construir un modelo de clasificación robusto y clínicamente interpretable que, a partir de características clínicas basales de pacientes con NF1, prediga la presencia de una complicación tumoral severa (`Tumour_Case`)?

### 2.3 Objetivo general

Desarrollar y validar un modelo de *ensemble learning* para la clasificación del riesgo de complicación tumoral severa en pacientes con NF1, a partir de un conjunto reducido y clínicamente relevante de características, garantizando estabilidad y generalización mediante validación cruzada estratificada.

### 2.4 Objetivos específicos

1. Realizar el preprocesamiento y limpieza de la base de datos clínica, asegurando consistencia en el nombrado de variables y en la codificación de la variable objetivo.
2. Aplicar dos métodos independientes de selección de características (regularización L1/Lasso e importancia por Random Forest) para reducir la dimensionalidad y mitigar el sobreajuste.
3. Entrenar y optimizar hiperparámetros de tres algoritmos de clasificación (Regresión Logística, Random Forest y XGBoost) mediante `GridSearchCV`.
4. Evaluar el desempeño y la estabilidad de los modelos mediante validación cruzada estratificada de 5 particiones (*Stratified K-Fold*, K=5), utilizando las métricas AUC-ROC, F1-Score y Recall.
5. Comparar los modelos en función de su desempeño promedio y su desviación estándar entre folds, para seleccionar el modelo con mejor balance entre capacidad predictiva y estabilidad.

### 2.5 Justificación metodológica

- **Desbalance de clases:** dado que las complicaciones tumorales severas suelen representar la clase minoritaria, se utiliza `stratify=y` en la partición de datos, `class_weight='balanced'` en los modelos lineales y basados en árboles, y `StratifiedKFold` en la validación cruzada, con el fin de preservar la proporción de clases en cada partición.
- **Selección de características por doble método (Lasso ∩ Random Forest):** se emplea la intersección de dos enfoques de selección —uno lineal (regularización L1) y uno no lineal (importancia de variables)— para obtener un subconjunto de características robusto, reduciendo el riesgo de sobreajuste asociado a un espacio de variables clínicas amplio y potencialmente correlacionado.
- **Eliminación de colinealidad:** se analiza la matriz de correlación absoluta sobre el conjunto de entrenamiento para identificar y documentar pares de variables altamente correlacionadas (|r| > 0.9), evitando redundancia informativa.
- **Ensemble comparativo:** la inclusión de tres familias de algoritmos (lineal, bagging y boosting) permite contrastar supuestos de linealidad frente a relaciones no lineales entre las características clínicas y el desenlace tumoral.
- **Validación cruzada estratificada repetida por fold:** cada modelo óptimo (resultado de `GridSearchCV`) se reentrena de manera independiente (`clone`) en cada uno de los 5 folds, permitiendo estimar no solo el desempeño promedio sino también su variabilidad (desviación estándar), un criterio clave de estabilidad para su eventual uso clínico.

### 2.6 Relevancia esperada

Los resultados de este proyecto buscan aportar evidencia preliminar sobre la viabilidad de modelos de aprendizaje automático como herramienta de apoyo a la decisión clínica en el seguimiento de pacientes con NF1, sentando las bases para futuros estudios de validación externa con cohortes clínicas independientes.

---

## 3. Estructura del pipeline

El script `NF1.py` está organizado en cuatro fases secuenciales:

### Fase 1: Preprocesamiento y limpieza
- Lectura del dataset desde un repositorio de GitHub (`neurofibromatosis.csv`).
- Normalización de nombres de columnas (eliminación de espacios y caracteres especiales).
- Renombrado de variables clave (`Tumour_Case`, `Cafe_au_Lait`).
- Separación de variables predictoras (`X`) y variable objetivo (`y = Tumour_Case`), excluyendo `CaseType`.
- División estratificada 80/20 en entrenamiento y prueba (`train_test_split`, `random_state=42`).
- Análisis de correlación absoluta para detectar variables redundantes (umbral > 0.9).

### Fase 2: Feature Engineering (Selección de características)
- **Método 1 — Regularización L1 (Lasso):** `LogisticRegression(penalty='l1', solver='liblinear')` con búsqueda de hiperparámetro `C` mediante `GridSearchCV` (`scoring='f1'`, `cv=5`).
- **Método 2 — Importancia por Random Forest:** `RandomForestClassifier` con selección de variables cuya importancia supera la media.
- **Subconjunto final:** intersección de las variables seleccionadas por ambos métodos (`FINAL_FEATURES`).

### Fase 3: Modelado y validación
- Definición de `StratifiedKFold` (K=5, `shuffle=True`, `random_state=42`).
- Optimización de hiperparámetros por modelo mediante `GridSearchCV` (`scoring='roc_auc'`).
- Reentrenamiento por fold del mejor modelo (`clone`) y cálculo de métricas: AUC-ROC, F1-Score y Recall.
- Modelos comparados: **Regresión Logística (RL)**, **Random Forest (RF)** y **XGBoost (XGB)**.

### Fase 4: Métricas y reporte
- Tabla de resultados detallados por fold y modelo (`results_df`).
- Tabla resumen con media y desviación estándar por modelo (`final_summary`).

---

## 4. Requisitos

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Instalación rápida:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 5. Fuente de datos

El conjunto de datos se obtiene directamente desde GitHub en formato CSV:

```
https://raw.githubusercontent.com/frontendbby/Fun-datasets/refs/heads/main/neurofibromatosis.csv
```

## 6. Uso

```bash
python NF1.py
```

El script imprimirá en consola:
- Resumen exploratorio del dataset (`df.info()`, `df.head()`).
- Variables seleccionadas por Lasso y por Random Forest, y su intersección final.
- Matriz de correlación (visualización con `seaborn.heatmap`).
- Mejores hiperparámetros por modelo (`GridSearchCV`).
- Tabla detallada de métricas por fold y tabla comparativa de estabilidad (media ± desviación estándar) por modelo.

## 7. Notas técnicas

> ⚠️ El script fue exportado originalmente desde Google Colab. Antes de ejecutarlo en un entorno local, es necesario:
> - Definir explícitamente la lista `models` (nombre, `Pipeline` y `param_grid` por modelo), ya que se referencia en el bucle de la Fase 3 pero no se declara en el script exportado.
> - Reemplazar `display(results_df)` por `print(results_df)` fuera de un notebook, dado que `display()` es una función nativa de entornos Jupyter/Colab.
> - Definir `X_train_final` o utilizar consistentemente `X_final` (basado en `selected_rf_features`), ya que el script contiene dos bloques de entrenamiento con nombres de variables distintos que deben unificarse.

## 8. Resultados

Los resultados finales (AUC, F1-Score y Recall, con media y desviación estándar por modelo) se generan dinámicamente al ejecutar el pipeline completo y se recomienda documentarlos en esta sección una vez validados, junto con las variables clínicas incluidas en `FINAL_FEATURES`.

## 9. Licencia

Este proyecto se comparte con fines académicos y de investigación. Ajustar la licencia según los lineamientos institucionales correspondientes.
