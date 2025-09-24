# Suite de Tests para Modelos de Detección de Anomalías

Esta suite de tests permite evaluar múltiples modelos de machine learning con diferentes configuraciones de parámetros y tipos de ruido, utilizando paralelización para acelerar la ejecución.

## Características

- ✅ **Configuración flexible**: Archivos JSON para definir parámetros de modelos
- ✅ **Paralelización**: Ejecución concurrente de tests usando ThreadPoolExecutor
- ✅ **Selección de modelos**: Ejecutar solo los modelos deseados
- ✅ **Múltiples tipos de ruido**: RUIDO, SPURIA, DROPOUT, BLOCKING
- ✅ **Dos frecuencias**: SMA y FM
- ✅ **Reportes detallados**: Métricas de rendimiento y estadísticas
- ✅ **Modo rápido**: Para testing rápido con configuraciones básicas

## Estructura de Archivos

```
models/
├── test_config.json              # Configuración completa
├── test_config_simple.json      # Configuración simplificada
├── run_tests.py                 # Script principal
├── src/test_suite/
│   ├── __init__.py
│   └── test_runner.py           # Lógica principal de testing
└── TEST_SUITE_README.md         # Este archivo
```

## Uso Básico

### 1. Ejecutar todos los modelos con configuración completa

```bash
python run_tests.py
```

### 2. Ejecutar solo modelos específicos

```bash
python run_tests.py --models knn mlp random_forest
```

### 3. Usar configuración simplificada

```bash
python run_tests.py --config test_config_simple.json
```

### 4. Modo rápido (solo KNN y MLP básicos)

```bash
python run_tests.py --quick
```

### 5. Configurar número de workers

```bash
python run_tests.py --workers 8
```

## Configuración de Modelos

Los modelos se ejecutan con sus parámetros por defecto. Solo necesitas habilitar/deshabilitar cada modelo en la configuración JSON.

### Modelos Supervisados

- **knn**: K-Nearest Neighbors (n_neighbors=5)
- **mlp**: Multi-Layer Perceptron (hidden_layer_sizes=(100,), max_iter=100)
- **random_forest**: Random Forest (n_estimators=100)
- **xgboost**: XGBoost (n_estimators=100)

### Modelos No Supervisados

- **isolation_forest**: Isolation Forest (n_estimators=100, contamination=0.1)
- **lof**: Local Outlier Factor (n_neighbors=20, contamination=0.1)
- **elliptic_envelope**: Elliptic Envelope (contamination=0.1)
- **autoencoder**: Autoencoder (hidden_layer_sizes=(8,), max_iter=200)
- **pca_iforest**: PCA + Isolation Forest (n_components=2, n_estimators=100)

## Ejemplo de Configuración JSON

```json
{
  "models": {
    "knn": {
      "enabled": true
    },
    "mlp": {
      "enabled": true
    },
    "random_forest": {
      "enabled": true
    },
    "isolation_forest": {
      "enabled": true
    }
  },
  "noise_types": {
    "RUIDO": [0.5, 2.0, 10.0],
    "SPURIA": [2.0, 10.0, 50.0],
    "DROPOUT": [2.0, 10.0, 50.0],
    "BLOCKING": [2.0, 10.0, 50.0]
  },
  "frequencies": ["SMA", "FM"],
  "execution": {
    "max_workers": 4,
    "save_models": true,
    "generate_reports": true,
    "verbose": true
  }
}
```

## Parámetros de Línea de Comandos

| Parámetro   | Descripción                             | Default               |
| ----------- | --------------------------------------- | --------------------- |
| `--config`  | Archivo de configuración JSON           | `test_config.json`    |
| `--models`  | Modelos específicos a ejecutar          | Todos los habilitados |
| `--workers` | Número de workers para paralelización   | 4                     |
| `--output`  | Archivo de salida para resultados       | `test_results.json`   |
| `--quick`   | Modo rápido con configuraciones básicas | False                 |

## Tipos de Ruido

1. **RUIDO**: Ruido gaussiano aditivo
2. **SPURIA**: Picos espectrales con decaimiento exponencial
3. **DROPOUT**: Caídas temporales con perfil cosenoidal
4. **BLOCKING**: Elevación constante con variación aleatoria

## Salida de Resultados

### Archivo JSON de Resultados

Cada test genera un resultado con:

- ID del test
- Nombre del modelo
- Parámetros utilizados
- Tipo y nivel de ruido
- Frecuencia (SMA/FM)
- Estado de éxito/fallo
- Métricas de rendimiento
- Tiempo de ejecución

### Reportes Individuales

**Sí, se genera un reporte individual para cada configuración de test.**

Los reportes se guardan en:

- **Modelos Supervisados**: `reports/supervised/{modelo}/`
- **Modelos No Supervisados**: `reports/unsupervised/{modelo}/`

**Formato de archivos**:

- `{modelo}_{test_id}_{tipo_ruido}_{nivel_ruido}_report.md`

**Ejemplo de archivos generados**:

```
reports/supervised/knn/
├── knn_T001_RUIDO_0.5_report.md
├── knn_T002_RUIDO_2.0_report.md
├── knn_T003_RUIDO_10.0_report.md
├── knn_T004_SPURIA_2.0_report.md
└── ...

reports/unsupervised/isolation_forest/
├── isolation_forest_T001_RUIDO_0.5_report.md
├── isolation_forest_T002_RUIDO_2.0_report.md
└── ...
```

**Contenido de cada reporte**:

- Información del test (ID, tipo de ruido, nivel, frecuencia)
- Métricas de rendimiento (Precision, Recall, F1-Score, Accuracy)
- Matriz de confusión
- Resumen de clasificación

**Nota**: Para deshabilitar la generación de reportes individuales, cambia `"generate_reports": false` en la configuración JSON.

### Reporte en Consola

- Resumen de tests ejecutados
- Estadísticas por modelo
- Lista de tests fallidos
- Tiempo total de ejecución

## Ejemplos de Uso

### Test rápido de KNN

```bash
python run_tests.py --models knn --quick
```

### Test completo con 8 workers

```bash
python run_tests.py --workers 8 --output full_results.json
```

### Test solo de modelos supervisados

```bash
python run_tests.py --models knn mlp random_forest xgboost
```

### Test solo de modelos no supervisados

```bash
python run_tests.py --models isolation_forest lof elliptic_envelope autoencoder pca_iforest
```

## Requisitos

- Python 3.7+
- Todas las dependencias del proyecto original
- Archivos de datos en `data/frecs/SMA/` y `data/frecs/FM/`

## Notas de Rendimiento

- La paralelización acelera significativamente la ejecución
- El número óptimo de workers depende de tu CPU
- Los modelos supervisados suelen ser más rápidos que los no supervisados
- El modo rápido es útil para desarrollo y debugging

## Troubleshooting

### Error: "No se encontró el archivo de configuración"

- Asegúrate de que `test_config.json` existe en el directorio actual
- O especifica la ruta correcta con `--config`

### Error: "Modelo no soportado"

- Verifica que el nombre del modelo esté en la lista de opciones
- Revisa la configuración JSON

### Tests fallan con errores de memoria

- Reduce el número de workers con `--workers`
- Usa configuraciones más simples
- Verifica que tienes suficiente RAM disponible
