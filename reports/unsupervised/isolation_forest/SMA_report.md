# Reporte de Evaluación (isolation_forest): SMA

## Métricas Clave
- **Precisión (Precision)**: 0.8636
- **Recall**: 1.0000
- **F1-Score**: 0.9268
- **Exactitud (Accuracy)**: 0.9211

## Resumen de la Clasificación
| Clase | Precisión | Recall | F1-Score | Soporte |
|:---|:---:|:---:|:---:|:---:|
| **Normal** | 1.00 | 0.84 | 0.91 | 38 |
| **Anomalía** | 0.86 | 1.00 | 0.93 | 38 |
| **Macro Promedio** | 0.93 | 0.92 | 0.92 | 76 |
| **Promedio Ponderado** | 0.93 | 0.92 | 0.92 | 76 |

## Matriz de Confusión
| | Predicción: Normal (0) | Predicción: Anomalía (1) |
|---|:---:|:---:|
| **Real: Normal (0)** | 32 | 6 |
| **Real: Anomalía (1)** | 0 | 38 |

## Detalles
- **Número de Anomalías Detectadas**: 44
- **Número de Anomalías Reales**: 38.0

