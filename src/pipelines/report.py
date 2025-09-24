import json
import os
from typing import Any, Dict


class Report:
    def __init__(self) -> None:
        pass

    def generate_unsupervised_report(self, folder: str, name: str, results: Dict[str, Any], 
                                   test_id: str = None, noise_type: str = None, noise_level: float = None) -> None:
        base_dir = f'reports/unsupervised/{folder}'
        
        # Crear nombre de archivo único si hay información adicional
        if test_id and noise_type and noise_level is not None:
            filename = f'{name}_{test_id}_{noise_type}_{noise_level}_report.md'
        else:
            filename = f'{name}_report.md'
            
        report_path = os.path.join(base_dir, filename)
        
        # 1. Crea el directorio si no existe
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 2. Formatea el reporte en Markdown para una mejor legibilidad
        report_content = self._format_markdown_report(folder, name, results, test_id, noise_type, noise_level)
        
        # 3. Guarda el reporte en el archivo
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Reporte de evaluación guardado en: {report_path}")

    def generate_supervised_report(self, folder: str, name: str, results: Dict[str, Any], 
                                 test_id: str = None, noise_type: str = None, noise_level: float = None) -> None:
        base_dir = f'reports/supervised/{folder}'
        
        # Crear nombre de archivo único si hay información adicional
        if test_id and noise_type and noise_level is not None:
            filename = f'{name}_{test_id}_{noise_type}_{noise_level}_report.md'
        else:
            filename = f'{name}_report.md'
            
        report_path = os.path.join(base_dir, filename)
        
        # 1. Crea el directorio si no existe
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 2. Formatea el reporte en Markdown para una mejor legibilidad
        report_content = self._format_markdown_report(folder, name, results, test_id, noise_type, noise_level)
        
        # 3. Guarda el reporte en el archivo
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Reporte de evaluación guardado en: {report_path}")

    def _format_markdown_report(self, folder: str, name: str, results: Dict[str, Any], 
                               test_id: str = None, noise_type: str = None, noise_level: float = None) -> str:
        report_str = f"# Reporte de Evaluación ({folder}): {name}\n\n"
        
        # Agregar información del test si está disponible
        if test_id and noise_type and noise_level is not None:
            report_str += "## Información del Test\n"
            report_str += f"- **ID del Test**: {test_id}\n"
            report_str += f"- **Tipo de Ruido**: {noise_type}\n"
            report_str += f"- **Nivel de Ruido**: {noise_level}\n"
            report_str += f"- **Frecuencia**: {name}\n\n"
        
        report_str += "## Métricas Clave\n"
        report_str += f"- **Precisión (Precision)**: {results['precision']:.4f}\n"
        report_str += f"- **Recall**: {results['recall']:.4f}\n"
        report_str += f"- **F1-Score**: {results['f1_score']:.4f}\n"
        report_str += f"- **Exactitud (Accuracy)**: {results['accuracy']:.4f}\n\n"
        
        report_str += "## Resumen de la Clasificación\n"
        classification_report = results['classification_report']
        report_str += "| Clase | Precisión | Recall | F1-Score | Soporte |\n"
        report_str += "|:---|:---:|:---:|:---:|:---:|\n"
        
        # Mapping para hacer las etiquetas más legibles
        label_map = {'0.0': 'Normal', '1.0': 'Anomalía', 'accuracy': 'Exactitud', 'macro avg': 'Macro Promedio', 'weighted avg': 'Promedio Ponderado'}

        for label, metrics in classification_report.items():
            display_label = label_map.get(label, label)
            if isinstance(metrics, dict):
                report_str += f"| **{display_label}** | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1-score']:.2f} | {int(metrics['support'])} |\n"
        
        report_str += "\n"
        report_str += "## Matriz de Confusión\n"
        conf_matrix = results['confusion_matrix']
        report_str += "| | Predicción: Normal (0) | Predicción: Anomalía (1) |\n"
        report_str += "|---|:---:|:---:|\n"
        report_str += f"| **Real: Normal (0)** | {conf_matrix[0][0]} | {conf_matrix[0][1]} |\n"
        report_str += f"| **Real: Anomalía (1)** | {conf_matrix[1][0]} | {conf_matrix[1][1]} |\n\n"
        
        report_str += "## Detalles\n"
        report_str += f"- **Número de Anomalías Detectadas**: {results['n_anomalies_detected']}\n"
        report_str += f"- **Número de Anomalías Reales**: {results['n_anomalies_actual']}\n\n"

        return report_str