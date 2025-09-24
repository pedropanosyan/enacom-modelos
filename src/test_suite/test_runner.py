import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import numpy as np
import pandas as pd

from src.common.load_anomaly import create_synthetic_anomaly
from src.common.load_clean import get_clean_data
from src.pipelines.setup import Setup
from src.pipelines.evaluate import Evaluate
from src.pipelines.report import Report

# Import all model functions
from src.models.supervised.knn import train_knn, predict_knn
from src.models.supervised.mlp import train_mlp, predict_mlp
from src.models.supervised.random_forest import train_random_forest, predict_random_forest
from src.models.supervised.xgboost import train_xgboost, predict_xgboost
from src.models.unsupervised.isolation_forest import train_isolation_forest, predict_isolation_forest
from src.models.unsupervised.lof import train_lof, predict_lof
from src.models.unsupervised.elliptic_envelope import train_elliptic_envelope, predict_elliptic_envelope
from src.models.unsupervised.autoencoder import train_autoencoder, predict_autoencoder
from src.models.unsupervised.pca import train_pca_iforest, predict_pca_iforest


@dataclass
class TestResult:
    """Resultado de un test individual"""
    test_id: str
    model_name: str
    model_params: Dict[str, Any]
    noise_type: str
    noise_level: float
    frequency: str
    success: bool
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None


class TestRunner:
    """Ejecutor principal de la suite de tests"""
    
    def __init__(self, config_path: str = "test_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.evaluator = Evaluate()
        self.report = Report()
        self.setup = Setup()
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        
        # Cargar datos una sola vez
        self._load_data()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo JSON"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _load_data(self):
        """Carga los datos limpios para todas las frecuencias"""
        print("Cargando datos...")
        self.clean_data = {}
        for frequency in self.config['frequencies']:
            self.clean_data[frequency] = get_clean_data(f"data/frecs/{frequency}")
        print("Datos cargados correctamente")
    
    def generate_test_combinations(self) -> List[Dict[str, Any]]:
        """Genera todas las combinaciones de tests basadas en la configuración"""
        test_combinations = []
        test_id = 1
        
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', False):
                continue
                
            # Usar parámetros por defecto (sin configuración específica)
            model_params = {}
            
            # Para cada tipo de ruido y nivel
            for noise_type, noise_levels in self.config['noise_types'].items():
                for noise_level in noise_levels:
                    # Para cada frecuencia
                    for frequency in self.config['frequencies']:
                        test_combinations.append({
                            'test_id': f"T{test_id:03d}",
                            'model_name': model_name,
                            'model_params': model_params,
                            'noise_type': noise_type,
                            'noise_level': noise_level,
                            'frequency': frequency
                        })
                        test_id += 1
        
        return test_combinations
    
    def _run_single_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Ejecuta un test individual"""
        start_time = time.time()
        
        try:
            # Crear datos con ruido
            clean_data = self.clean_data[test_config['frequency']]
            noisy_data = create_synthetic_anomaly(
                clean_data, 
                test_config['noise_type'], 
                test_config['noise_level']
            )
            
            # Preparar datos según el tipo de modelo
            if test_config['model_name'] in ['knn', 'mlp', 'random_forest', 'xgboost']:
                # Modelos supervisados
                x_train, x_test, y_train, y_test = self.setup.get_train_data_supervised(
                    clean_data, noisy_data
                )
                
                # Entrenar modelo
                model = self._train_supervised_model(
                    test_config['model_name'], 
                    x_train, y_train, 
                    test_config['frequency']
                )
                
                # Predecir
                predictions = self._predict_supervised_model(
                    test_config['model_name'], 
                    model, 
                    x_test
                )
                
                # Evaluar
                metrics = self.evaluator.evaluate_supervised(y_test, predictions)
                
                # Generar reporte individual para modelo supervisado
                if self.config['execution']['generate_reports']:
                    self.report.generate_supervised_report(
                        test_config['model_name'], 
                        test_config['frequency'], 
                        metrics,
                        test_id=test_config['test_id'],
                        noise_type=test_config['noise_type'],
                        noise_level=test_config['noise_level']
                    )
                
            else:
                # Modelos no supervisados
                train_data, clean_test, anomaly_test, test_data = self.setup.get_train_data_unsupervised(
                    clean_data, noisy_data
                )
                
                # Entrenar modelo
                model = self._train_unsupervised_model(
                    test_config['model_name'], 
                    train_data, 
                    test_config['frequency']
                )
                
                # Predecir
                predictions, scores = self._predict_unsupervised_model(
                    test_config['model_name'], 
                    model, 
                    test_data
                )
                
                # Crear etiquetas verdaderas
                y_true = np.concatenate([
                    np.zeros(len(clean_test)), 
                    np.ones(len(anomaly_test))
                ])
                
                # Evaluar
                metrics = self.evaluator.evaluate_unsupervised(y_true, predictions, scores)
                
                # Generar reporte individual para modelo no supervisado
                if self.config['execution']['generate_reports']:
                    self.report.generate_unsupervised_report(
                        test_config['model_name'], 
                        test_config['frequency'], 
                        metrics,
                        test_id=test_config['test_id'],
                        noise_type=test_config['noise_type'],
                        noise_level=test_config['noise_level']
                    )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_config['test_id'],
                model_name=test_config['model_name'],
                model_params=test_config['model_params'],
                noise_type=test_config['noise_type'],
                noise_level=test_config['noise_level'],
                frequency=test_config['frequency'],
                success=True,
                metrics=metrics,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_config['test_id'],
                model_name=test_config['model_name'],
                model_params=test_config['model_params'],
                noise_type=test_config['noise_type'],
                noise_level=test_config['noise_level'],
                frequency=test_config['frequency'],
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _train_supervised_model(self, model_name: str, 
                              x_train: pd.DataFrame, y_train: pd.Series, 
                              frequency: str):
        """Entrena un modelo supervisado"""
        if model_name == 'knn':
            return train_knn(x_train.values, y_train.values, frequency)
        elif model_name == 'mlp':
            return train_mlp(x_train.values, y_train.values, frequency)
        elif model_name == 'random_forest':
            return train_random_forest(x_train.values, y_train.values, frequency)
        elif model_name == 'xgboost':
            return train_xgboost(x_train.values, y_train.values, frequency)
        else:
            raise ValueError(f"Modelo supervisado no soportado: {model_name}")
    
    def _predict_supervised_model(self, model_name: str, model, x_test: pd.DataFrame):
        """Predice con un modelo supervisado"""
        if model_name == 'knn':
            return predict_knn(model, x_test.values)
        elif model_name == 'mlp':
            return predict_mlp(model, x_test.values)
        elif model_name == 'random_forest':
            return predict_random_forest(model, x_test.values)
        elif model_name == 'xgboost':
            return predict_xgboost(model, x_test.values)
        else:
            raise ValueError(f"Modelo supervisado no soportado: {model_name}")
    
    def _train_unsupervised_model(self, model_name: str, 
                                train_data: np.ndarray, frequency: str):
        """Entrena un modelo no supervisado"""
        if model_name == 'isolation_forest':
            return train_isolation_forest(train_data, frequency=frequency)
        elif model_name == 'lof':
            return train_lof(train_data, frequency=frequency)
        elif model_name == 'elliptic_envelope':
            return train_elliptic_envelope(train_data, frequency=frequency)
        elif model_name == 'autoencoder':
            return train_autoencoder(train_data, frequency=frequency)
        elif model_name == 'pca_iforest':
            return train_pca_iforest(train_data, frequency=frequency)
        else:
            raise ValueError(f"Modelo no supervisado no soportado: {model_name}")
    
    def _predict_unsupervised_model(self, model_name: str, model, test_data: np.ndarray):
        """Predice con un modelo no supervisado"""
        if model_name == 'isolation_forest':
            return predict_isolation_forest(model, test_data)
        elif model_name == 'lof':
            return predict_lof(model, test_data)
        elif model_name == 'elliptic_envelope':
            return predict_elliptic_envelope(model, test_data)
        elif model_name == 'autoencoder':
            # El modelo devuelto por train_autoencoder es una tupla (scaler, autoencoder)
            scaler, autoencoder_model = model
            return predict_autoencoder(scaler, autoencoder_model, test_data)
        elif model_name == 'pca_iforest':
            # El modelo devuelto por train_pca_iforest es una tupla (pca, iforest)
            pca, iforest_model = model
            return predict_pca_iforest(pca, iforest_model, test_data)
        else:
            raise ValueError(f"Modelo no supervisado no soportado: {model_name}")
    
    def run_tests(self, selected_models: Optional[List[str]] = None, 
                  max_workers: Optional[int] = None) -> List[TestResult]:
        """Ejecuta todos los tests con paralelización"""
        
        # Filtrar modelos si se especifican
        if selected_models:
            for model_name in self.config['models']:
                if model_name not in selected_models:
                    self.config['models'][model_name]['enabled'] = False
        
        # Generar combinaciones de tests
        test_combinations = self.generate_test_combinations()
        print(f"Total de tests a ejecutar: {len(test_combinations)}")
        
        # Configurar workers
        if max_workers is None:
            max_workers = self.config['execution']['max_workers']
        
        # Ejecutar tests en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todos los trabajos
            future_to_test = {
                executor.submit(self._run_single_test, test_config): test_config 
                for test_config in test_combinations
            }
            
            # Procesar resultados conforme se completan
            for future in as_completed(future_to_test):
                test_config = future_to_test[future]
                try:
                    result = future.result()
                    with self.lock:
                        self.results.append(result)
                    
                    if self.config['execution']['verbose']:
                        status = "✓" if result.success else "✗"
                        print(f"{status} {result.test_id}: {result.model_name} - "
                              f"{result.noise_type}@{result.noise_level} - "
                              f"{result.frequency} - {result.execution_time:.2f}s")
                        
                except Exception as e:
                    print(f"Error procesando test {test_config['test_id']}: {e}")
        
        return self.results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Genera un reporte resumen de todos los tests"""
        if not self.results:
            return {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
                "model_statistics": {},
                "failed_test_details": []
            }
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        # Estadísticas por modelo
        model_stats = {}
        for result in successful_tests:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'avg_execution_time': 0,
                    'best_accuracy': 0,
                    'noise_performance': {}
                }
            
            model_stats[result.model_name]['total_tests'] += 1
            model_stats[result.model_name]['successful_tests'] += 1
            
            if result.metrics and 'accuracy' in result.metrics:
                model_stats[result.model_name]['best_accuracy'] = max(
                    model_stats[result.model_name]['best_accuracy'],
                    result.metrics['accuracy']
                )
            
            # Performance por tipo de ruido
            noise_key = f"{result.noise_type}_{result.noise_level}"
            if noise_key not in model_stats[result.model_name]['noise_performance']:
                model_stats[result.model_name]['noise_performance'][noise_key] = []
            
            if result.metrics and 'accuracy' in result.metrics:
                model_stats[result.model_name]['noise_performance'][noise_key].append(
                    result.metrics['accuracy']
                )
        
        # Calcular tiempos promedio
        for model_name in model_stats:
            model_times = [r.execution_time for r in successful_tests 
                          if r.model_name == model_name and r.execution_time]
            if model_times:
                model_stats[model_name]['avg_execution_time'] = sum(model_times) / len(model_times)
        
        summary = {
            'total_tests': len(self.results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(self.results) if self.results else 0,
            'model_statistics': model_stats,
            'failed_test_details': [
                {
                    'test_id': r.test_id,
                    'model': r.model_name,
                    'error': r.error_message
                } for r in failed_tests
            ]
        }
        
        return summary
    
    def save_results(self, output_path: str = "test_results.json"):
        """Guarda los resultados en un archivo JSON"""
        def to_json_safe(obj: Any):
            """Convierte objetos numpy a tipos nativos serializables por JSON."""
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json_safe(v) for v in obj]
            return obj

        results_data = []
        for result in self.results:
            results_data.append({
                'test_id': result.test_id,
                'model_name': result.model_name,
                'model_params': result.model_params,
                'noise_type': result.noise_type,
                'noise_level': result.noise_level,
                'frequency': result.frequency,
                'success': result.success,
                'error_message': result.error_message,
                'metrics': to_json_safe(result.metrics) if result.metrics is not None else None,
                'execution_time': result.execution_time
            })
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Resultados guardados en: {output_path}")


def main():
    """Función principal para ejecutar la suite de tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Suite de Tests para Modelos de Detección de Anomalías')
    parser.add_argument('--config', default='test_config.json', 
                       help='Archivo de configuración JSON')
    parser.add_argument('--models', nargs='+', 
                       choices=['knn', 'mlp', 'random_forest', 'xgboost', 
                               'isolation_forest', 'lof', 'elliptic_envelope', 
                               'autoencoder', 'pca_iforest'],
                       help='Modelos específicos a ejecutar (opcional)')
    parser.add_argument('--workers', type=int, 
                       help='Número de workers para paralelización')
    parser.add_argument('--output', default='test_results.json',
                       help='Archivo de salida para los resultados')
    
    args = parser.parse_args()
    
    # Crear y ejecutar test runner
    runner = TestRunner(args.config)
    
    print("Iniciando suite de tests...")
    start_time = time.time()
    
    results = runner.run_tests(
        selected_models=args.models,
        max_workers=args.workers
    )
    
    total_time = time.time() - start_time
    
    # Generar reporte resumen
    summary = runner.generate_summary_report()
    print(f"\n{'='*50}")
    print("RESUMEN DE TESTS")
    print(f"{'='*50}")
    print(f"Total de tests: {summary['total_tests']}")
    print(f"Tests exitosos: {summary['successful_tests']}")
    print(f"Tests fallidos: {summary['failed_tests']}")
    print(f"Tasa de éxito: {summary['success_rate']:.2%}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    
    # Mostrar estadísticas por modelo
    print(f"\n{'='*50}")
    print("ESTADÍSTICAS POR MODELO")
    print(f"{'='*50}")
    for model_name, stats in summary['model_statistics'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Tests exitosos: {stats['successful_tests']}/{stats['total_tests']}")
        print(f"  Mejor accuracy: {stats['best_accuracy']:.4f}")
        print(f"  Tiempo promedio: {stats['avg_execution_time']:.2f}s")
    
    # Guardar resultados
    runner.save_results(args.output)
    
    # Mostrar tests fallidos si los hay
    if summary['failed_test_details']:
        print(f"\n{'='*50}")
        print("TESTS FALLIDOS")
        print(f"{'='*50}")
        for failed in summary['failed_test_details']:
            print(f"{failed['test_id']}: {failed['model']} - {failed['error']}")


if __name__ == "__main__":
    main()
