#!/usr/bin/env python3
"""
Script principal para ejecutar la suite de tests
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.test_suite.test_runner import TestRunner
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Suite de Tests para Modelos de Detección de Anomalías')
    parser.add_argument('--config', default='test_config.json', 
                       help='Archivo de configuración JSON (default: test_config.json)')
    parser.add_argument('--models', nargs='+', 
                       choices=['knn', 'mlp', 'random_forest', 'xgboost', 
                               'isolation_forest', 'lof', 'elliptic_envelope', 
                               'autoencoder', 'pca_iforest'],
                       help='Modelos específicos a ejecutar (opcional)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Número de workers para paralelización (default: 4)')
    parser.add_argument('--output', default='test_results.json',
                       help='Archivo de salida para los resultados (default: test_results.json)')
    parser.add_argument('--quick', action='store_true',
                       help='Ejecutar solo un subconjunto rápido de tests')
    
    args = parser.parse_args()
    
    # Crear configuración rápida si se solicita
    if args.quick:
        print("Modo rápido: ejecutando solo KNN y MLP...")
        quick_config = {
            "models": {
                "knn": {
                    "enabled": True
                },
                "mlp": {
                    "enabled": True
                }
            },
            "noise_types": {
                "RUIDO": [1.0, 5.0],
                "SPURIA": [5.0, 20.0]
            },
            "frequencies": ["SMA"],
            "execution": {
                "max_workers": args.workers,
                "save_models": True,
                "generate_reports": True,
                "verbose": True
            }
        }
        
        import json
        with open('quick_test_config.json', 'w') as f:
            json.dump(quick_config, f, indent=2)
        args.config = 'quick_test_config.json'
    
    try:
        # Crear y ejecutar test runner
        runner = TestRunner(args.config)
        
        print("Iniciando suite de tests...")
        print(f"Configuración: {args.config}")
        print(f"Workers: {args.workers}")
        if args.models:
            print(f"Modelos seleccionados: {', '.join(args.models)}")
        print("-" * 50)
        
        start_time = time.time()
        
        results = runner.run_tests(
            selected_models=args.models,
            max_workers=args.workers
        )
        
        total_time = time.time() - start_time
        
        # Generar reporte resumen
        summary = runner.generate_summary_report()
        print(f"\n{'='*60}")
        print("RESUMEN DE TESTS")
        print(f"{'='*60}")
        print(f"Total de tests: {summary['total_tests']}")
        print(f"Tests exitosos: {summary['successful_tests']}")
        print(f"Tests fallidos: {summary['failed_tests']}")
        print(f"Tasa de éxito: {summary['success_rate']:.2%}")
        print(f"Tiempo total: {total_time:.2f} segundos")
        print(f"Tiempo promedio por test: {total_time/len(results):.2f} segundos")
        
        # Mostrar estadísticas por modelo
        print(f"\n{'='*60}")
        print("ESTADÍSTICAS POR MODELO")
        print(f"{'='*60}")
        for model_name, stats in summary['model_statistics'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Tests exitosos: {stats['successful_tests']}/{stats['total_tests']}")
            if stats['best_accuracy'] > 0:
                print(f"  Mejor accuracy: {stats['best_accuracy']:.4f}")
            print(f"  Tiempo promedio: {stats['avg_execution_time']:.2f}s")
        
        # Guardar resultados
        runner.save_results(args.output)
        print(f"\nResultados guardados en: {args.output}")
        
        # Mostrar tests fallidos si los hay
        if summary['failed_test_details']:
            print(f"\n{'='*60}")
            print("TESTS FALLIDOS")
            print(f"{'='*60}")
            for failed in summary['failed_test_details'][:10]:  # Mostrar solo los primeros 10
                print(f"{failed['test_id']}: {failed['model']} - {failed['error']}")
            if len(summary['failed_test_details']) > 10:
                print(f"... y {len(summary['failed_test_details']) - 10} más")
        
        print(f"\n{'='*60}")
        print("SUITE DE TESTS COMPLETADA")
        print(f"{'='*60}")
        
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de configuración: {e}")
        print("Asegúrate de que test_config.json existe en el directorio actual.")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)
    finally:
        # Limpiar archivo temporal si existe
        if args.quick and os.path.exists('quick_test_config.json'):
            os.remove('quick_test_config.json')

if __name__ == "__main__":
    main()
