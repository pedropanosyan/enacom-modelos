#!/usr/bin/env python3
"""
Script to generate summary tables from model evaluation reports.
Reads all report files and creates markdown tables summarizing performance metrics.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

def parse_report_file(file_path: str) -> Dict[str, Any]:
    """Parse a single report file and extract key metrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract basic info
    model_match = re.search(r'# Reporte de Evaluación \((.+?)\):', content)
    frequency_match = re.search(r'- \*\*Frecuencia\*\*: (.+)', content)
    test_id_match = re.search(r'- \*\*ID del Test\*\*: (.+)', content)
    noise_type_match = re.search(r'- \*\*Tipo de Ruido\*\*: (.+)', content)
    noise_level_match = re.search(r'- \*\*Nivel de Ruido\*\*: (.+)', content)
    
    # Extract key metrics
    precision_match = re.search(r'- \*\*Precisión \(Precision\)\*\*: (.+)', content)
    recall_match = re.search(r'- \*\*Recall\*\*: (.+)', content)
    f1_match = re.search(r'- \*\*F1-Score\*\*: (.+)', content)
    accuracy_match = re.search(r'- \*\*Exactitud \(Accuracy\)\*\*: (.+)', content)
    
    # Extract confusion matrix data
    cm_pattern = r'\|\| \*\*Real: Normal \(0\)\*\* \| (\d+) \| (\d+) \|'
    cm_normal = re.search(cm_pattern, content)
    
    cm_pattern2 = r'\|\| \*\*Real: Anomalía \(1\)\*\* \| (\d+) \| (\d+) \|'
    cm_anomaly = re.search(cm_pattern2, content)
    
    # Extract detected vs real anomalies
    detected_match = re.search(r'- \*\*Número de Anomalías Detectadas\*\*: (.+)', content)
    real_match = re.search(r'- \*\*Número de Anomalías Reales\*\*: (.+)', content)
    
    return {
        'model': model_match.group(1) if model_match else 'Unknown',
        'frequency': frequency_match.group(1) if frequency_match else 'Unknown',
        'test_id': test_id_match.group(1) if test_id_match else 'Unknown',
        'noise_type': noise_type_match.group(1) if noise_type_match else 'Unknown',
        'noise_level': float(noise_level_match.group(1)) if noise_level_match else 0.0,
        'precision': float(precision_match.group(1)) if precision_match else 0.0,
        'recall': float(recall_match.group(1)) if recall_match else 0.0,
        'f1_score': float(f1_match.group(1)) if f1_match else 0.0,
        'accuracy': float(accuracy_match.group(1)) if accuracy_match else 0.0,
        'tn': int(cm_normal.group(1)) if cm_normal else 0,  # True Negatives
        'fp': int(cm_normal.group(2)) if cm_normal else 0,  # False Positives
        'fn': int(cm_anomaly.group(1)) if cm_anomaly else 0,  # False Negatives
        'tp': int(cm_anomaly.group(2)) if cm_anomaly else 0,  # True Positives
        'detected_anomalies': int(detected_match.group(1)) if detected_match else 0,
        'real_anomalies': float(real_match.group(1)) if real_match else 0.0,
        'file_path': file_path
    }

def get_all_reports(base_path: str) -> List[Dict[str, Any]]:
    """Get all report files and parse them."""
    reports = []
    
    # Supervised models
    supervised_models = ['knn', 'mlp', 'random_forest', 'xgboost']
    for model in supervised_models:
        model_path = os.path.join(base_path, 'reports', 'supervised', model)
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith('_report.md'):
                    file_path = os.path.join(model_path, file)
                    try:
                        report_data = parse_report_file(file_path)
                        reports.append(report_data)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
    
    # Unsupervised models
    unsupervised_models = ['isolation_forest', 'lof', 'autoencoder', 'pca_iforest']
    for model in unsupervised_models:
        model_path = os.path.join(base_path, 'reports', 'unsupervised', model)
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith('_report.md'):
                    file_path = os.path.join(model_path, file)
                    try:
                        report_data = parse_report_file(file_path)
                        reports.append(report_data)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
    
    return reports

def create_model_summary_table(reports: List[Dict[str, Any]], model_name: str) -> str:
    """Create a summary table for a specific model."""
    model_reports = [r for r in reports if r['model'] == model_name]
    
    if not model_reports:
        return f"# {model_name.upper()} Summary\n\nNo reports found for this model.\n"
    
    # Group by frequency and noise type
    df = pd.DataFrame(model_reports)
    
    # Create summary table
    summary_data = []
    
    for freq in ['FM', 'SMA']:
        freq_data = df[df['frequency'] == freq]
        
        for noise_type in ['RUIDO', 'SPURIA', 'DROPOUT', 'BLOCKING']:
            noise_data = freq_data[freq_data['noise_type'] == noise_type]
            
            if not noise_data.empty:
                # Calculate averages across all noise levels for this noise type
                avg_precision = noise_data['precision'].mean()
                avg_recall = noise_data['recall'].mean()
                avg_f1 = noise_data['f1_score'].mean()
                avg_accuracy = noise_data['accuracy'].mean()
                
                # Get best and worst performance
                best_f1_idx = noise_data['f1_score'].idxmax()
                worst_f1_idx = noise_data['f1_score'].idxmin()
                
                best_f1 = noise_data.loc[best_f1_idx, 'f1_score']
                worst_f1 = noise_data.loc[worst_f1_idx, 'f1_score']
                best_noise_level = noise_data.loc[best_f1_idx, 'noise_level']
                worst_noise_level = noise_data.loc[worst_f1_idx, 'noise_level']
                
                summary_data.append({
                    'Frequency': freq,
                    'Noise Type': noise_type,
                    'Avg Precision': f"{avg_precision:.4f}",
                    'Avg Recall': f"{avg_recall:.4f}",
                    'Avg F1-Score': f"{avg_f1:.4f}",
                    'Avg Accuracy': f"{avg_accuracy:.4f}",
                    'Best F1': f"{best_f1:.4f} (noise: {best_noise_level})",
                    'Worst F1': f"{worst_f1:.4f} (noise: {worst_noise_level})",
                    'Tests': len(noise_data)
                })
    
    if not summary_data:
        return f"# {model_name.upper()} Summary\n\nNo valid data found for this model.\n"
    
    # Create markdown table
    summary_df = pd.DataFrame(summary_data)
    
    markdown = f"# {model_name.upper()} Model Summary\n\n"
    markdown += "## Performance Summary by Frequency and Noise Type\n\n"
    markdown += summary_df.to_markdown(index=False)
    markdown += "\n\n"
    
    # Add detailed breakdown by noise level
    markdown += "## Detailed Performance by Noise Level\n\n"
    
    for freq in ['FM', 'SMA']:
        freq_data = df[df['frequency'] == freq]
        if not freq_data.empty:
            markdown += f"### {freq} Frequency\n\n"
            
            # Create detailed table
            detailed_data = []
            for _, row in freq_data.iterrows():
                detailed_data.append({
                    'Noise Type': row['noise_type'],
                    'Noise Level': row['noise_level'],
                    'Precision': f"{row['precision']:.4f}",
                    'Recall': f"{row['recall']:.4f}",
                    'F1-Score': f"{row['f1_score']:.4f}",
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'Test ID': row['test_id']
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df = detailed_df.sort_values(['Noise Type', 'Noise Level'])
            markdown += detailed_df.to_markdown(index=False)
            markdown += "\n\n"
    
    return markdown

def create_overall_comparison_table(reports: List[Dict[str, Any]]) -> str:
    """Create an overall comparison table across all models."""
    df = pd.DataFrame(reports)
    
    # Calculate average performance for each model
    model_summary = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        avg_precision = model_data['precision'].mean()
        avg_recall = model_data['recall'].mean()
        avg_f1 = model_data['f1_score'].mean()
        avg_accuracy = model_data['accuracy'].mean()
        
        # Calculate standard deviation
        std_precision = model_data['precision'].std()
        std_recall = model_data['recall'].std()
        std_f1 = model_data['f1_score'].std()
        std_accuracy = model_data['accuracy'].std()
        
        # Best and worst performance
        best_f1 = model_data['f1_score'].max()
        worst_f1 = model_data['f1_score'].min()
        
        model_summary.append({
            'Model': model,
            'Avg Precision': f"{avg_precision:.4f} ± {std_precision:.4f}",
            'Avg Recall': f"{avg_recall:.4f} ± {std_recall:.4f}",
            'Avg F1-Score': f"{avg_f1:.4f} ± {std_f1:.4f}",
            'Avg Accuracy': f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
            'Best F1': f"{best_f1:.4f}",
            'Worst F1': f"{worst_f1:.4f}",
            'Total Tests': len(model_data)
        })
    
    summary_df = pd.DataFrame(model_summary)
    summary_df = summary_df.sort_values('Avg F1-Score', ascending=False)
    
    markdown = "# Overall Model Performance Comparison\n\n"
    markdown += "## Average Performance Across All Tests\n\n"
    markdown += summary_df.to_markdown(index=False)
    markdown += "\n\n"
    
    # Add performance by frequency
    markdown += "## Performance by Frequency\n\n"
    
    freq_summary = []
    for freq in ['FM', 'SMA']:
        freq_data = df[df['frequency'] == freq]
        if not freq_data.empty:
            for model in freq_data['model'].unique():
                model_freq_data = freq_data[freq_data['model'] == model]
                avg_f1 = model_freq_data['f1_score'].mean()
                avg_accuracy = model_freq_data['accuracy'].mean()
                
                freq_summary.append({
                    'Model': model,
                    'Frequency': freq,
                    'Avg F1-Score': f"{avg_f1:.4f}",
                    'Avg Accuracy': f"{avg_accuracy:.4f}",
                    'Tests': len(model_freq_data)
                })
    
    if freq_summary:
        freq_df = pd.DataFrame(freq_summary)
        freq_df = freq_df.sort_values(['Frequency', 'Avg F1-Score'], ascending=[True, False])
        markdown += freq_df.to_markdown(index=False)
        markdown += "\n\n"
    
    return markdown

def main():
    """Main function to generate all summary tables."""
    base_path = "/Users/pedropanosyan/Documents/Austral/enacom/models"
    
    print("Loading all reports...")
    reports = get_all_reports(base_path)
    print(f"Loaded {len(reports)} reports")
    
    if not reports:
        print("No reports found!")
        return
    
    # Get unique models
    models = list(set(r['model'] for r in reports))
    print(f"Found models: {models}")
    
    # Create output directory
    output_dir = os.path.join(base_path, "summary_tables")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate individual model summaries
    for model in models:
        print(f"Generating summary for {model}...")
        summary = create_model_summary_table(reports, model)
        
        output_file = os.path.join(output_dir, f"{model}_summary.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved {output_file}")
    
    # Generate overall comparison
    print("Generating overall comparison...")
    overall_summary = create_overall_comparison_table(reports)
    
    output_file = os.path.join(output_dir, "overall_comparison.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(overall_summary)
    print(f"Saved {output_file}")
    
    print("\nSummary generation complete!")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main()

