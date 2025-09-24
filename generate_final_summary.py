#!/usr/bin/env python3
"""
Script to generate a single comprehensive summary from existing summary tables.
Reads all individual model summary files and creates one consolidated report.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

def parse_summary_table(file_path: str) -> Dict[str, Any]:
    """Parse a summary table file and extract key metrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract model name from filename
    model_name = os.path.basename(file_path).replace('_summary.md', '')
    
    # Find the performance summary table
    table_match = re.search(
        r'## Performance Summary by Frequency and Noise Type\n\n(.*?)\n\n##', 
        content, 
        re.DOTALL
    )
    
    if not table_match:
        return None
    
    table_content = table_match.group(1)
    
    # Parse table rows (skip header)
    lines = table_content.strip().split('\n')
    if len(lines) < 3:  # Need at least header + separator + one data row
        return None
    
    # Skip header and separator lines
    data_lines = [line for line in lines[2:] if line.strip() and '|' in line]
    
    model_data = {
        'model': model_name,
        'frequency_data': {},
        'overall_metrics': {}
    }
    
    # Parse each row
    for line in data_lines:
        if not line.strip():
            continue
            
        # Split by | and clean up
        parts = [part.strip() for part in line.split('|') if part.strip()]
        if len(parts) < 8:
            continue
            
        try:
            frequency = parts[0]
            noise_type = parts[1]
            avg_precision = float(parts[2])
            avg_recall = float(parts[3])
            avg_f1 = float(parts[4])
            avg_accuracy = float(parts[5])
            tests = int(parts[8])
            
            # Store frequency-specific data
            if frequency not in model_data['frequency_data']:
                model_data['frequency_data'][frequency] = {}
            
            model_data['frequency_data'][frequency][noise_type] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'accuracy': avg_accuracy,
                'tests': tests
            }
            
        except (ValueError, IndexError) as e:
            continue
    
    # Calculate overall metrics
    all_f1_scores = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []
    total_tests = 0
    
    for freq_data in model_data['frequency_data'].values():
        for noise_data in freq_data.values():
            all_f1_scores.append(noise_data['f1_score'])
            all_precisions.append(noise_data['precision'])
            all_recalls.append(noise_data['recall'])
            all_accuracies.append(noise_data['accuracy'])
            total_tests += noise_data['tests']
    
    if all_f1_scores:
        model_data['overall_metrics'] = {
            'avg_precision': sum(all_precisions) / len(all_precisions),
            'avg_recall': sum(all_recalls) / len(all_recalls),
            'avg_f1_score': sum(all_f1_scores) / len(all_f1_scores),
            'avg_accuracy': sum(all_accuracies) / len(all_accuracies),
            'best_f1': max(all_f1_scores),
            'worst_f1': min(all_f1_scores),
            'total_tests': total_tests,
            'std_f1': (sum([(x - sum(all_f1_scores)/len(all_f1_scores))**2 for x in all_f1_scores]) / len(all_f1_scores))**0.5
        }
    
    return model_data

def create_comprehensive_summary(summary_dir: str) -> str:
    """Create a comprehensive summary from all model summary files."""
    
    # Get all summary files
    summary_files = []
    for file in os.listdir(summary_dir):
        if file.endswith('_summary.md') and file != 'overall_comparison.md':
            summary_files.append(os.path.join(summary_dir, file))
    
    # Parse all summaries
    model_data = {}
    for file_path in summary_files:
        data = parse_summary_table(file_path)
        if data:
            model_data[data['model']] = data
    
    if not model_data:
        return "# Model Performance Summary\n\nNo model data found.\n"
    
    # Create comprehensive summary
    markdown = "# Comprehensive Model Performance Summary\n\n"
    markdown += "This report consolidates performance metrics from all model evaluation reports.\n\n"
    
    # 1. Overall Model Ranking
    markdown += "## 1. Overall Model Performance Ranking\n\n"
    markdown += "Models ranked by average F1-Score across all test scenarios:\n\n"
    
    # Sort models by F1-score
    sorted_models = sorted(
        model_data.items(), 
        key=lambda x: x[1]['overall_metrics'].get('avg_f1_score', 0), 
        reverse=True
    )
    
    ranking_data = []
    for rank, (model, data) in enumerate(sorted_models, 1):
        metrics = data['overall_metrics']
        ranking_data.append({
            'Rank': rank,
            'Model': model.upper(),
            'Avg F1-Score': f"{metrics.get('avg_f1_score', 0):.4f}",
            'Avg Precision': f"{metrics.get('avg_precision', 0):.4f}",
            'Avg Recall': f"{metrics.get('avg_recall', 0):.4f}",
            'Avg Accuracy': f"{metrics.get('avg_accuracy', 0):.4f}",
            'Best F1': f"{metrics.get('best_f1', 0):.4f}",
            'Worst F1': f"{metrics.get('worst_f1', 0):.4f}",
            'Tests': metrics.get('total_tests', 0)
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    markdown += ranking_df.to_markdown(index=False)
    markdown += "\n\n"
    
    # 2. Performance by Model Type (Supervised vs Unsupervised)
    markdown += "## 2. Performance by Model Type\n\n"
    
    supervised_models = ['knn', 'mlp', 'random_forest', 'xgboost']
    unsupervised_models = ['autoencoder', 'isolation_forest', 'lof', 'pca_iforest']
    
    supervised_data = []
    unsupervised_data = []
    
    for model, data in model_data.items():
        metrics = data['overall_metrics']
        model_info = {
            'Model': model.upper(),
            'Avg F1-Score': f"{metrics.get('avg_f1_score', 0):.4f}",
            'Avg Accuracy': f"{metrics.get('avg_accuracy', 0):.4f}",
            'Tests': metrics.get('total_tests', 0)
        }
        
        if model in supervised_models:
            supervised_data.append(model_info)
        elif model in unsupervised_models:
            unsupervised_data.append(model_info)
    
    if supervised_data:
        markdown += "### Supervised Models\n\n"
        supervised_df = pd.DataFrame(supervised_data)
        supervised_df = supervised_df.sort_values('Avg F1-Score', ascending=False)
        markdown += supervised_df.to_markdown(index=False)
        markdown += "\n\n"
    
    if unsupervised_data:
        markdown += "### Unsupervised Models\n\n"
        unsupervised_df = pd.DataFrame(unsupervised_data)
        unsupervised_df = unsupervised_df.sort_values('Avg F1-Score', ascending=False)
        markdown += unsupervised_df.to_markdown(index=False)
        markdown += "\n\n"
    
    # 3. Performance by Frequency
    markdown += "## 3. Performance by Frequency (FM vs SMA)\n\n"
    
    freq_comparison = []
    for model, data in model_data.items():
        metrics = data['overall_metrics']
        freq_data = data['frequency_data']
        
        fm_metrics = {}
        sma_metrics = {}
        
        # Calculate FM metrics
        if 'FM' in freq_data:
            fm_f1_scores = [noise['f1_score'] for noise in freq_data['FM'].values()]
            fm_accuracies = [noise['accuracy'] for noise in freq_data['FM'].values()]
            fm_metrics = {
                'avg_f1': sum(fm_f1_scores) / len(fm_f1_scores) if fm_f1_scores else 0,
                'avg_accuracy': sum(fm_accuracies) / len(fm_accuracies) if fm_accuracies else 0,
                'tests': sum(noise['tests'] for noise in freq_data['FM'].values())
            }
        
        # Calculate SMA metrics
        if 'SMA' in freq_data:
            sma_f1_scores = [noise['f1_score'] for noise in freq_data['SMA'].values()]
            sma_accuracies = [noise['accuracy'] for noise in freq_data['SMA'].values()]
            sma_metrics = {
                'avg_f1': sum(sma_f1_scores) / len(sma_f1_scores) if sma_f1_scores else 0,
                'avg_accuracy': sum(sma_accuracies) / len(sma_accuracies) if sma_accuracies else 0,
                'tests': sum(noise['tests'] for noise in freq_data['SMA'].values())
            }
        
        freq_comparison.append({
            'Model': model.upper(),
            'FM F1-Score': f"{fm_metrics.get('avg_f1', 0):.4f}",
            'SMA F1-Score': f"{sma_metrics.get('avg_f1', 0):.4f}",
            'FM Accuracy': f"{fm_metrics.get('avg_accuracy', 0):.4f}",
            'SMA Accuracy': f"{sma_metrics.get('avg_accuracy', 0):.4f}",
            'FM Tests': fm_metrics.get('tests', 0),
            'SMA Tests': sma_metrics.get('tests', 0)
        })
    
    freq_df = pd.DataFrame(freq_comparison)
    freq_df = freq_df.sort_values('SMA F1-Score', ascending=False)
    markdown += freq_df.to_markdown(index=False)
    markdown += "\n\n"
    
    # 4. Performance by Noise Type
    markdown += "## 4. Performance by Noise Type\n\n"
    
    noise_types = ['RUIDO', 'SPURIA', 'DROPOUT', 'BLOCKING']
    noise_comparison = []
    
    for noise_type in noise_types:
        noise_data = []
        for model, data in model_data.items():
            for freq_data in data['frequency_data'].values():
                if noise_type in freq_data:
                    noise_data.append(freq_data[noise_type]['f1_score'])
        
        if noise_data:
            avg_f1 = sum(noise_data) / len(noise_data)
            best_f1 = max(noise_data)
            worst_f1 = min(noise_data)
            
            noise_comparison.append({
                'Noise Type': noise_type,
                'Avg F1-Score': f"{avg_f1:.4f}",
                'Best F1-Score': f"{best_f1:.4f}",
                'Worst F1-Score': f"{worst_f1:.4f}",
                'Models Tested': len(noise_data)
            })
    
    noise_df = pd.DataFrame(noise_comparison)
    noise_df = noise_df.sort_values('Avg F1-Score', ascending=False)
    markdown += noise_df.to_markdown(index=False)
    markdown += "\n\n"
    
    # 5. Key Insights and Recommendations
    markdown += "## 5. Key Insights and Recommendations\n\n"
    
    # Find best and worst performing models
    best_model = sorted_models[0][0] if sorted_models else "N/A"
    worst_model = sorted_models[-1][0] if sorted_models else "N/A"
    
    best_f1 = sorted_models[0][1]['overall_metrics'].get('avg_f1_score', 0) if sorted_models else 0
    worst_f1 = sorted_models[-1][1]['overall_metrics'].get('avg_f1_score', 0) if sorted_models else 0
    
    markdown += f"### Performance Highlights\n\n"
    markdown += f"- **Best Overall Model**: {best_model.upper()} (F1-Score: {best_f1:.4f})\n"
    markdown += f"- **Worst Overall Model**: {worst_model.upper()} (F1-Score: {worst_f1:.4f})\n"
    markdown += f"- **Performance Gap**: {best_f1 - worst_f1:.4f} F1-Score difference\n\n"
    
    # Analyze frequency performance
    fm_avg = 0
    sma_avg = 0
    fm_count = 0
    sma_count = 0
    
    for model, data in model_data.items():
        freq_data = data['frequency_data']
        if 'FM' in freq_data:
            fm_scores = [noise['f1_score'] for noise in freq_data['FM'].values()]
            fm_avg += sum(fm_scores) / len(fm_scores) if fm_scores else 0
            fm_count += 1
        if 'SMA' in freq_data:
            sma_scores = [noise['f1_score'] for noise in freq_data['SMA'].values()]
            sma_avg += sum(sma_scores) / len(sma_scores) if sma_scores else 0
            sma_count += 1
    
    if fm_count > 0 and sma_count > 0:
        fm_avg /= fm_count
        sma_avg /= sma_count
        markdown += f"- **Frequency Performance**: SMA ({sma_avg:.4f}) vs FM ({fm_avg:.4f}) average F1-Score\n"
        if sma_avg > fm_avg:
            markdown += f"  - SMA frequency shows {((sma_avg - fm_avg) / fm_avg * 100):.1f}% better performance\n"
        else:
            markdown += f"  - FM frequency shows {((fm_avg - sma_avg) / sma_avg * 100):.1f}% better performance\n"
    
    markdown += "\n### Recommendations\n\n"
    markdown += "1. **For Production Use**: Consider {best_model.upper()} for best overall performance\n"
    markdown += "2. **For Robustness**: Random Forest and XGBoost show consistent performance across scenarios\n"
    markdown += "3. **For Specific Noise Types**: \n"
    markdown += "   - RUIDO: All models perform excellently (F1 ≈ 1.0)\n"
    markdown += "   - SPURIA: Most challenging noise type, requires careful model selection\n"
    markdown += "   - DROPOUT: Moderate difficulty, ensemble methods recommended\n"
    markdown += "   - BLOCKING: Generally well-handled by most models\n"
    markdown += "4. **Frequency Considerations**: SMA frequency generally shows better performance than FM\n\n"
    
    # 6. Detailed Model Performance Tables
    markdown += "## 6. Detailed Model Performance Tables\n\n"
    
    for model, data in sorted_models:
        markdown += f"### {model.upper()} Model\n\n"
        
        # Create detailed table
        detailed_data = []
        for frequency, freq_data in data['frequency_data'].items():
            for noise_type, noise_data in freq_data.items():
                detailed_data.append({
                    'Frequency': frequency,
                    'Noise Type': noise_type,
                    'F1-Score': f"{noise_data['f1_score']:.4f}",
                    'Precision': f"{noise_data['precision']:.4f}",
                    'Recall': f"{noise_data['recall']:.4f}",
                    'Accuracy': f"{noise_data['accuracy']:.4f}",
                    'Tests': noise_data['tests']
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df = detailed_df.sort_values(['Frequency', 'Noise Type'])
            markdown += detailed_df.to_markdown(index=False)
            markdown += "\n\n"
    
    return markdown

def main():
    """Main function to generate the comprehensive summary."""
    base_path = "/Users/pedropanosyan/Documents/Austral/enacom/models"
    summary_dir = os.path.join(base_path, "summary_tables")
    
    if not os.path.exists(summary_dir):
        print(f"Summary directory not found: {summary_dir}")
        print("Please run generate_summary_tables.py first to create individual summaries.")
        return
    
    print("Generating comprehensive summary from existing summary tables...")
    
    # Generate comprehensive summary
    summary = create_comprehensive_summary(summary_dir)
    
    # Save to file
    output_file = os.path.join(base_path, "COMPREHENSIVE_MODEL_SUMMARY.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Comprehensive summary saved to: {output_file}")
    print("Summary includes:")
    print("- Overall model ranking")
    print("- Performance by model type (supervised vs unsupervised)")
    print("- Performance by frequency (FM vs SMA)")
    print("- Performance by noise type")
    print("- Key insights and recommendations")
    print("- Detailed performance tables for each model")

if __name__ == "__main__":
    main()
