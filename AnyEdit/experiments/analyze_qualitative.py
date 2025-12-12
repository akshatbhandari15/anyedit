"""
Analyze qualitative results and generate CSV report + analysis.
"""
import json
import csv
from pathlib import Path
import pandas as pd


def load_results(results_dir):
    """Load all qualitative analysis JSON files."""
    results_dir = Path(results_dir)
    all_results = []
    
    for json_file in results_dir.glob("qualitative_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.append(data)
    
    return all_results


def create_comparison_csv(all_results, output_file):
    """Create a CSV comparing before/after responses for each model."""
    rows = []
    
    for result in all_results:
        model_name = result['model_name']
        model_short = result['hparams_fname'].replace('.json', '')
        
        # Get all prompt keys
        prompt_keys = result['before_edit'].keys()
        
        for key in prompt_keys:
            before_data = result['before_edit'][key]
            after_data = result['after_edit'][key]
            
            row = {
                'Model': model_short,
                'Prompt_ID': key,
                'Category': before_data.get('category', 'N/A'),
                'Question': before_data['question'][:150] + '...' if len(before_data['question']) > 150 else before_data['question'],
                'Ground_Truth': before_data.get('ground_truth', 'N/A')[:100] if before_data.get('ground_truth') else 'N/A',
                'Response_Before_Edit': before_data['response'][:300] + '...' if len(before_data['response']) > 300 else before_data['response'],
                'Response_After_Edit': after_data['response'][:300] + '...' if len(after_data['response']) > 300 else after_data['response'],
                'Response_Changed': 'Yes' if before_data['response'] != after_data['response'] else 'No',
            }
            rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Comparison CSV saved to: {output_file}")
    
    return df


def create_summary_csv(all_results, output_file):
    """Create a summary CSV with key statistics per model."""
    rows = []
    
    for result in all_results:
        model_short = result['hparams_fname'].replace('.json', '')
        
        # Count changes
        total_prompts = len(result['before_edit'])
        changed_prompts = 0
        
        for key in result['before_edit'].keys():
            if result['before_edit'][key]['response'] != result['after_edit'][key]['response']:
                changed_prompts += 1
        
        # Check specific categories
        edit_sample_changed = result['before_edit']['edit_sample']['response'] != result['after_edit']['edit_sample']['response']
        unke_random_changed = result['before_edit']['unke_random']['response'] != result['after_edit']['unke_random']['response']
        gsm8k_changed = result['before_edit']['gsm8k']['response'] != result['after_edit']['gsm8k']['response']
        tiramisu_changed = result['before_edit']['tiramisu_recipe']['response'] != result['after_edit']['tiramisu_recipe']['response']
        
        row = {
            'Model': model_short,
            'Edit_Time_Seconds': f"{result['edit_time']:.2f}",
            'Total_Prompts': total_prompts,
            'Responses_Changed': changed_prompts,
            'Change_Rate': f"{(changed_prompts/total_prompts)*100:.1f}%",
            'Edit_Sample_Changed': 'Yes' if edit_sample_changed else 'No',
            'UnKE_Random_Changed': 'Yes' if unke_random_changed else 'No',
            'GSM8K_Changed': 'Yes' if gsm8k_changed else 'No',
            'Tiramisu_Recipe_Changed': 'Yes' if tiramisu_changed else 'No',
            'Edit_Question': result['edit_info']['question'][:100] + '...' if len(result['edit_info']['question']) > 100 else result['edit_info']['question'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Summary CSV saved to: {output_file}")
    
    return df


def analyze_results(all_results, output_file):
    """Generate a detailed text analysis report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("QUALITATIVE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append(f"Total Models Analyzed: {len(all_results)}")
    report_lines.append("")
    
    for result in all_results:
        model_short = result['hparams_fname'].replace('.json', '')
        report_lines.append("="*80)
        report_lines.append(f"MODEL: {model_short}")
        report_lines.append("="*80)
        report_lines.append("")
        
        report_lines.append(f"Full Model Name: {result['model_name']}")
        report_lines.append(f"Edit Time: {result['edit_time']:.2f} seconds")
        report_lines.append("")
        
        report_lines.append("Edit Information:")
        report_lines.append(f"  Question: {result['edit_info']['question']}")
        report_lines.append(f"  Expected Answer: {result['edit_info'].get('answer', 'N/A')}")
        report_lines.append("")
        
        # Analyze each prompt type
        report_lines.append("Response Analysis:")
        report_lines.append("-"*80)
        
        total_prompts = len(result['before_edit'])
        changed_count = 0
        
        for key in result['before_edit'].keys():
            before = result['before_edit'][key]
            after = result['after_edit'][key]
            changed = before['response'] != after['response']
            if changed:
                changed_count += 1
            
            report_lines.append(f"\n{key.upper()} - {before.get('category', 'N/A')}")
            report_lines.append(f"  Changed: {'YES ✓' if changed else 'NO'}")
            report_lines.append(f"  Question: {before['question'][:100]}...")
            
            if key == 'edit_sample':
                report_lines.append(f"  >>> This is the EDITED sample - should change!")
                report_lines.append(f"  Before: {before['response'][:200]}...")
                report_lines.append(f"  After:  {after['response'][:200]}...")
            elif changed:
                report_lines.append(f"  Before: {before['response'][:150]}...")
                report_lines.append(f"  After:  {after['response'][:150]}...")
        
        report_lines.append("")
        report_lines.append("-"*80)
        report_lines.append(f"Summary: {changed_count}/{total_prompts} responses changed ({(changed_count/total_prompts)*100:.1f}%)")
        report_lines.append("")
        
        # Key findings
        report_lines.append("Key Findings:")
        edit_changed = result['before_edit']['edit_sample']['response'] != result['after_edit']['edit_sample']['response']
        unke_changed = result['before_edit']['unke_random']['response'] != result['after_edit']['unke_random']['response']
        gsm8k_changed = result['before_edit']['gsm8k']['response'] != result['after_edit']['gsm8k']['response']
        tiramisu_changed = result['before_edit']['tiramisu_recipe']['response'] != result['after_edit']['tiramisu_recipe']['response']
        
        if edit_changed:
            report_lines.append("  ✓ Edit sample response changed (EXPECTED)")
        else:
            report_lines.append("  ✗ Edit sample response DID NOT change (UNEXPECTED - edit may have failed)")
        
        if not unke_changed:
            report_lines.append("  ✓ Random UnKE sample unchanged (GOOD - no unintended side effects)")
        else:
            report_lines.append("  ⚠ Random UnKE sample changed (possible side effect)")
        
        if not gsm8k_changed:
            report_lines.append("  ✓ GSM8K math reasoning unchanged (GOOD - no capability degradation)")
        else:
            report_lines.append("  ⚠ GSM8K response changed (possible capability impact)")
        
        if not tiramisu_changed:
            report_lines.append("  ✓ Creative generation unchanged (GOOD - no capability degradation)")
        else:
            report_lines.append("  ⚠ Creative generation changed (possible capability impact)")
        
        report_lines.append("")
    
    # Overall findings
    report_lines.append("="*80)
    report_lines.append("OVERALL FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Count how many models successfully edited
    successful_edits = sum(1 for r in all_results if r['before_edit']['edit_sample']['response'] != r['after_edit']['edit_sample']['response'])
    report_lines.append(f"Successful Edits: {successful_edits}/{len(all_results)} models")
    
    # Count side effects
    unke_side_effects = sum(1 for r in all_results if r['before_edit']['unke_random']['response'] != r['after_edit']['unke_random']['response'])
    gsm8k_side_effects = sum(1 for r in all_results if r['before_edit']['gsm8k']['response'] != r['after_edit']['gsm8k']['response'])
    creative_side_effects = sum(1 for r in all_results if r['before_edit']['tiramisu_recipe']['response'] != r['after_edit']['tiramisu_recipe']['response'])
    
    report_lines.append(f"UnKE Side Effects: {unke_side_effects}/{len(all_results)} models")
    report_lines.append(f"Math Reasoning Side Effects: {gsm8k_side_effects}/{len(all_results)} models")
    report_lines.append(f"Creative Generation Side Effects: {creative_side_effects}/{len(all_results)} models")
    report_lines.append("")
    
    report_lines.append("Interpretation:")
    report_lines.append("  - Ideal: Edit sample changes, all others remain unchanged")
    report_lines.append("  - Good: Edit sample changes, minimal side effects (<30%)")
    report_lines.append("  - Concerning: Edit fails OR major side effects (>50%)")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    # Save report
    report_text = '\n'.join(report_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Analysis report saved to: {output_file}")
    print("\n" + report_text)
    
    return report_text


def main():
    results_dir = "output/qualitative_analysis_run2"
    
    print("Loading qualitative analysis results...")
    all_results = load_results(results_dir)
    print(f"✓ Loaded {len(all_results)} model results")
    print("")
    
    # Create output directory
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CSVs
    print("Generating CSV files...")
    comparison_df = create_comparison_csv(all_results, output_dir / "qualitative_comparison.csv")
    summary_df = create_summary_csv(all_results, output_dir / "qualitative_summary.csv")
    print("")
    
    # Generate analysis report
    print("Generating analysis report...")
    analyze_results(all_results, output_dir / "qualitative_analysis_report.txt")
    print("")
    
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Files saved to: {output_dir}/")
    print("  - qualitative_comparison.csv (detailed comparison)")
    print("  - qualitative_summary.csv (summary statistics)")
    print("  - qualitative_analysis_report.txt (detailed analysis)")
    print("="*80)


if __name__ == "__main__":
    main()
