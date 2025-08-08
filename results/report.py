import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_report(report_text):
    lines = [line.strip() for line in report_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        if line.lstrip().startswith('precision'):
            header_idx = i
            break

    rows = []
    for line in lines[header_idx+1:]:

        if not re.search(r'\d+\.\d+', line):
            continue
        
        parts = re.split(r'\s{2,}', line.strip())
        
        if len(parts) == 5:  # class_name + 4 metrics
            class_name, precision, recall, f1, support = parts
        elif len(parts) == 2:  # only a class_name and metrics
            class_name = parts[0]
            metrics = parts[1].strip().split()
            precision, recall, f1, support = metrics
        else:
            continue
        
        rows.append({'class': class_name,
                     'precision': float(precision),
                     'recall': float(recall),
                     'f1-score': float(f1),
                     'support': int(support)
                     })
    
    return pd.DataFrame(rows)

def load_and_convert(filepath):
    with open(filepath, 'r') as f:
        report_text = f.read()
    
    df = parse_report(report_text)
    return df

def analyze_class_groups(df):
    df_classes = df[~df['class'].str.contains('avg', case=False, na=False)]
    
    causative_df = df_classes[df_classes['class'].str.contains('causative', case=False)]
    potential_df = df_classes[df_classes['class'].str.contains('potential', case=False)]
    passive_df = df_classes[df_classes['class'].str.contains('passive', case=False)]
    other_df = df_classes[~(df_classes['class'].str.contains('causative', case=False) | 
                            df_classes['class'].str.contains('potential', case=False) | 
                            df_classes['class'].str.contains('passive', case=False)
                            )]
    
    results = []
    
    def calculate_metrics(group_df, name):
        if len(group_df) == 0:
            return None
            
        total_support = group_df['support'].sum()
        
        macro_precision = group_df['precision'].mean()
        macro_recall = group_df['recall'].mean()
        macro_f1 = group_df['f1-score'].mean()
        
        weighted_precision = (group_df['precision'] * group_df['support']).sum() / total_support
        weighted_recall = (group_df['recall'] * group_df['support']).sum() / total_support
        weighted_f1 = (group_df['f1-score'] * group_df['support']).sum() / total_support
        
        return {'Group': name,
                'Classes': len(group_df),
                'Total Support': total_support,
                # 'Macro Precision': round(macro_precision, 3),
                # 'Macro Recall': round(macro_recall, 3),
                'Macro F1': round(macro_f1, 3),
                # 'Weighted Precision': round(weighted_precision, 3),
                # 'Weighted Recall': round(weighted_recall, 3),
                'Weighted F1': round(weighted_f1, 3)
                }

    for group_df, name in [(causative_df, 'Causative'), 
                           (potential_df, 'Potential'), 
                           (passive_df, 'Passive'), 
                           (other_df, 'Other')
                           ]:
        metrics = calculate_metrics(group_df, name)
        if metrics:
            results.append(metrics)
    
    return pd.DataFrame(results)

def plot_metrics(df, output_path):
    df_classes = df[~df['class'].str.contains('avg', case=False, na=False)].copy()
    
    df_classes['category'] = 'Other'
    df_classes.loc[df_classes['class'].str.contains('causative', case=False), 'category'] = 'Causative'
    df_classes.loc[df_classes['class'].str.contains('potential', case=False), 'category'] = 'Potential'
    df_classes.loc[df_classes['class'].str.contains('passive', case=False), 'category'] = 'Passive'
    
    df_classes = df_classes.sort_values(by=['category'])
    colors = {'Causative': 'blue', 'Potential': 'green', 'Passive': 'orange', 'Other': 'gray'}
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    metrics = ['precision', 'recall', 'f1-score']
    titles = ['Precision', 'Recall', 'F1-Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        bar_positions = np.arange(len(df_classes))
        bar_colors = [colors[cat] for cat in df_classes['category']]
        bars = axs[i].bar(bar_positions, df_class/es[metric], color=bar_colors, alpha=0.7)
        
        axs[i].set_title(title, fontsize=30)
        axs[i].set_ylim([0, 1])
        axs[i].set_xticks([])
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        if i == 0:
            handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7) for color in colors.values()]
            labels = list(colors.keys())
            handles.append(plt.Line2D([0], [0], color='red', linestyle='--'))
            axs[i].legend(handles, labels, loc='upper right',prop={'size': 30})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

df = load_and_convert('models/baseline_model_evaluation.txt')
plot_metrics(df,"plots/baseline_metrics.png")
group_analysis = analyze_class_groups(df)
print("\nBaseline Group Analysis:")
print(group_analysis)

df = load_and_convert('models/bilstm_model_evaluation.txt')
plot_metrics(df,"plots/BiLSTM_metrics.png")
group_analysis = analyze_class_groups(df)
print("\nBiLSTM Group Analysis:")
print(group_analysis)

