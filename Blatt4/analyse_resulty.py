import pandas as pd

df = pd.read_csv('mlp_results.csv', quotechar='"', sep = ';')

group_columns = ['max_iter', 'learning_rate', 'layers', 'solver']

with open('averages.txt', 'w') as f:
    for cols in group_columns:
        df_copy = df.copy(deep=True)
        for c in group_columns:
            if c!=cols:
                del df_copy[c]
        grouped = df_copy.groupby(cols).mean()
        
        f.write(f"\nGrouped by: {cols}\n")
        f.write("===================================\n")
        
        for index, row in grouped.iterrows():
            f.write(f"{cols}={index}\n")
            f.write(f"Average accuracy: {row['accuracy']}\n")
            f.write(f"Average precision: {row['precision']}\n")
            f.write(f"Average recall: {row['recall']}\n")
            f.write(f"Average F1-score: {row['f1_score']}\n\n")