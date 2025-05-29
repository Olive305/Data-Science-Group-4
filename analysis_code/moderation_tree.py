import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from data_extraction.utils import load_excel

# Load merged dataset
df = load_excel('../data/fm_dem_merged.xlsx')

# Drop unnamed first column if present (index from Excel export)
if df.columns[0].startswith('Unnamed'):
    df = df.drop(df.columns[0], axis=1)

# Ensure group variable is categorical
df['Krankenkasse'] = df['Krankenkasse'].astype('category')

# Define independent and dependent variables
iv = 'ZB_diff'  # Zusatzbeitragsänderung
dv = 'Mitglieder_diff_next'  # Mitgliederveränderung

# Exclude variables that are not potential moderators
exclude_cols = [dv, iv, 'Krankenkasse']
potential_moderators = [col for col in df.columns if col not in exclude_cols]

# Standardize all moderators in one batch to avoid fragmentation
scaler = StandardScaler()
moderator_data = df[potential_moderators].select_dtypes(include='number')  # only numeric moderators
z_moderators = pd.DataFrame(
    scaler.fit_transform(moderator_data),
    columns=[mod + '_z' for mod in moderator_data.columns],
    index=df.index
)
df = pd.concat([df, z_moderators], axis=1)
df = df.copy()  # defragment memory

# Store model results
results = []

for mod in moderator_data.columns:
    mod_z = mod + '_z'
    formula = f"{dv} ~ {iv} + {mod_z} + {iv}:{mod_z}"
    try:
        model = smf.mixedlm(formula, df, groups=df['Krankenkasse'], re_formula=f'~{iv}')
        result = model.fit(reml=True)

        # Extract interaction p-value; default to NaN if missing
        pval = result.pvalues.get(f"{iv}:{mod_z}", float('nan'))
        results.append((mod, pval))

        print(f"Tested moderator: {mod}, interaction p-value: {pval:.4e}")
    except Exception as e:
        print(f"Failed to fit model with moderator {mod}: {e}")
        results.append((mod, float('nan')))

# Build result DataFrame
results_df = pd.DataFrame(results, columns=['Moderator', 'Interaction_pvalue'])

# Apply FDR correction (Benjamini-Hochberg)
valid_mask = results_df['Interaction_pvalue'].notna()
pvals = results_df.loc[valid_mask, 'Interaction_pvalue'].values

# Only correct if any p-values exist
if len(pvals) > 0:
    _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    results_df.loc[valid_mask, 'p_adj'] = p_adj
else:
    results_df['p_adj'] = float('nan')

# Sort by corrected p-value
results_df = results_df.sort_values('p_adj', na_position='last')
significant_moderators = results_df[(results_df['p_adj'].notna()) & (results_df['p_adj'] < 0.05)]

print("\nModerators sorted by FDR-corrected p-value:")
print(significant_moderators)


# Optional: Save to CSV
# results_df.to_csv('moderator_screening_results_fdr.csv', index=False)
