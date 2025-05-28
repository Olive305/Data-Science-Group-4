import pandas as pd
import statsmodels.formula.api as smf
from data_extraction.utils import load_excel

# Load data and drop first index column
df = load_excel('../data/fm_dem_merged.xlsx')
df = df.drop(df.columns[0], axis=1)

# Convert 'Krankenkasse' to categorical if not already
df['Krankenkasse'] = df['Krankenkasse'].astype('category')

# Define independent variable (IV) and dependent variable (DV)
iv = 'ZB_diff'  # Zusatzbeitragsänderung
dv = 'Mitglieder_diff_next'  # Mitgliederveränderung

# List all potential moderators by excluding known columns
exclude_cols = [dv, iv, 'Krankenkasse']
potential_moderators = [col for col in df.columns if col not in exclude_cols]

# Store results for later filtering
results = []

for mod in potential_moderators:
    try:
        # Build formula with interaction term
        formula = f"{dv} ~ {iv} + {mod} + {iv}:{mod}"

        # Fit mixed effects model with random intercept and slope for iv per KK
        model = smf.mixedlm(formula, df, groups=df['Krankenkasse'], re_formula=f'~{iv}')
        result = model.fit(reml=True)

        # Extract p-value of interaction term, fallback to 1 if not found
        pval = result.pvalues.get(f'{iv}:{mod}', 1.0)

        # Store moderator and p-value
        results.append((mod, pval))

        print(f"Tested moderator: {mod}, Interaction p-value: {pval:.4f}")

    except Exception as e:
        print(f"Failed to fit model with moderator {mod}: {e}")

# Convert results to DataFrame for easier sorting and filtering
results_df = pd.DataFrame(results, columns=['Moderator', 'Interaction_pvalue'])
results_df = results_df.sort_values('Interaction_pvalue')

print("\nModerators sorted by interaction p-value:")
print(results_df)

# Optionally save to CSV for further inspection
#results_df.to_csv('moderator_screening_results.csv', index=False)
