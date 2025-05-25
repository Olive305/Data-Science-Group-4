import pandas as pd
from rapidfuzz import process, fuzz

from data_extraction.data_extractor import find_demographics
from data_extraction.data_merging import fuz_combine_fees_morbidity
from data_extraction.utils import basic_data_cleanup

#loading data
df_fm = fuz_combine_fees_morbidity()
df_dem = find_demographics()

df_dem['Krankenkasse_clean'] =basic_data_cleanup(df_dem['Krankenkasse'])

fm_names = df_fm["Krankenkasse"].unique()

#mapping
mapping = []

for name_dem in df_dem["Krankenkasse_clean"].unique():
    match, score, _ = process.extractOne(name_dem, fm_names, scorer=fuzz.token_sort_ratio)
    if score >= 75:
        mapping.append({
            "Name_dem": name_dem,
            "Name_fm": match
        })
    else:

        mapping.append({
            "Name_dem": name_dem,
            "Name_fm": ""
        })

df_mapping = pd.DataFrame(mapping)

manual_fixes = {
    "hkkkrankenkasse" : "hkk",
    "kkh" : "kaufmännischekrankenkasse(kkh)",
    "mhpluskrankenkasse": "mhplusbkk",
    "mobilkrankenkasse":"betriebskrankenkassemobil",
    "r+vbkk":"r+vbetriebskrankenkasse",
    "tk": "technikerkrankenkasse(tk)"
}


df_mapping["Name_fm"] = df_mapping.apply(
    lambda row: manual_fixes[row["Name_dem"]]
    if row["Name_dem"] in manual_fixes else row["Name_fm"],
    axis=1
)

gemappte_namen = df_mapping["Name_fm"].unique()
alle_namen_fm = df_fm["Krankenkasse"].unique()

nicht_gemappte = [name for name in alle_namen_fm if name not in gemappte_namen]


sonstige_zeilen = pd.DataFrame({
    "Name_dem": ["sonstigegkv"] * len(nicht_gemappte),
    "Name_fm": nicht_gemappte
})


df_mapping = pd.concat([df_mapping, sonstige_zeilen], ignore_index=True)
df_mapping = df_mapping.dropna(subset=['Name_fm'])

df_mapping = df_mapping.dropna(subset=["Name_fm"])


df_mapping.to_excel("../data/matching_tabelle.xlsx", index=False)
