import pandas as pd
from rapidfuzz import process, fuzz

from data_extraction.data_extractor import find_demo_24, find_demo_23

from data_extraction.utils import basic_data_cleanup, write_excel, load_excel

#loading data
df_fm = load_excel("../data/morb_fee_merged.xlsx")

def general_matching(df_dem):


    df_dem = basic_data_cleanup(df_dem, 'Krankenkasse')

    fm_names = df_fm["Krankenkasse"].unique()

    # mapping
    mapping = []

    for name_dem in df_dem["Krankenkasse"].unique():
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
    return df_mapping

def build_matching_24():
    df_dem = find_demo_24()
    df_mapping=general_matching(df_dem)
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

    mapped = df_mapping["Name_fm"].unique()
    un_names_fm = df_fm["Krankenkasse"].unique()

    unmapped = [name for name in un_names_fm if name not in mapped]

    #for plenty of KKs there are no specific data so it is mapped to "others"
    others = pd.DataFrame({
        "Name_dem": ["sonstigegkv"] * len(unmapped),
        "Name_fm": unmapped
    })


    df_mapping = pd.concat([df_mapping, others], ignore_index=True)

    #removes empty ones
    df_mapping = df_mapping.dropna(subset=['Name_fm'])
    df_mapping = df_mapping[df_mapping["Name_fm"].str.strip() != ""]
    df_mapping.rename(columns={"Name_dem": "Name_dem_24"}, inplace=True)
    return df_mapping

def build_matching_23():
    df_dem = find_demo_23()
    df_mapping = general_matching(df_dem)
    df_result = build_matching_24()

    manual_fixes = {
        "kkh": "kaufmännischekrankenkasse(kkh)",
        "mobilkrankenkasse": "betriebskrankenkassemobil",
        "r+vbkk": "r+vbetriebskrankenkasse",
        "tk": "technikerkrankenkasse(tk)",
    }

    df_mapping["Name_fm"] = df_mapping.apply(
        lambda row: manual_fixes[row["Name_dem"]]
        if row["Name_dem"] in manual_fixes else row["Name_fm"],
        axis=1
    )

    mapped = df_mapping["Name_fm"].unique()
    un_names_fm = df_fm["Krankenkasse"].unique()

    unmapped = [name for name in un_names_fm if name not in mapped]

    # for plenty of KKs there are no specific data so it is mapped to "others"
    others = pd.DataFrame({
        "Name_dem": ["sonstigegkv"] * len(unmapped),
        "Name_fm": unmapped
    })

    df_mapping = pd.concat([df_mapping, others], ignore_index=True)

    # removes empty ones
    df_mapping = df_mapping.dropna(subset=['Name_fm'])
    df_mapping = df_mapping[df_mapping["Name_fm"].str.strip() != ""]

    df_mapping.rename(columns={"Name_dem": "Name_dem_23"}, inplace=True)
    df_result = df_result.merge(df_mapping, on="Name_fm", how="left")
    write_excel(df_result, "../data/matching_tabelle.xlsx")
df_mapping = build_matching_23()


