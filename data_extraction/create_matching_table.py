import pandas as pd
import sys
import os
from rapidfuzz import process, fuzz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.data_extractor import find_demo_24, find_demo_23, sat_23, sat_24

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

    return df_result
def assign_group_name(name):
    lname = name.lower()
    if "aok" in lname:
        return "aokgesamt"
    elif "bkk" in lname:
        return "bkkgesamt"
    elif "ikk" in lname:
        return "ikkgesamt"
    else:
        return "sonstigegkv"

def matching_sat_23():
    df_sat = sat_23()
    #df_sat = basic_data_cleanup(df_sat, 'Krankenkasse')
    df_result= build_matching_23()
    df_mapping = general_matching(df_sat)

    manual_fixes = {
        "hkkkrankenkasse": "hkk",
        "kkh": "kaufmännischekrankenkasse(kkh)",
        "mhpluskrankenkasse": "mhplusbkk",
        "mobilkrankenkasse": "betriebskrankenkassemobil",
        "r+vbkk": "r+vbetriebskrankenkasse",
        "tk": "technikerkrankenkasse(tk)",
        "badenwürtt." : "aokbadenwürttemberg",
        "plus" : "aokplus",
        "heimatkk" : "heimatkrankenkasse",
        "bkk" : "sonstigegkv"
    }

    df_mapping["Name_fm"] = df_mapping.apply(
        lambda row: manual_fixes[row["Name_dem"]]
        if row["Name_dem"] in manual_fixes else row["Name_fm"],
        axis=1
    )

    #mapping sonstige to the remainders
    mapped = df_mapping["Name_fm"].unique()
    un_names_fm = df_fm["Krankenkasse"].unique()
    unmapped = [name for name in un_names_fm if name not in mapped]
    # for plenty of KKs there are no specific data so it is mapped to "others"
    others = pd.DataFrame({
        "Name_dem": [assign_group_name(name) for name in unmapped],
        "Name_fm": unmapped
    })

    df_mapping = pd.concat([df_mapping, others], ignore_index=True)
    df_mapping = df_mapping.dropna(subset=['Name_fm'])
    df_mapping.rename(columns={"Name_dem": "Name_sat_23"}, inplace=True)
    df_result = df_result.merge(df_mapping, on="Name_fm", how="left")
    return df_result
def matching_sat_24():
    df_sat = sat_24()
    df_result= matching_sat_23()
    df_mapping = general_matching(df_sat)

    manual_fixes = {
        "hkkkrankenkasse": "hkk",
        "kkh": "kaufmännischekrankenkasse(kkh)",
        "mhpluskrankenkasse": "mhplusbkk",
        "mobilkrankenkasse": "betriebskrankenkassemobil",
        "r+vbkk": "r+vbetriebskrankenkasse",
        "tk": "technikerkrankenkasse(tk)",
        "badenwürtt." : "aokbadenwürttemberg",
        "plus" : "aokplus",
        "heimatkk" : "heimatkrankenkasse",
        "bkk" : "sonstigegkv",
        "viactivkk" :  "viactivkrankenkasse",
        "bkkgs" : "bkkgildemeisterseidensticker",
    }

    df_mapping["Name_fm"] = df_mapping.apply(
        lambda row: manual_fixes[row["Name_dem"]]
        if row["Name_dem"] in manual_fixes else row["Name_fm"],
        axis=1
    )

    #mapping sonstige to the remainders
    mapped = df_mapping["Name_fm"].unique()
    un_names_fm = df_fm["Krankenkasse"].unique()
    unmapped = [name for name in un_names_fm if name not in mapped]
    # for plenty of KKs there are no specific data so it is mapped to "others"
    others = pd.DataFrame({
        "Name_dem": [assign_group_name(name) for name in unmapped],
        "Name_fm": unmapped
    })

    df_mapping = pd.concat([df_mapping, others], ignore_index=True)
    df_mapping = df_mapping.dropna(subset=['Name_fm'])
    df_mapping.rename(columns={"Name_dem": "Name_sat_24"}, inplace=True)
    df_result = df_result.merge(df_mapping, on="Name_fm", how="left")
    write_excel(df_result, "../data/matching_tabelle.xlsx", index=False)
matching_sat_24()

#write_excel(df_result, "../data/matching_tabelle.xlsx", index = False)

