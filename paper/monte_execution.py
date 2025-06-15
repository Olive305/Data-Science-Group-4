import pandas as pd
from data_extraction.utils import load_excel
from paper.monte_carlo import monte

# Income categories in order from lowest to highest
income_order = [
    "Haushaltsnettoeinkommen in Euro_Unter 1.000",
    "Haushaltsnettoeinkommen in Euro_1.000-1.499",
    "Haushaltsnettoeinkommen in Euro_1.500-1.999",
    "Haushaltsnettoeinkommen in Euro_2.000-2.499",
    "Haushaltsnettoeinkommen in Euro_2.500-3.999",
    "Haushaltsnettoeinkommen in Euro_ber 3.999"
]

# Extracts the income distribution from a row, infers the highest category
def get_income_distribution(row):
    dist = {col: row[col] for col in income_order if col in row}
    sum_known = sum(dist.values())
    dist["Haushaltsnettoeinkommen in Euro_ber 3.999"] = 1.0 - sum_known
    return dist

# Simulates income redistribution based on economic index
def simulate_income(economy, row):
    dist = get_income_distribution(row)
    if economy < 1.0:
        # Economy worsens: income shifts downward
        factor = 1 - economy
        value = dist["Haushaltsnettoeinkommen in Euro_ber 3.999"] * factor
        dist["Haushaltsnettoeinkommen in Euro_ber 3.999"] -= value
        dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] * factor
        dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] -= value
        dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] * factor
        dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] -= value
        dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] * factor
        dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] -= value
        dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] * factor
        dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] -= value
        dist["Haushaltsnettoeinkommen in Euro_Unter 1.000"] += value
    elif economy > 1.0:
        # Economy improves: income shifts upward
        factor = economy - 1
        value = dist["Haushaltsnettoeinkommen in Euro_Unter 1.000"] * factor
        dist["Haushaltsnettoeinkommen in Euro_Unter 1.000"] -= value
        dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] * factor
        dist["Haushaltsnettoeinkommen in Euro_1.000-1.499"] -= value
        dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] * factor
        dist["Haushaltsnettoeinkommen in Euro_1.500-1.999"] -= value
        dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] * factor
        dist["Haushaltsnettoeinkommen in Euro_2.000-2.499"] -= value
        dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] += value

        value = dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] * factor
        dist["Haushaltsnettoeinkommen in Euro_2.500-3.999"] -= value
        dist["Haushaltsnettoeinkommen in Euro_ber 3.999"] += value
    return dist

# Simulates demographic shift and recalculates member counts
def simulate_demography(monte_row, kk_row):
    age_cats = [
        "Alter_16-29 Jahre",
        "Alter_30-39 Jahre",
        "Alter_40-49 Jahre",
        "Alter_50-59 Jahre",
        "Alter_60-69 Jahre"
    ]
    sum_known = sum(kk_row[cat] for cat in age_cats)
    age_dist = {cat: kk_row[cat] for cat in age_cats}
    age_dist["Alter_70+ Jahre"] = 1.0 - sum_known

    change_rates = {
        "Alter_16-29 Jahre": monte_row["16-29 Jahre"],
        "Alter_30-39 Jahre": monte_row["30-39 Jahre"],
        "Alter_40-49 Jahre": monte_row["40-49 Jahre"],
        "Alter_50-59 Jahre": monte_row["50-59 Jahre"],
        "Alter_60-69 Jahre": monte_row["60-69 Jahre"],
        "Alter_70+ Jahre": monte_row["70+ Jahre"]
    }

    for age_cat in age_dist:
        age_dist[age_cat] *= change_rates[age_cat]

    # Apply factor to member and insured counts
    factor = sum(age_dist.values())
    mitglieder = kk_row["Mitglieder"] * factor
    versicherte = kk_row["Versicherte"] * factor

    return age_dist, mitglieder, versicherte
