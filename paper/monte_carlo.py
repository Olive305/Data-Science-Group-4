import numpy as np
import pandas as pd
from data_extraction.utils import load_excel
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def simulate_demographic():
    """
    simulates future demographic
    returns: list of each demographic segment with its future development
    """
    demo= [["16-29 Jahre", -0.005, 0.002],
    ["30-39 Jahre", -0.004, 0.002],
    ["40-49 Jahre", -0.006, 0.002],
    ["50-59 Jahre", -0.003, 0.002],
    ["60-69 Jahre",  0.002, 0.002],
    ["70+ Jahre",    0.006, 0.003]]
    result = []
    #these estimates are the demographics listed and are roughly what experts expect
    for x in demo:
        growth_rate = np.random.normal(loc=x[1], scale=x[2])
        result.append([x[0],1+(growth_rate/4)])

    return result
def simulate_economy():
    """
    simulates economy => Income distribution
    returns: float
    """
    growth_rate = np.random.normal(loc=0.02, scale=0.04) #growth of 2% with big variance to have bad years
    return 1+(growth_rate/4)


def competitor_behavior():
    """
    Calculates average change in contribution rate over all insurers and
    returns a simulated growth factor for the competitor behavior.
    Returns: float
    """
    df = load_excel('../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
    df = df.sort_values(by=['Krankenkasse', 'Quartal'])

    df['Rate'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].diff()
    overall_mean = df.groupby('Krankenkasse')['Rate'].mean()
    overall_mean = overall_mean.mean()+0.2
    overall_std = df.groupby('Krankenkasse')['Rate'].std()
    overall_std = overall_std.mean()*0.5
    growth_rate = np.random.normal(loc=overall_mean, scale=overall_std)
    return growth_rate

def customer_behavior():
    """
    assumes some random variance in satisfaction and other factors that were taken as moderators
    returns: float
    """
    growth_rate = np.random.normal(loc=0.00, scale=0.002)
    return 1+(growth_rate/4)



def monte(iterations, years):
    """
    executes Monte Carlo simulations
    :param iterations:
    :param years:
    :returns dataframe:
    """
    results = []
    age_groups = ['16-29 Jahre', '30-39 Jahre', '40-49 Jahre', '50-59 Jahre', '60-69 Jahre', '70+ Jahre']

    for x in range(iterations):
        np.random.seed(x)
        quarters = years * 4

        for q in range(quarters):
            sd = simulate_demographic()
            se = simulate_economy()
            cob = competitor_behavior()
            cub = customer_behavior()

            row = {
                'iteration': x,
                'year': q // 4,
                'quarter': (q % 4) + 1,
                'economy': se,
                'competitor': cob,
                'customer': cub
            }
            for dem in sd:
                row[dem[0]] = dem[1]

            results.append(row)

    df = pd.DataFrame(results)
    #print(df)
    return df

if __name__ == '__main__':
    print(monte(3, 5))
#print(competitor_behavior())
