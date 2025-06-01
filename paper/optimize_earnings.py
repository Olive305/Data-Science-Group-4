from data_extraction.utils import load_excel


def predict_membership_change(contrib_change, moderators, model):
    """
    Predicts the expected relative change in membership based on a contribution rate change
    and a set of moderators using a Difference-in-Differences (DiD) style linear model.
    does not yet take demographic change into concideration
    Parameters:
    - contrib_change (float): Change in contribution rate (e.g. 0.005 for a 0.5 percentage point increase)
    - moderators (dict): Dictionary of moderator values (e.g. {'mean_age': 45, 'income_level': 32000})
    - model (dict): Dictionary of estimated model coefficients, including:
    Returns:
    - float: Predicted relative membership change
    """

    # Start with the intercept and contribution change effect
    prediction = model['intercept'] + model['contrib_coef'] * contrib_change

    # Add effects of each moderator
    for key, value in moderators.items():
        if key in model:
            prediction += model[key] * value
        else:
            raise ValueError(f"Moderator '{key}' not found in model.")

    return prediction

def cost_calculation(contribution):
    df = load_excel('../data/Krankkosten.xlsx')
    change = predict_membership_change()
    result = 0
    for x in df['Alter']:
        result = result +(x*change)
    return result


def revenue_calculation(contribution_increase):
    base = 14.5 #data has yet to be imported
    contrib = base + contribution_increase
    result = contrib * predict_membership_change()
    return result

def max_earnings():
    best = []
    for x in range(50):
        contribution = x/100
        result = (revenue_calculation(contribution) - cost_calculation(contribution))
        best.append(contribution, result)

    return best



