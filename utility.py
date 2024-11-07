import json
import numpy as np
import pandas as pd

from models import CategoricalCriteriaPart

def calculate_normalized_risk_coefficient(valuations:np.array):
    # Calcul du coefficient de risque initial sans constante
    initial_risk_coefficients = 1 / np.sqrt(valuations)
   
    # Trouver le coefficient de risque maximum pour normaliser
    max_risk_coefficient = initial_risk_coefficients.max()
   
    # Normaliser sur une base de 100
    normalized_risk_coefficients = (initial_risk_coefficients / max_risk_coefficient) * 100
   
    return normalized_risk_coefficients



def calculate_mean_scores(new_employees, scores):
    """
    Calculate mean JT scores for each funding stage based on cumulative joining position and configurable scores.

    Parameters:
    new_employees (list of int): List of new employee counts per funding stage.
    scores (list of float): List of scores for each threshold [<10, <25, <50, <100, <150].

    Returns:
    list of float: Mean JT score for each funding stage.
    """
    # Fixed thresholds
    thresholds = [10, 25, 50, 100, 150]
    cumulative_position = 0
    mean_scores = []

    # Function to assign score based on cumulative position using fixed thresholds and provided scores
    def assign_score(position):
        for i, threshold in enumerate(thresholds):
            if position < threshold:
                return scores[i]
        return 0  # Default score if position exceeds all thresholds

    # Loop through each stage
    for num_employees in new_employees:
        # Calculate scores for each new employee in this stage
        stage_scores = []
        for _ in range(num_employees):
            cumulative_position += 1
            score = assign_score(cumulative_position)
            stage_scores.append(score)

        # Calculate the mean score for the current stage
        mean_score = np.mean(stage_scores)
        mean_scores.append(mean_score)

    return mean_scores


def calculate_score(raw_json, criteria_list):
    # Initialize the result dictionary
    result = {
        "rounds": []
    }

    # Loop through the raw_json (which is a list of rounds)
    for round_data in raw_json:
        round_result = {
            "round": round_data["Round"],
            "employee_count": round_data["Employees"],
            "criteria_scores": {},
            "JT_score": round_data["JT"],
        }

        
        # Calculate scores for other criteria (Salary, Role, etc.)
        for criterion, selected_value in round_data.items():
            if criterion == "Round" or criterion == "Employees" or criterion == "JT":
                continue  # Skip non-criteria fields

            # Search for the matching criteria
            matched_criteria = next((c for c in criteria_list if c.label == criterion), None)
            
            if matched_criteria:
                for part in matched_criteria.criteria_parts:
                    if isinstance(part, CategoricalCriteriaPart) and part.label == selected_value:
                        round_result["criteria_scores"][criterion] = part.score
                        break
        
        # Add round result to the final result
        result["rounds"].append(round_result)

    # Return the result as a JSON object
    return json.dumps(result, indent=4)



def calculate_adjusted_coefficient(round):
    SL_central = round["criteria_scores"]["Seniority Level"]
    RI_central = round["criteria_scores"]["Role Importance"]
    SC_central = round["criteria_scores"]["Salary Compensation"]
    JT = round["JT_score"]
    return (SL_central + RI_central + SC_central + JT) / 4



# Simulation de déplétion pour le risque en utilisant le coefficient ajusté
def simulate_depletion_full_with_adjustment(allocation, btu, risk_coefficients, employees_per_phase, stages):
    remaining_allocation = []
    for i, (employees, risk_coef, stage) in enumerate(zip(employees_per_phase, risk_coefficients, stages)):
        adjusted_coefficient = calculate_adjusted_coefficient(stage)
        for _ in range(employees):
            if allocation > 0:
                remaining_allocation.append(allocation / 1_000_000)  # En millions
                allocation -= btu * risk_coef * adjusted_coefficient
            else:
                remaining_allocation.append(0)  # Allocation épuisée
    return remaining_allocation, allocation / 1_000_000  # En millions


def get_data(data, spec):
    employees = []
    for r in data["rounds"]:
        employees.append(r[spec])
    return employees


def format_dataframe(model, scores):

    number_employe = get_data(scores,"employee_count")
    risk_coef = [1.0, 0.598, 0.5, 0.32, 0.267, 0.218, 0.19]
    
    adjusted_tokens_per_emp = [value * model.get_btu() for value in risk_coef]
    
    token_distributed_per_phase = [a * b for a, b in zip(number_employe, adjusted_tokens_per_emp)]
    
    allocated_tokens = model.max_supply * model.base_alloc
    percentage_total_alloc = [(value / allocated_tokens) * 100 for value in token_distributed_per_phase]
    cost_per_employee = [value * model.listing_price for value in adjusted_tokens_per_emp] 
    

    return pd.DataFrame({
        "Phase" :get_data(scores, "round" ),
        "New Employee" : number_employe  ,
        "Risk Coefficient" : risk_coef,
        "Adjusted Tokens per Employee" : adjusted_tokens_per_emp,
        "Tokens Distributed per Phase" : token_distributed_per_phase,
        "Percentage of Total Allocation" : percentage_total_alloc,
        "Cost per Employee ($)":cost_per_employee

    })

    
