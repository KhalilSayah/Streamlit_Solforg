import json
import random
import numpy as np
import pandas as pd
from scipy.stats import skewnorm

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

    
def get_cycle_from_date(date_str,start_date):
    date = pd.to_datetime(date_str)
    return int((date - start_date).days // 182) + 1  # Conversion en cycles de 6 mois, ajusté pour commencer à 1

def initialize_arrival_cycles(employee_arrivals,start_date):
    arrival_cycles = []
    for phase, (num_employees, start, end) in employee_arrivals.items():
        start_cycle = get_cycle_from_date(start,start_date)
        end_cycle = get_cycle_from_date(end,start_date)
        arrival_cycles.extend([random.randint(start_cycle, end_cycle) for _ in range(num_employees)])
    return arrival_cycles

def generate_stochastic_coefficients(skewness,IP_values,PI_values,IC_values,TA_values):
    def skewed_choice(values):
        choice = list(values.values())
        if skewness == 0:
            return random.choice(choice)
        else:
            skewed_distribution = skewnorm.rvs(a=skewness, size=1)
            index = min(max(int((skewed_distribution[0] + 1) * (len(choice) / 2)), 0), len(choice) - 1)
            return choice[index]
    
    return {
        "IP": skewed_choice(IP_values),
        "PI": skewed_choice(PI_values),
        "IC": skewed_choice(IC_values),
        "TA": skewed_choice(TA_values)
    }

def adjust_score(score):
    variation = random.uniform(-0.3, 0.3)  # Variation de -30% à +30%
    return score * (1 + variation)

def simulate_employee_paths_custom(arrival_cycles, btu, bcf_options, total_allocation,IP_values,PI_values,IC_values,TA_values, skewness=0, cycles=8, min_token=1_000):
    employee_paths = {emp: [] for emp in range(len(arrival_cycles))}
    remaining_allocation = []
    
    for cycle in range(1, cycles + 1):  # Parcourir chaque cycle, en commençant à 1
        cycle_allocation = total_allocation
        for employee_id, arrival_cycle in enumerate(arrival_cycles):
            if arrival_cycle > cycle:
                employee_paths[employee_id].append(employee_paths[employee_id][-1] if employee_paths[employee_id] else 0)
                continue
            
            coefficients = generate_stochastic_coefficients(skewness,IP_values,PI_values,IC_values,TA_values)
            base_score = (coefficients["IP"] + coefficients["PI"] + coefficients["IC"] + coefficients["TA"]) / 4
            adjusted_score = adjust_score(base_score)
            
            bcf = random.choice(list(bcf_options.values()))
            tokens_received = max(btu * bcf * adjusted_score, min_token)
            
            tokens_distributed = min(tokens_received, total_allocation)
            cycle_allocation -= tokens_distributed
            total_allocation -= tokens_distributed
            if employee_paths[employee_id]:
                employee_paths[employee_id].append(employee_paths[employee_id][-1] + tokens_distributed / 1_000_000)
            else:
                employee_paths[employee_id].append(tokens_distributed / 1_000_000)
        
        remaining_allocation.append(total_allocation / 1_000_000)
    
    return employee_paths, remaining_allocation



def get_bonus_criteria_values(criterialist):
    # Initialize the dictionaries
    IP_values = {}
    PI_values = {}
    IC_values = {}
    TA_values = {}

    # Iterate through criteria list and extract the relevant bonus criteria
    for criteria in criterialist:
        if criteria.label == "Bonus - Individual Performance":
            for part in criteria.criteria_parts:
                IP_values[part.label] = part.score
        
        elif criteria.label == "Bonus - Project Impact":
            for part in criteria.criteria_parts:
                PI_values[part.label] = part.score
        
        elif criteria.label == "Bonus - Innovation Contribution":
            for part in criteria.criteria_parts:
                IC_values[part.label] = part.score
        
        elif criteria.label == "Bonus - Tenure Adjustment":
            for part in criteria.criteria_parts:
                TA_values[part.label] = part.score

    # Return the dictionaries with the extracted values
    return IP_values, PI_values, IC_values, TA_values