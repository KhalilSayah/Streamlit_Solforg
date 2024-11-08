import json
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from criterias import get_criteria_list
from models import CategoricalCriteria, FinanceRound, FinancingRounds, ModelInit, NumericCriteria
from utility import calculate_mean_scores, calculate_score, format_dataframe, get_bonus_criteria_values, get_data, initialize_arrival_cycles, simulate_depletion_full_with_adjustment, simulate_employee_paths_custom

# Initialize session state for criteria list
if 'criteria_list' not in st.session_state:
    st.session_state.criteria_list = get_criteria_list()  # Use the imported function

# Initialize session state for model
if 'model_init' not in st.session_state:
    st.session_state.model_init = None

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Model Initialization", "Modify Scores & View Tables", "Calculate Score"])

# Model Initialization Page
if page == "Model Initialization":
    st.title("Model Initialization")

    # Input fields for the primary parameters
    st.subheader("Input Initialization")
    max_supply = st.number_input("Max supply", value=1000000000, step=1000000)
    listing_price = st.number_input("Listing price", value=0.75, step=0.01)
    base_alloc = st.number_input("Allocation Base", value=0.08, step=0.01)
    bonus_alloc = st.number_input("Allocation Bonus", value=0.02, step=0.01)
    employees = st.number_input("Employees", value=95, step=1)

    # Table to input financing rounds data
    st.subheader("Financing Rounds")
    round_labels = ["Pre-seed", "Strategic Angels", "Seed", "Strategic Bridge", "Series A", "Private Sale", "TGE"]
    rounds_data = []

    for label in round_labels:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            start_date = st.date_input(f"{label} Start Date", key=f"{label}_start")
        with col2:
            end_date = st.date_input(f"{label} End Date", key=f"{label}_end")
        with col3:
            raised_funds = st.number_input(f"{label} Raised Funds", value=0, step=100000, key=f"{label}_raised")
        with col4:
            valuation = st.number_input(f"{label} Valuation", value=0, step=100000, key=f"{label}_valuation")

        rounds_data.append(FinanceRound(
            label=label,
            start=start_date,
            end=end_date,
            raised_funds=raised_funds,
            valuation=valuation
        ))

    # Save model in session state
    if st.button("Save Model"):
        model_init = ModelInit(
            max_supply=max_supply,
            listing_price=listing_price,
            base_alloc=base_alloc,
            bonus_alloc=bonus_alloc,
            employees=employees,
            finance_rounds=FinancingRounds(rounds=rounds_data)
        )
        st.session_state.model_init = model_init
        st.success("Model initialized and saved in session state.")

    # Display the saved model information if it exists
    if st.session_state.model_init:
        st.subheader("Saved Model")
        st.write(st.session_state.model_init.dict())

# Modify Scores & View Tables Page
elif page == "Modify Scores & View Tables":
    st.header("Modify Scores and View All Criteria Tables")

    # Sidebar for selecting a criterion to modify
    st.sidebar.header("Select a Criterion to Modify")
    criteria_labels = [criterion.label for criterion in st.session_state.criteria_list]
    selected_criteria_label = st.sidebar.selectbox("Choose a criterion", criteria_labels)

    # Find the selected criterion
    selected_criteria = next(
        (criterion for criterion in st.session_state.criteria_list if criterion.label == selected_criteria_label), None
    )

    # Display form to modify the selected criterion
    if selected_criteria:
        st.subheader(f"Modify '{selected_criteria_label}' Scores")

        if isinstance(selected_criteria, NumericCriteria):
            for part in selected_criteria.criteria_parts:
                part.score = st.sidebar.number_input(
                    f"{part.label} (Current Score: {part.score})", 
                    value=part.score, 
                    step=0.1, 
                    key=f"{selected_criteria_label}_{part.label}"
                )
        elif isinstance(selected_criteria, CategoricalCriteria):
            for part in selected_criteria.criteria_parts:
                part.score = st.sidebar.number_input(
                    f"{part.label} (Current Score: {part.score})", 
                    value=part.score, 
                    step=0.1, 
                    key=f"{selected_criteria_label}_{part.label}"
                )

    # Display all criteria tables on the right side
    st.subheader("All Criteria and Scores")
    for criterion in st.session_state.criteria_list:
        st.subheader(criterion.label)
        
        if isinstance(criterion, NumericCriteria):
            data = {
                "Range": [f"{part.min_value} - {part.max_value}" for part in criterion.criteria_parts],
                "Score": [part.score for part in criterion.criteria_parts]
            }
            df = pd.DataFrame(data, columns=["Range", "Score"])
            st.table(df)
            
        elif isinstance(criterion, CategoricalCriteria):
            data = {
                "Category": [part.label for part in criterion.criteria_parts],
                "Score": [part.score for part in criterion.criteria_parts]
            }
            df = pd.DataFrame(data, columns=["Category", "Score"])
            st.table(df)

# Calculate Score Page
elif page == "Calculate Score":
    st.header("Round-by-Round Score Calculation")

    # Initialize session state for tracking rounds and storing inputs
    if 'current_round_index' not in st.session_state:
        st.session_state.current_round_index = 0  # Start with the first round
        st.session_state.round_inputs = []  # Initialize list to store round data
    

    # Check if the model has been initialized
    if st.session_state.model_init:
        # Retrieve the finance rounds and criteria list
        finance_rounds = st.session_state.model_init.finance_rounds.rounds
        criteria_list = st.session_state.criteria_list
        current_index = st.session_state.current_round_index

        
        # Check if there are more rounds to enter
        if current_index < len(finance_rounds):
            current_round = finance_rounds[current_index]
            st.subheader(f"Enter Inputs for {current_round.label} Round")

            # Temporary storage for input values for the current round
            round_data = {
                "Round": current_round.label,
                "Criterion": [],
                "Input Value": []
            }

            # Clear the previous round's employee data before adding new
            # (Only clear if you need to reset each round, but in this case, we keep accumulating)
            # new_employees.clear()  # Do not clear the list to retain previous rounds' values

            for criterion in criteria_list:
                if isinstance(criterion, NumericCriteria) and criterion.label == "Joining Time":
                    continue  # Skip Joining Time (JT) criteria

                if isinstance(criterion, CategoricalCriteria):
                    # For Categorical Criteria, ask user for input value
                    options = {part.label: part.score for part in criterion.criteria_parts}
                    selected_label = st.selectbox(f"Choose {criterion.label}", options.keys())
                    round_data["Criterion"].append(criterion.label)
                    round_data["Input Value"].append(selected_label)

                if isinstance(criterion, NumericCriteria):
                    # Skip the "Joining Time" numeric criterion, but ask for employees in each phase
                    if criterion.label == "Joining Time":
                        continue
                    else:
                        st.write("No Numeric Criteria needed for JT")

            # Ask for the employee count per phase (add only once per round)
            num_employees = st.number_input("Enter the number of new employees in this round", min_value=0)
            round_data["Criterion"].append("Employees")
            round_data["Input Value"].append(num_employees)

            # Save button for the current round
            if st.button("Save Round Data", key=current_index):
                st.session_state.round_inputs.append(round_data)  # Save data for this round
                st.session_state.current_round_index += 1  # Move to the next round
                st.success(f"Data for {current_round.label} saved. Proceeding to the next round.")

        # If all rounds are completed, display the summary table
        else:
            st.subheader("Summary of All Round Inputs")

            # Initialize an empty dictionary to store the data
            summary_data = {}

            # Add each round's data to the summary table
            for round_data in st.session_state.round_inputs:
                round_name = round_data["Round"]

                # If this round is not in the summary_data, create an entry
                if round_name not in summary_data:
                    summary_data[round_name] = {}

                # Loop through the criteria and input values
                for i in range(len(round_data["Criterion"])):  # Now includes JT data
                    criterion = round_data["Criterion"][i]
                    input_value = round_data["Input Value"][i]

                    # Add the input value under the corresponding criterion
                    summary_data[round_name][criterion] = input_value

            # Convert summary_data to a DataFrame for display
            df_summary = pd.DataFrame.from_dict(summary_data, orient='index').reset_index()
            df_summary.columns = ['Round'] + list(df_summary.columns[1:])  # Rename the first column to 'Round'

            # After gathering all inputs, calculate the JT score using the function
            jt_scores = [part.score for part in criteria_list[0].criteria_parts]
            mean_jt_scores = calculate_mean_scores(df_summary["Employees"], jt_scores)
            df_summary["JT"] = mean_jt_scores

            # Display the DataFrame
            st.table(df_summary)

            # Print the actual JSON that contains all this data
            raw_data = json.loads(df_summary.to_json(orient='records', indent=4))
            scores_output = json.loads(calculate_score(raw_data, st.session_state.criteria_list))

            st.text_area("JSON Output", value=scores_output, height=300)

            # Données pour les phases et leurs employés de base
            employees_per_phase = get_data(scores_output,"employee_count") # Nombre d'employés pour chaque phase

            # Allocation de base et BTU
            total_tokens_for_employees = st.session_state.model_init.max_supply * st.session_state.model_init.base_alloc
            BTU = st.session_state.model_init.get_btu()

            # Coefficients de risque pour chaque phase
            risk_coefficients = [1.0, 0.598, 0.5, 0.32, 0.267, 0.218, 0.19, 0.19]

            employees_per_phase_30_percent_increase = [int(employees * 1.3) for employees in employees_per_phase]

            remaining_allocation_with_risk_adjusted, final_reserve_adjusted = simulate_depletion_full_with_adjustment(total_tokens_for_employees, BTU, risk_coefficients, employees_per_phase, scores_output["rounds"])
            remaining_allocation_risk_30_adjusted, final_reserve_risk_30_adjusted = simulate_depletion_full_with_adjustment(total_tokens_for_employees, BTU, risk_coefficients, employees_per_phase_30_percent_increase, scores_output["rounds"])

            # Tracer le graphique pour la déplétion liée au risque (BTU1) avec ajustement et +30% d'employés
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(remaining_allocation_with_risk_adjusted)), remaining_allocation_with_risk_adjusted, marker='o', color='purple', label="Token Risk Allocation (Base Case, Adjusted)")
            plt.plot(range(len(remaining_allocation_risk_30_adjusted)), remaining_allocation_risk_30_adjusted, marker='x', color='blue', linestyle='--', label="Token Risk Allocation (+30% Employees, Adjusted)")
            plt.xlabel("Number of Employees Hired")
            plt.ylabel("Remaining Allocation (Millions)")
            plt.title("Depletion of Token Risk Allocation (BTU1) with Adjusted Coefficients and +30% Employees")
            plt.legend()
            plt.tight_layout()
            plt.text(len(remaining_allocation_with_risk_adjusted)-1, remaining_allocation_with_risk_adjusted[-1] + 2, f'Final Reserve Base Adjusted: {final_reserve_adjusted:.2f}M', color='purple', ha='right', fontsize=10, weight='bold')
            plt.text(len(remaining_allocation_risk_30_adjusted)-1, remaining_allocation_risk_30_adjusted[-1] + 2, f'Final Reserve +30% Adjusted: {final_reserve_risk_30_adjusted:.2f}M', color='blue', ha='right', fontsize=10, weight='bold')

            # Display the plot in Streamlit
            st.pyplot(plt)


            df_employees = format_dataframe(st.session_state.model_init, scores_output)
            # Visualisation des résultats avec trois graphiques
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

            # Premier graphique - Tokens ajustés par employé et nouveaux employés par phase
            bars = ax1.bar(df_employees['Phase'], df_employees['Adjusted Tokens per Employee'], color='mediumseagreen', label='Adjusted Tokens per Employee')
            ax1.set_xlabel("Phase")
            ax1.set_ylabel("Adjusted Tokens per Employee", color='mediumseagreen')
            ax1.tick_params(axis='y', labelcolor='mediumseagreen')

            # Ajouter les nouveaux employés en créneaux
            ax2_1 = ax1.twinx()
            ax2_1.step(df_employees['Phase'], df_employees['New Employee'], color='firebrick', where='mid', label='New Employees')
            ax2_1.set_ylabel("New Employees", color='firebrick')
            ax2_1.tick_params(axis='y', labelcolor='firebrick')
            ax1.set_title("Average Tokens Issued per Employee by Phase")

            # Second graphique - Tokens distribués par phase sans cumul avec annotations de pourcentage
            bars2 = ax2.bar(df_employees['Phase'], df_employees['Tokens Distributed per Phase'], color='royalblue', label='Tokens Distributed per Phase')
            ax2.set_xlabel("Phase")
            ax2.set_ylabel("Tokens Distributed per Phase", color='royalblue')
            ax2.tick_params(axis='y', labelcolor='royalblue')
            ax2.set_title("Tokens Distributed per Phase (Non-cumulative)")

            # Annoter chaque barre avec le pourcentage de répartition du total alloué
            for bar, pct in zip(bars2, df_employees['Percentage of Total Allocation']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{pct:.1f}%', ha='center', va='bottom', color='black', fontsize=10)

            # Troisième graphique - Coût en dollars par employé par phase
            bars3 = ax3.bar(df_employees['Phase'], df_employees['Cost per Employee ($)'], color='slateblue', label='Cost per Employee ($)')
            ax3.set_xlabel("Phase")
            ax3.set_ylabel("Cost per Employee ($)", color='slateblue')
            ax3.tick_params(axis='y', labelcolor='slateblue')
            ax3.set_title("Base Compensation per Employee by Phase in Dollars @TGE listing price")

            # Annoter chaque barre avec le coût en dollars
            for bar, cost in zip(bars3, df_employees['Cost per Employee ($)']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2, height, f'${cost:.2f}', ha='center', va='bottom', color='black', fontsize=10)

            plt.tight_layout()
            plt.show()
            st.pyplot(fig)



            # THIS PART NEED TO BE RECODE 
            total_tokens_for_bonus = st.session_state.model_init.max_supply * st.session_state.model_init.bonus_alloc
            cycles = 8
            BTU2 = st.session_state.model_init.get_btu("bonus")
            bcf_options = {1.0: 1.0, 0.85: 0.85, 0.7: 0.7, 0.55: 0.55}
            skewness_input = 0

            rounds = df_employees['Phase']
            start_dates = [round.start for round in st.session_state.model_init.finance_rounds.rounds]
            end_dates = [round.end for round in st.session_state.model_init.finance_rounds.rounds]
            employee_numbers = employees_per_phase

            employee_arrivals_base = {
                rounds[i]: (employee_numbers[i], start_dates[i], end_dates[i]) 
                for i in range(len(rounds))
            }

            employee_arrivals_30 = {phase: (int(count * 1.3), start, end) for phase, (count, start, end) in employee_arrivals_base.items()}

            # Coefficients pour chaque critère de performance (IP, PI, IC, TA)
            IP_values, PI_values, IC_values, TA_values = get_bonus_criteria_values(st.session_state.criteria_list)
            print("IP Values:", IP_values)
            print("PI Values:", PI_values)
            print("IC Values:", IC_values)
            print("TA Values:", TA_values)


            # Convertir les dates en format cycle
            start_date = pd.to_datetime("2024-05-01")

            arrival_cycles_base = initialize_arrival_cycles(employee_arrivals_base,start_date)
            arrival_cycles_30 = initialize_arrival_cycles(employee_arrivals_30,start_date)

            employee_paths_base, remaining_allocation_base = simulate_employee_paths_custom(
                arrival_cycles_base, BTU2, bcf_options, total_tokens_for_bonus, IP_values,PI_values,IC_values,TA_values,skewness=skewness_input, cycles=cycles,
            )
            employee_paths_30, remaining_allocation_30 = simulate_employee_paths_custom(
                arrival_cycles_30, BTU2, bcf_options, total_tokens_for_bonus,IP_values,PI_values,IC_values,TA_values, skewness=skewness_input, cycles=cycles
            )

            fig2 = plt.figure(figsize=(12, 6))
            colors = plt.cm.plasma(np.linspace(0, 1, len(employee_paths_base)))  # Palette de couleurs plus distincte
            for idx, (employee_id, path) in enumerate(employee_paths_base.items()):
                plt.step(range(1, len(path) + 1), path, where='post', alpha=0.5, color=colors[idx])

            plt.xlabel("Cycles (6 months)")
            plt.ylabel("Cumulative Tokens per Employee (Millions)")
            plt.title("Token Bonus Allocation: Adjustable Skewness and Total Allocation Depletion")

            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(range(1, len(remaining_allocation_base) + 1), remaining_allocation_base, color='blue', linestyle='--', label="Total Allocation Remaining (Base)")
            ax2.set_ylim(0, max(remaining_allocation_base) + 2)
            ax2.set_ylabel("Remaining Allocation (Millions)", color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            ax2.plot(range(1, len(remaining_allocation_30) + 1), remaining_allocation_30, color='green', linestyle='--', label="Total Allocation Remaining (+30% Employees)")

            ax2.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
            st.pyplot(fig2)












    else:
        st.warning("Please initialize the model on the 'Model Initialization' page first.")

