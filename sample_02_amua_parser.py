# run_parser.py
# -*- coding: utf-8 -*-
"""
Parses the specified AMUA XML file using the existing parser.py.
Presents a summary of the parsed Project object.
"""

import sys
import os
import json
from pathlib import Path
# Add import for typing modules used in type hints
from typing import Dict, List, Optional, Any # <-- Keep this line

# Add the current directory to sys.path to ensure model.py and parser.py can be imported
# This is helpful if you run the script directly.
# Adjust this if model.py/parser.py are in a different location relative to this script.

current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Adding '{current_dir}' to sys.path for imports.") # Debugging line
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) # Add to the beginning of the path

# Assuming your structure is modules/decisiontree/...
decisiontree_package_dir = Path(__file__).parent # This gets /.../modules/decisiontree
modules_dir = decisiontree_package_dir.parent         # This gets /.../modules
if str(modules_dir) not in sys.path:
     sys.path.insert(0, str(modules_dir)) # Add the 'modules' directory to sys.path

try:
    # Import the main parsing function from parser.py
    from decisiontree.parser import parse_amua_project # Adjusted import path based on project structure
    # Import the Project and MarkovChain model classes from model.py
    from decisiontree.model import Project, MarkovChain, TreeNode # <-- Remove ChainResult and SimulationResult here
    # Import the cohort simulation function AND the new result classes from simulator.py
    from decisiontree.simulator import run_cohort_simulation, display_aggregated_results, ChainResult, SimulationResult # <-- Import ChainResult and SimulationResult from simulator

except ImportError as e:
    print(f"Fatal Error: Could not import necessary modules (parser, model, or simulator).")
    print(f"Please ensure your project structure and sys.path are correct.")
    print(f"Details: {e}")
    # Print sys.path for debugging import issues
    # print("Current sys.path:", sys.path)
    sys.exit(1)

def display_project_summary(project_model: Project):
    """Displays summary information about the parsed project (excluding global params/vars detail)."""
    print("\n--- Project Summary ---")
    print(f"Project Name: {project_model.name if project_model.name else 'Unnamed'}")
    print(f"Model Type: {project_model.model_type}")
    print(f"Scale: {project_model.scale}")
    print(f"Simulation Type: {project_model.sim_type}")
    print(f"Cohort Size: {project_model.cohort_size}")
    print(f"CRN Enabled: {project_model.crn}")
    print(f"Number of Trees: {len(project_model.decision_trees)}")
    print(f"Number of Tables Parsed: {len(project_model.tables)}")
    # Global params/vars counts are already in get_variables, maybe keep here too or remove from one.
    print(f"Number of Global Parameters Parsed: {len(project_model.global_parameters)}")
    print(f"Number of Global Variables Parsed: {len(project_model.global_variables)}")


    if project_model.dim_info:
        print(f"Number of Dimensions: {len(project_model.dim_info.dimensions)}")
        dim_names = [d.name for d in project_model.dim_info.dimensions]
        print(f"Dimensions: {dim_names}")
        print(f"Cost Dimension Index: {project_model.dim_info.cost_dim_index}")
        print(f"Effect Dimension Index: {project_model.dim_info.effect_dim_index}")
        print(f"WTP: {project_model.dim_info.wtp}")

    print("-" * 30)


# --- START MODIFIED FUNCTION ---
def get_tree_node_instances_json(nodes_to_serialize):
    """
    Serializes TreeNode instances (or a list/tuple of them) to JSON format and prints.
    Calls to_dict() on each valid node.
    """
    if nodes_to_serialize is None:
        print("\n--- No nodes provided for serialization (input was None). ---")
        return

    # Ensure we have an iterable list of nodes
    if not isinstance(nodes_to_serialize, (list, tuple)):
        nodes_list = [nodes_to_serialize]
    else:
        nodes_list = nodes_to_serialize

    if not nodes_list:
        print("\n--- No nodes provided for serialization (list is empty). ---")
        return

    print(f"\n--- Attempting to serialize {len(nodes_list)} TreeNode instance(s) ---")

    for i, node in enumerate(nodes_list):
        # Check if the object is actually a TreeNode instance or one of its subclasses
        # Use isinstance with the base class TreeNode
        if not isinstance(node, TreeNode):
            print(f"Warning: Item {i+1} in the list is not a TreeNode instance ({type(node)}). Skipping serialization.")
            continue

        # Determine a suitable identifier for the print statement (using name if available, otherwise index)
        node_identifier = f"'{node.name}'" if hasattr(node, 'name') and node.name else f"Index: {node.index}"

        print(f"\n--- JSON Output for TreeNode {i+1} ({node.__class__.__name__}): {node_identifier} ---")
        try:
            # Get the dictionary representation (this should recursively call to_dict on children)
            node_dict = node.to_dict()

            # Convert the dictionary to a JSON string for pretty printing
            node_json_string = json.dumps(node_dict, indent=4, sort_keys=False)

            print(node_json_string)

        except AttributeError as ae:
             print(f"Error during JSON serialization for node '{node.name}' (Index {node.index}): Missing attribute for serialization - {ae}")
        except Exception as e:
            print(f"Error during JSON serialization for node '{node.name}' (Index {node.index}): {e}")

    print("\n------------------- End of TreeNode JSON(s) -------------------")
# --- END MODIFIED FUNCTION ---


# --- Restored get_variables function ---
def get_variables(project_model: Project):
    """Displays summary information including parameters and variables."""
    print("\n--- Parsing Completed Successfully ---")
    print("Project object successfully constructed.")

    # Display key information from the project object
    print(f"Project Name: {project_model.name if project_model.name else 'Unnamed'}")
    print(f"Model Type: {project_model.model_type}")
    print(f"Simulation Type: {project_model.sim_type}")
    if project_model.dim_info:
        print(f"Number of Dimensions: {len(project_model.dim_info.dimensions)}")
        print(f"Cost Dimension Index: {project_model.dim_info.cost_dim_index}")
        print(f"Effect Dimension Index: {project_model.dim_info.effect_dim_index}")

    print(f"\nNumber of Tables Parsed: {len(project_model.tables)}")
    print(f"\nNumber of Global Parameters Parsed: {len(project_model.global_parameters)}")
    # Add call to display parameters table (assuming Project class has this method)
    project_model.display_parameters()

    print(f"\nNumber of Global Variables Parsed: {len(project_model.global_variables)}")
    # Add call to display variables table (assuming Project class has this method)
    project_model.display_variables()

def get_tree_node_instances(decision_tree):
    print(f"\n--- Attempting to serialize DecisionNodes in tree '{decision_tree.name}' ---")

    # Projectオブジェクトの辞書形式の decision_trees 属性から名前でツリーを取得
    target_tree = project_model.decision_trees.get(decision_tree.name)
    print(target_tree)

    if target_tree:
            print(f"Accessing DecisionTree by name: '{decision_tree.name}'")

            # そのツリーに含まれる全てのDecisionNodeを取得 (DecisionTreeクラスにget_decision_nodesメソッドが必要)
            decision_nodes_in_tree = target_tree.get_decision_nodes() # <--- このメソッドはmodel.pyのDecisionTreeに追加が必要です

            if decision_nodes_in_tree:
                print(f"Found {len(decision_nodes_in_tree)} DecisionNode(s) in tree '{decision_tree.name}'. Serializing each:")

                for i, decision_node in enumerate(decision_nodes_in_tree):
                    print(f"\n--- JSON Output for DecisionNode {i+1} in '{decision_tree.name}': '{decision_node.name}' (Index: {decision_node.index}) ---")
                    try:
                        # Get the dictionary representation (includes children recursively)
                        node_dict = decision_node.to_dict() # <--- to_dictメソッドはmodel.pyのTreeNodeサブクラスに追加が必要です

                        # Convert the dictionary to a JSON string
                        node_json_string = json.dumps(node_dict, indent=4, sort_keys=False)

                        print(node_json_string)

                    except Exception as e:
                        print(f"Error during JSON serialization for DecisionNode '{decision_node.name}': {e}")
                print("\n------------------- End of DecisionNode JSONs in tree -------------------")
            else:
                print(f"No DecisionNode found within DecisionTree '{decision_tree.name}'.")
    else:
            print(f"DecisionTree with name '{decision_tree.name}' not found in the project.")

def get_project_model(amua_file_name: str):
    """
    Main function to parse the specified AMUA XML file and display results.

    Args:
        amua_file_name: The name of the XML file to parse (assumed to be in the same directory).
    """
    amua_file_path = Path(current_dir) / amua_file_name # Use pathlib for cleaner path joining

    print(f"Attempting to parse the file: {amua_file_path}")

    # Call the parse_amua_project function to parse the XML and build the model
    project_model = parse_amua_project(amua_file_path)

    # Check the result and display a summary
    if project_model:
        print("\n--- Parsing Completed Successfully ---")
        print("Project object successfully constructed.")

        get_variables(project_model)

        # --- Demonstrate JSON Serialization ---
        print("\n--- Decision Node to JSON Serialization ---")
    else:
        print("\n--- Parsing Failed ---")
        print(f"Could not construct the Project object from '{amua_file_path}'.")
        print("Please check for errors reported during parsing above.")
    
    return project_model


if __name__ == "__main__":
    print("--- Running AMUA Analysis Script ---")
    base_dir = Path(__file__).parent
    amua_file_name = "S3_model_pancreatic_cancer.amua" # Replace with your actual AMUA file name
    amua_file_path = base_dir / "sample" / "model" / amua_file_name # Adjust path if needed

    # Check if the sample file exists
    if not amua_file_path.exists():
        print(f"Error: Sample file not found at {amua_file_path}")
        print("Please place a sample .amua file in the ./sample directory or update the file_path.")
        sys.exit(1)

    # When the script is executed directly, call the main function
    project_model = get_project_model(amua_file_path)

    if project_model:
        # Display project summary
        display_project_summary(project_model)

        # --- Run Cohort Simulation for a Specific Markov Chain ---
        print("\n--- Starting Cohort Simulation ---")

        # Get the target Markov Chain for simulation
        target_mc_name_sim = "Average Treatment" # Choose an MC name to simulate
        markov_chain_to_simulate = project_model.get_markov_chain_by_name(target_mc_name_sim)

        if markov_chain_to_simulate:
            print(f"\nSerializing specific Markov Chain: '{markov_chain_to_simulate.name}'")
            # 特定のChainResultオブジェクトを関数に渡す
            get_tree_node_instances_json(markov_chain_to_simulate)

        else:
            print(f"\nMarkov Chain '{target_mc_name_sim}' not found for serialization example.")
            print(f"Available Markov Chains: {project_model.list_all_markov_chain_names()}")

        print("\n--- End of JSON Serialization Examples ---")
        # --- JSON Serialization 例 終了 ---
        # Create a dictionary to hold results for all simulated chains (even if only one)
        all_chain_results: Dict[str, ChainResult] = {}

        if markov_chain_to_simulate:
            print(f"Attempting to simulate Markov Chain: '{markov_chain_to_simulate.name}'")

            # Call the cohort simulation function, which now returns a ChainResult or None
            chain_result = run_cohort_simulation(
                markov_chain=markov_chain_to_simulate,
                project_model=project_model
            )

            # If simulation was successful, add the ChainResult to our collection
            if chain_result:
                all_chain_results[chain_result.chain_name] = chain_result
                print(f"Simulation for '{chain_result.chain_name}' completed.")
            else:
                print(f"Simulation for '{markov_chain_to_simulate.name}' failed.")


        else:
            print(f"Error: Markov Chain '{target_mc_name_sim}' not found for simulation.")
            print(f"Available Markov Chains: {project_model.list_all_markov_chain_names()}")

        print("\n--- Simulation Phase Complete ---")

        # --- Create the top-level SimulationResult object ---
        # Wrap the results from all simulated chains into a single SimulationResult object
        final_simulation_result = SimulationResult(
            project_name=project_model.name if project_model.name else "Unnamed Project",
            chain_results=all_chain_results, # Pass the dictionary of ChainResult objects
            project_dim_info=project_model.dim_info # Pass the DimInfo object
        )

        # --- Display the aggregated statistics using the updated display function ---
        # Pass the SimulationResult object to the display function
        display_aggregated_results(final_simulation_result)


    else:
        print("\n--- Parsing Failed ---")
        print(f"Could not construct the Project object from '{amua_file_path}'.")
        print("Please check for errors reported during parsing above.")

    print("\n--- Script Finished ---")

