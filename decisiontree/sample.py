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

# Add the current directory to sys.path to ensure model.py and parser.py can be imported
# This is helpful if you run the script directly.
# Adjust this if model.py/parser.py are in a different location relative to this script.
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Adding '{current_dir}' to sys.path for imports.") # Debugging line
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) # Add to the beginning of the path

try:
    # Import the main parsing function from parser.py
    # Note: This assumes parser.py correctly imports classes from model.py
    from parser import parse_amua_project

    # Optional: Import Project class for type hinting or explicit type checks
    # from model import Project

except ImportError as e:
    print(f"Fatal Error: Could not import parser.py or model.py. Please ensure they are in the correct location relative to this script.")
    print(f"Details: {e}")
    # Print sys.path for debugging import issues
    # print("Current sys.path:", sys.path)
    sys.exit(1) # Exit the script as essential functionality is missing


def get_variables(project_model):
    print("\n--- Parsing Completed Successfully ---")
    print("Project object successfully constructed.")

    # Display key information from the project object
    # ... (Previous summary prints) ...

    print(f"\nNumber of Tables Parsed: {len(project_model.tables)}")
    # print("Table Names:", list(project_model.tables.keys()))

    print(f"\nNumber of Global Parameters Parsed: {len(project_model.global_parameters)}")
    # Add call to display parameters table
    project_model.display_parameters() # <-- Call the existing method

    print(f"\nNumber of Global Variables Parsed: {len(project_model.global_variables)}")
    # Add call to display variables table
    project_model.display_variables() # <-- Call the new method

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

    markov_chain_names = project_model.list_all_markov_chain_names()
    markov_chain = project_model.get_markov_chain_by_name("Average Treatment")
    # get_tree_node_instances(markov_chain)


