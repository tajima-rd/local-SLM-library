# parser.py
# -*- coding: utf-8 -*-
"""
Parses an AMUA XML file and constructs the corresponding Project object model.
Reads global settings, metadata, dimensions, tables, parameters, variables,
and the decision tree/markov chain structure.
Assumes node index is determined by order within the <markov> block
and references in <childIndices> use this order index.
Identifies roots primarily by parentType -1, falling back to level 0.
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

# Import all necessary model classes from model.py
# Ensure model.py is accessible via sys.path or this is a relative import
try:
    from .model import (
        Project,
        Parameter, Variable, TreeNode, DecisionNode,
        MarkovChain, MarkovState, ChanceNode, StateTransition,
        DecisionTree, Dimension, DimensionInfo, Metadata,
        Table
    )
except ImportError as e:
    print(f"Error during initial import in parser.py: {e}")
    print("Please ensure model.py is accessible via sys.path and contains expected classes.")
    # Depending on execution context, you might need more robust import handling or exit here.
    # For typical usage with a main analysis script setting sys.path, this should work.
    # If running this script directly, ensure model.py is in the current directory.


# --- Helper functions to extract global definitions ---

def extract_parameters(root: ET.Element) -> Dict[str, Parameter]:
    """Extracts global parameters from the XML root."""
    parameters = {}
    # Search anywhere under root for <Parameter> tags
    for param_elem in root.findall(".//Parameter"):
        name = param_elem.findtext("name", "").strip()
        expr = param_elem.findtext("expression", "").strip()
        notes = param_elem.findtext("notes", "").strip()

        if name:
            param = Parameter(name, expr, notes)
            # Value evaluation happens later when context is available within MarkovChain
            parameters[name] = param
        # else: # Uncomment for detailed warnings about unnamed parameters
        #     print(f"Warning: Found Parameter tag with no name.")
    return parameters

def extract_variables(root: ET.Element) -> Dict[str, Variable]:
    """Extracts global variables from the XML root."""
    variables = {}
    # Search anywhere under root for <Variable> tags
    for var_elem in root.findall(".//Variable"):
        name = var_elem.findtext("name", "").strip()
        expr = var_elem.findtext("expression", "").strip()
        notes = var_elem.findtext("notes", "").strip()

        if name:
            var = Variable(name, expr, notes)
            # Value evaluation happens later when context is available within MarkovChain
            variables[name] = var
        # else: # Uncomment for detailed warnings about unnamed variables
        #     print(f"Warning: Found Variable tag with no name.")
    return variables

def parse_metadata(root: ET.Element) -> Optional[Metadata]:
    """
    Parses the <Metadata> section from the XML root.
    <Metadata> is typically a direct child of the root <Model> element.
    """
    metadata_elem = root.find("Metadata")

    if metadata_elem is None:
        # print("Warning: <Metadata> element not found in the XML.") # Uncomment for warning
        return None

    try:
        # Extract text content for each known metadata tag
        author = metadata_elem.findtext("author", default="")
        date_created = metadata_elem.findtext("dateCreated", default="")
        version_created = metadata_elem.findtext("versionCreated", default="")
        modifier = metadata_elem.findtext("modifier", default="")
        date_modified = metadata_elem.findtext("dateModified", default="")
        version_modified = metadata_elem.findtext("versionModified", default="")

        # Create the Metadata object, using None if the text was empty after stripping
        parsed_metadata = Metadata(
            author=author.strip() if author else None,
            date_created=date_created.strip() if date_created else None,
            version_created=version_created.strip() if version_created else None,
            modifier=modifier.strip() if modifier else None,
            date_modified=date_modified.strip() if date_modified else None,
            version_modified=version_modified.strip() if version_modified else None
        )
        # print(f"Successfully parsed Metadata: {parsed_metadata}") # Uncomment for logging
        return parsed_metadata

    except Exception as e:
        print(f"Error parsing Metadata section: {e}")
        return None # Return None if any error occurs during parsing

def parse_tables(root: ET.Element) -> Dict[str, Table]:
    """
    Parses all <Table> sections from the XML root.
    Returns a dictionary of Tables, keyed by table name.
    Tables can appear anywhere, searching using findall(".//Table").
    """
    tables: Dict[str, Table] = {}
    # Find all <Table> elements anywhere under the root
    for table_elem in root.findall(".//Table"):
        try:
            # --- Extract Table metadata ---
            name = table_elem.findtext("name", default="").strip()
            if not name:
                print(f"Warning: Found Table tag with no name. Skipping this table.")
                continue # Skip tables without a name

            # Use descriptive variable names to avoid conflict with Python built-ins
            table_type_str = table_elem.findtext("type", default="").strip()
            lookup_method_raw = table_elem.findtext("lookupMethod", default="").strip()
            # Set lookup_method to None if type is not 'Lookup' or method is empty
            lookup_method = lookup_method_raw if table_type_str == 'Lookup' and lookup_method_raw else None


            # Parse dimensions, handle missing/invalid values
            num_rows: int = -1 # Initialize with invalid default
            num_cols: int = -1 # Initialize with invalid default
            num_rows_text = table_elem.findtext("numRows", default="-1")
            num_cols_text = table_elem.findtext("numCols", default="-1")

            try:
                num_rows = int(num_rows_text)
            except ValueError:
                print(f"Warning: Invalid numRows value '{num_rows_text}' for table '{name}'. Will determine rows from data.")
                # Continue, num_rows will be determined from actual data rows

            try:
                num_cols = int(num_cols_text)
            except ValueError:
                 print(f"Warning: Invalid numCols value '{num_cols_text}' for table '{name}'. Will determine columns from first data row.")
                 # Continue, num_cols will be determined from first data row

            if num_rows < 0 and table_elem.find("data") is not None:
                 print(f"Warning: Negative numRows ({num_rows}) for table '{name}' but data found. Will determine rows from data.")
            if num_cols < 0 and table_elem.find("data/item") is not None:
                 print(f"Warning: Negative numCols ({num_cols}) for table '{name}' but data items found. Will determine columns from data.")


            # --- Extract headers ---
            headers = [elem.text.strip() for elem in table_elem.findall("headers") if elem.text is not None]
            # Validation against num_cols might be done later if num_cols is derived from data

            # --- Extract data ---
            data: List[List[float]] = []
            # Find all <data> elements that are children of *this specific* <Table> element
            data_row_elements = table_elem.findall("data")

            if num_rows >= 0 and len(data_row_elements) != num_rows:
                 print(f"Warning: Table '{name}' declared numRows={num_rows} but found {len(data_row_elements)} data rows. Using found rows count.")
                 # num_rows = len(data_row_elements) # Option to override declared num_rows

            actual_num_rows = len(data_row_elements) # Use actual count

            for row_elem in data_row_elements:
                row_data: List[float] = []
                item_elements = row_elem.findall("item")

                # Determine number of columns for this row, and potentially for the table
                current_row_cols = len(item_elements)
                if num_cols >= 0 and current_row_cols != num_cols:
                     print(f"Warning: Table '{name}' data row has {current_row_cols} items but declared numCols={num_cols}. Using found items for this row.")
                     # This row might have incorrect number of columns. Decide how to handle.
                     # For robustness, we'll take the items found, but this might lead to downstream errors.

                if num_cols < 0 and len(data) == 0 and current_row_cols > 0:
                    # If num_cols was not declared or invalid, infer from the first data row
                    num_cols = current_row_cols
                    if len(headers) != num_cols:
                         print(f"Warning: Table '{name}': Inferred numCols={num_cols} from data, but headers count ({len(headers)}) does not match.")


                for item_elem in item_elements:
                    if item_elem.text is not None:
                        try:
                            # Attempt to convert item text to float
                            row_data.append(float(item_elem.text.strip()))
                        except ValueError:
                            print(f"Warning: Non-numeric value '{item_elem.text.strip()}' found in table '{name}' data. Using 0.0.")
                            row_data.append(0.0) # Default to 0.0 for invalid numeric data
                    else:
                        # Handle empty item tag text
                        # print(f"Warning: Empty item tag found in table '{name}' data row. Using 0.0.") # Uncomment for warning
                        row_data.append(0.0) # Default to 0.0 for empty items

                # Add the parsed row to the data list if it has the expected number of items (or any items if num_cols was inferred)
                # Or simply append whatever was parsed:
                data.append(row_data)


            # --- Extract notes ---
            notes = table_elem.findtext("notes", default="").strip()
            notes = notes if notes else None # Store as None if empty


            # --- Create Table object and add to dictionary ---
            # Use actual parsed row count, and potentially inferred col count
            final_num_rows = len(data)
            final_num_cols = num_cols if num_cols >= 0 else (data[0].__len__() if data else 0) # Use inferred if declared was invalid

            # Final validation of headers vs actual column count
            if len(headers) != final_num_cols and final_num_cols > 0:
                 print(f"Warning: Table '{name}': Final column count ({final_num_cols}) differs from headers count ({len(headers)}).")


            table = Table(
                name=name,
                table_type=table_type_str,
                lookup_method=lookup_method,
                num_rows=final_num_rows,
                num_cols=final_num_cols,
                headers=headers,
                data=data,
                notes=notes
            )
            tables[name] = table # Add to the dictionary using the name as key
            # print(f"Parsed table: {name} ({final_num_rows}x{final_num_cols})") # Uncomment for logging

        except Exception as e:
            print(f"Error parsing a Table element (possibly named '{name}'): {e}. Skipping this table.")
            # Continue parsing other tables even if one fails

    # print(f"Successfully parsed {len(tables)} table(s).") # Uncomment for logging
    return tables

def parse_dim_info(root: ET.Element) -> Optional[DimensionInfo]:
    """
    Parses the <DimInfo> section from the XML root.
    <DimInfo> is typically a direct child of the root <Model> element.
    """
    dim_info_elem = root.find("DimInfo")

    if dim_info_elem is None:
        # print("Warning: <DimInfo> element not found in the XML.") # Uncomment for warning
        return None

    try:
        # 1. Extract dimension names, symbols, and decimals lists
        dim_names_list = [elem.text.strip() for elem in dim_info_elem.findall("dimNames") if elem.text is not None]
        dim_symbols_list = [elem.text.strip() for elem in dim_info_elem.findall("dimSymbols") if elem.text is not None]
        decimals_list_str = [elem.text.strip() for elem in dim_info_elem.findall("decimals") if elem.text is not None]

        # Convert decimals to integers, providing a default for invalid entries
        decimals_list = []
        for dec_str in decimals_list_str:
            try:
                decimals_list.append(int(dec_str))
            except ValueError:
                print(f"Warning: Invalid decimal value '{dec_str}' found in DimInfo decimals list. Using 0.")
                decimals_list.append(0)

        # 2. Create Dimension objects from the lists
        dimensions: List[Dimension] = []
        num_names = len(dim_names_list)
        # Iterate based on the number of names, using default values for missing symbol/decimal entries
        for i in range(num_names):
            name = dim_names_list[i]
            # Provide default empty string/zero if symbol/decimal list is shorter than name list
            symbol = dim_symbols_list[i] if i < len(dim_symbols_list) else ""
            decimals = decimals_list[i] if i < len(decimals_list) else 0
            dimensions.append(Dimension(name=name, symbol=symbol, decimals=decimals))

        # 3. Extract other single-value settings, handling potential errors and providing defaults
        # Helper for getting integer values safely
        def get_int(tag: str, default: int = -1) -> int:
            text = dim_info_elem.findtext(tag)
            if text is not None:
                try:
                    return int(text.strip())
                except ValueError:
                    # print(f"Warning: Invalid integer value '{text}' for tag <{tag}> in DimInfo. Using default {default}.") # Uncomment for warning
                    pass # Use default if conversion fails
            return default

        # Helper for getting float values safely
        def get_float(tag: str, default: float = 0.0) -> float:
             text = dim_info_elem.findtext(tag)
             if text is not None:
                  try:
                       return float(text.strip())
                  except ValueError:
                       # print(f"Warning: Invalid float value '{text}' for tag <{tag}> in DimInfo. Using default {default}.") # Uncomment for warning
                       pass # Use default if conversion fails
             return default


        analysis_type = get_int("analysisType", default=0) # Default analysis type is often 0
        objective = get_int("objective", default=0)       # Default objective is often 0
        # Use -1 for dimension indices if tag is missing or invalid, as it indicates 'not set' or invalid index
        objective_dim_index = get_int("objectiveDim", default=-1)
        cost_dim_index = get_int("costDim", default=-1)
        effect_dim_index = get_int("effectDim", default=-1)
        wtp = get_float("WTP", default=0.0)
        extended_dim_index = get_int("extendedDim", default=-1)

        # 4. Create the DimensionInfo object
        parsed_dim_info = DimensionInfo(
            dimensions=dimensions,
            analysis_type=analysis_type,
            objective=objective,
            objective_dim_index=objective_dim_index,
            cost_dim_index=cost_dim_index,
            effect_dim_index=effect_dim_index,
            wtp=wtp,
            extended_dim_index=extended_dim_index
        )
        # print(f"Successfully parsed DimInfo: {parsed_dim_info}") # Uncomment for logging
        return parsed_dim_info

    except Exception as e:
        print(f"Error parsing DimInfo section: {e}")
        return None # Return None if any unexpected error occurs during parsing

def parse_markov_settings(markov_elem: ET.Element) -> Dict[str, Union[int, float, bool, str]]:
    """
    Parses settings directly under a <markov> element.
    """
    settings: Dict[str, Union[int, float, bool, str]] = {}

    # Helper for getting integer values safely from markov elem
    def get_int(tag: str, default: Optional[int] = None) -> Optional[int]:
         text = markov_elem.findtext(tag)
         if text is not None:
              try: return int(text.strip())
              except ValueError: pass # Return default if conversion fails
         return default

    # Helper for getting float values safely from markov elem
    def get_float(tag: str, default: Optional[float] = None) -> Optional[float]:
         text = markov_elem.findtext(tag)
         if text is not None:
              try: return float(text.strip())
              except ValueError: pass # Return default if conversion fails
         return default

    # Helper for getting boolean values safely from markov elem ('true'/'false')
    def get_bool(tag: str, default: Optional[bool] = None) -> Optional[bool]:
         text = markov_elem.findtext(tag)
         if text is not None:
              text_lower = text.strip().lower()
              if text_lower == 'true': return True
              if text_lower == 'false': return False
              # print(f"Warning: Invalid boolean value '{text}' for tag <{tag}> in <markov>. Using default {default}.") # Uncomment for warning
         return default

    # Parse known settings tags
    settings['max_cycles'] = get_int("maxCycles")
    settings['state_decimals'] = get_int("stateDecimals")
    settings['half_cycle_correction'] = get_bool("halfCycleCorrection")
    settings['discount_rewards'] = get_bool("discountRewards")
    settings['discount_start_cycle'] = get_int("discountStartCycle")
    settings['cycles_per_year'] = get_float("cyclesPerYear")
    settings['show_trace'] = get_bool("showTrace")
    settings['compile_traces'] = get_bool("compileTraces")

    # Add any other direct children of <markov> as raw data if needed
    # for child in markov_elem:
    #     if child.tag not in settings and child.text is not None:
    #         settings[child.tag] = child.text.strip()

    # Filter out None values if you prefer not to store them
    # settings = {k: v for k, v in settings.items() if v is not None}

    return settings

# --- New Helper function to parse a single Markov block into a DecisionTree ---
# --- Helper function to parse a single Markov block into a DecisionTree ---
def _parse_single_decision_tree_block(
    markov_elem: ET.Element,
    markov_elem_index: int,
    global_parameters: Dict[str, Parameter],
    global_variables: Dict[str, Variable],
    dim_info: Optional[DimensionInfo] # DimInfo を引数として受け取る
) -> Optional[DecisionTree]:
    """
    Parses a single <markov> XML element and constructs a DecisionTree object from it.

    Args:
        markov_elem: The XML Element object for the <markov> block.
        markov_elem_index: The 0-based index of this <markov> element in the file (for logging).
        global_parameters: Dictionary of global parameters to attach to MCs.
        global_variables: Dictionary of global variables to attach to MCs.
        dim_info: The parsed DimensionInfo object (needed to map costs/rewards to dimensions).

    Returns:
        A DecisionTree object if successfully parsed, None otherwise.
    """
    print(f"  Parsing markov block {markov_elem_index}...")

    # --- Step 2a: Parse markov-specific settings ---
    markov_settings = parse_markov_settings(markov_elem)

    # --- Step 2b: Get <Node> elements and map index to XML element ---
    markov_nodes_list_in_block = list(markov_elem.findall("Node"))
    if not markov_nodes_list_in_block:
        print(f"  Warning: No <Node> elements found under <markov> block {markov_elem_index}. Skipping tree construction.")
        return None

    index_to_elem_in_block: Dict[int, ET.Element] = {idx: elem for idx, elem in enumerate(markov_nodes_list_in_block)}


    # --- Step 2c: Create all node objects for *this specific* block (Pass 1) ---
    nodes_in_this_block: Dict[int, TreeNode] = {}
    for idx, elem in enumerate(markov_nodes_list_in_block):
        node_type_text = elem.findtext("type", default="-1")
        try: node_type = int(node_type_text)
        except ValueError: print(f"  Warning: Invalid node type '{node_type_text}' for node at index {idx} in block {markov_elem_index}. Skipping creation."); continue

        name = elem.findtext("name", default=f"Node_{idx}_Block{markov_elem_index}").strip()
        level = int(elem.findtext("level", default="-1"))
        parent_type = int(elem.findtext("parentType", default="-1"))

        # Collect miscellaneous data, excluding tags handled specifically below
        misc_data = {child.tag: child.text.strip() if child.text is not None else ""
                     for child in elem
                     # Added 'varUpdates' to exclude list here
                     if child.tag not in ['type', 'name', 'level', 'parentType', 'childIndices', 'transition', 'prob', 'terminationCondition', 'stateNames', 'cost', 'rewards', 'varUpdates',
                                         'xPos', 'yPos', 'width', 'height', 'parentX', 'parentY', 'visible', 'collapsed']}


        node: Optional[TreeNode] = None
        if node_type == 0: # DecisionNode
            node = DecisionNode(index=idx, name=name, level=level, parent_type=parent_type, data=misc_data)
        elif node_type == 1: # MarkovChain
            termination_raw = elem.findtext("terminationCondition", default="t==5").strip()
            termination_condition = termination_raw if termination_raw else "t==5"
            node = MarkovChain(index=idx, name=name, level=level, parent_type=parent_type, termination_condition=termination_condition, data=misc_data)
        elif node_type == 2: # MarkovState
            node = MarkovState(index=idx, name=name, level=level, parent_type=parent_type, data=misc_data) # Create MarkovState

            # --- パース処理: MarkovState のコスト、報酬、変数更新タグ ---
            # DimInfo が利用可能であれば、コストと報酬を構造化
            if dim_info:
                 cost_elements = elem.findall("cost")
                 reward_elements = elem.findall("rewards")

                 # コスト式の格納: DimInfo の次元数に合わせてループ
                 for i, dim in enumerate(dim_info.dimensions):
                     cost_text = cost_elements[i].text.strip() if i < len(cost_elements) and cost_elements[i].text is not None else "0"
                     node.cost_expressions[dim.name] = cost_text
                     # Optional: Warn if fewer cost tags than dimensions...

                 # 報酬式の格納: DimInfo の次元数に合わせてループ
                 for i, dim in enumerate(dim_info.dimensions):
                     reward_text = reward_elements[i].text.strip() if i < len(reward_elements) and reward_elements[i].text is not None else "0"
                     node.reward_expressions[dim.name] = reward_text
                     # Optional: Warn if fewer rewards tags than dimensions...

            # 変数更新式の格納: タグの出現順序でリストに格納 (DimInfo がなくてもパース可能)
            var_updates_elements = elem.findall("varUpdates")
            # Assign to the new variable_updates attribute
            node.variable_updates = [update_elem.text.strip() for update_elem in var_updates_elements if update_elem.text is not None]

            # Note: If DimInfo was missing, cost_expressions and reward_expressions will remain empty {}.


        elif node_type == 3: # ChanceNode
            node = ChanceNode(index=idx, name=name, level=level, parent_type=parent_type, data=misc_data)
        elif node_type == 4: # StateTransition
            transition_target = elem.findtext("transition", default="").strip()
            prob_text = elem.findtext("prob", default="").strip()
            misc_data['prob'] = prob_text # Still store prob in data for StateTransition's own method
            node = StateTransition(index=idx, name=name, level=level, parent_type=parent_type, transition=transition_target, data=misc_data)


        if node:
             nodes_in_this_block[idx] = node
             # ### Removed: all_nodes_from_all_blocks logic ###


    if not nodes_in_this_block:
         print(f"  Warning: No valid node objects created for markov block {markov_elem_index}. Skipping tree construction.")
         return None


    # --- Step 2d: Attach global parameters and variables to MarkovChains in *this* block ---
    for node in nodes_in_this_block.values():
         if isinstance(node, MarkovChain):
             node.parameters = global_parameters
             node.variables = global_variables


    # --- Step 2e: Build tree structure by linking parent/child relationships *within* this block (Pass 2) ---
    # ... (Linking logic is the same) ...
    for idx, parent_node in list(nodes_in_this_block.items()):
        elem = index_to_elem_in_block.get(idx)
        if elem is None: print(f"  Internal Error: Cannot find original XML element for node index {idx} ('{parent_node.name if parent_node.name else 'Unnamed'}') in block {markov_elem_index}. Skipping child linking."); continue
        child_indices_elements = elem.findall("childIndices")
        child_indices_values: List[int] = []
        for ci_elem in child_indices_elements:
            if ci_elem.text is not None and ci_elem.text.strip().isdigit():
                child_idx_val = int(ci_elem.text.strip())
                if 0 <= child_idx_val < len(markov_nodes_list_in_block): child_indices_values.append(child_idx_val)

        for child_idx in child_indices_values:
            child_node = nodes_in_this_block.get(child_idx)
            if child_node:
                 child_node.parent = parent_node
                 if isinstance(parent_node, DecisionNode) and isinstance(child_node, MarkovChain): parent_node.markov_chains.append(child_node)
                 elif isinstance(parent_node, MarkovChain) and isinstance(child_node, MarkovState): parent_node.markov_states.append(child_node)
                 elif isinstance(parent_node, MarkovState):
                     if isinstance(child_node, ChanceNode): parent_node.chance_nodes.append(child_node)
                     elif isinstance(child_node, StateTransition): parent_node.state_transitions.append(child_node)
                 elif isinstance(parent_node, ChanceNode) and isinstance(child_node, StateTransition): parent_node.state_transitions.append(child_node)


    # --- Step 2f: Identify the root node(s) *within this block* ---
    # ... (Root finding logic is the same) ...
    root_nodes_in_block = [ node for node in nodes_in_this_block.values() if node.parent_type == -1 or (node.parent is None and node.level == 0) ]
    if not root_nodes_in_block:
         root_nodes_in_block = [ node for node in nodes_in_this_block.values() if node.level == 0 ]
         if not root_nodes_in_block:
              print(f"  Warning: No root nodes found within block {markov_elem_index}. Skipping tree construction.")
              return None

    # --- Step 2g: Build DecisionTree object for *this* block ---
    primary_root_node = root_nodes_in_block[0]

    tree = DecisionTree(
        root_index=primary_root_node.index,
        nodes=nodes_in_this_block, # Pass the dictionary of nodes for THIS block
        **markov_settings # Unpack the dictionary as keyword arguments for settings
    )

    if isinstance(primary_root_node, DecisionNode): tree.markov_chains = primary_root_node.markov_chains
    elif isinstance(primary_root_node, MarkovChain): tree.markov_chains = [primary_root_node]

    print(f"  Successfully parsed Decision Tree (Root: '{primary_root_node.name if primary_root_node.name else 'Unnamed'}', Index {primary_root_node.index}) from block {markov_elem_index}.")

    return tree


# --- parse_amua_project 関数 (unchanged) ---
# This function calls _parse_single_decision_tree_block and collects results.
def parse_amua_project(file_path: str | Path) -> Optional[Project]:
    """
    Parses an AMUA XML file and constructs the corresponding Project object model.
    ... (Docstring continues) ...
    """
    print(f"Parsing file: {file_path}")
    try:
        file_path_str = str(file_path) if isinstance(file_path, Path) else file_path
        tree_xml = ET.parse(file_path_str)
        root = tree_xml.getroot()
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); return None
    except ET.ParseError as e: print(f"Error parsing XML file {file_path}: {e}"); return None
    except Exception as e: print(f"An unexpected error occurred while reading or parsing {file_path}: {e}"); return None

    # --- Step 1: Parse global sections from the root <Model> element ---
    project_name = root.findtext("name", default="Unnamed Project").strip()

    def get_root_int(tag: str) -> Optional[int]: text = root.findtext(tag); return int(text.strip()) if text is not None and text.strip().isdigit() else None
    def get_root_bool(tag: str) -> Optional[bool]: text = root.findtext(tag); return text.strip().lower() == 'true' if text is not None else None

    project_type = get_root_int("type"); project_scale = get_root_int("scale"); align_right = get_root_bool("alignRight"); sim_param_sets = get_root_bool("simParamSets"); sim_type = get_root_int("simType"); cohort_size = get_root_int("cohortSize"); crn = get_root_bool("CRN"); crn_seed = get_root_int("crnSeed"); display_ind_results = get_root_bool("displayIndResults"); num_threads = get_root_int("numThreads"); report_subgroups = get_root_bool("reportSubgroups")

    parsed_dim_info = parse_dim_info(root) # Call to parse DimInfo
    parsed_metadata = parse_metadata(root)
    parsed_tables = parse_tables(root)
    global_parameters = extract_parameters(root)
    global_variables = extract_variables(root)

    # --- Step 2: Parse each <markov> element into a DecisionTree using a helper function ---
    markov_elements = root.findall(".//markov")
    decision_trees: List[DecisionTree] = []

    if not markov_elements:
        print(f"Warning: No <markov> element found in {file_path}. Returning Project with no trees.")
    else:
        for markov_elem_index, markov_elem in enumerate(markov_elements):
            # Call the helper function to parse a single markov block into a DecisionTree
            # Pass the global parameters, variables, and DimInfo to the helper
            tree = _parse_single_decision_tree_block(
                markov_elem=markov_elem,
                markov_elem_index=markov_elem_index,
                global_parameters=global_parameters,
                global_variables=global_variables,
                dim_info=parsed_dim_info # Pass the parsed DimInfo object here
            )
            if tree:
                 # Add the successfully parsed tree to the list
                 decision_trees.append(tree)
            # Note: Warnings/Errors during block parsing are handled within the helper function.


    # --- Step 3: Create the final Project object and return it ---
    project = Project(
        name=project_name, model_type=project_type, scale=project_scale,
        align_right=align_right, sim_param_sets=sim_param_sets, sim_type=sim_type,
        cohort_size=cohort_size, crn=crn, crn_seed=crn_seed,
        display_ind_results=display_ind_results, num_threads=num_threads,
        report_subgroups=report_subgroups, metadata=parsed_metadata,
        dim_info=parsed_dim_info, tables=parsed_tables,
        global_parameters=global_parameters, global_variables=global_variables,
        decision_trees=decision_trees
    )

    # Optional: Add final logging/summary
    print(f"Successfully parsed Project '{project.name if project.name else 'Unnamed'}'.")
    print(f"  Model Type: {project.model_type}, Sim Type: {project.sim_type}, Cohort Size: {project.cohort_size}")
    print(f"  Found {len(project.decision_trees)} Decision Tree(s).")
    if project.decision_trees:
        first_tree = project.decision_trees[0]
        # Safely get root node name
        root_node = first_tree.get_node(first_tree.root_index)
        root_name = root_node.name if root_node else 'None'
        print(f"    First Tree (Root: '{root_name}', Index {first_tree.root_index}):")
        # Print some markov settings from the DecisionTree object
        print(f"      Markov Settings: max_cycles={first_tree.max_cycles}, cycles_per_year={first_tree.cycles_per_year}, discount_rewards={first_tree.discount_rewards}, ...")
        # Optional: Print costs/rewards/var_updates for a state if possible
        if first_tree.markov_chains:
            first_mc = first_tree.markov_chains[0]
            if first_mc.markov_states:
                 # Find a state with costs/rewards/varUpdates for better logging example
                 state_with_data = next((s for s in first_mc.markov_states if s.cost_expressions or s.reward_expressions or s.variable_updates), None)
                 if state_with_data:
                     print(f"    Example State with Data ('{state_with_data.name}', Index {state_with_data.index}):")
                     print(f"      Costs Parsed: {state_with_data.cost_expressions}")
                     print(f"      Rewards Parsed: {state_with_data.reward_expressions}")
                     print(f"      Variable Updates Parsed: {state_with_data.variable_updates}")
                 else:
                     print(f"    No State with costs, rewards, or variable updates found in the first MC.")


    print(f"  Found {len(project.tables)} Table(s).")
    print(f"  Found {len(project.global_parameters)} Parameter(s) and {len(project.global_variables)} Variable(s).")
    if project.dim_info: print(f"  DimInfo parsed with {len(project.dim_info.dimensions)} dimensions.")
    if project.metadata: print(f"  Metadata parsed.")


    # Return the final Project object
    return project

# --- Main parsing function ---
# This function parses the entire file and returns a single Project object.
def parse_amua_project(file_path: str | Path) -> Optional[Project]:
    """
    Parses an AMUA XML file and constructs the corresponding Project object model.

    Reads global settings, metadata, dimensions, tables, parameters, variables,
    and the decision tree/markov chain structure (parsed per block).

    Args:
        file_path: Path to the .amua XML file.

    Returns:
        A Project object if parsing is successful, None otherwise (e.g., file not found, major parse error).
    """
    print(f"Parsing file: {file_path}")
    try:
        # Ensure file_path is a string for ET.parse
        file_path_str = str(file_path) if isinstance(file_path, Path) else file_path
        tree_xml = ET.parse(file_path_str)
        root = tree_xml.getroot()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None # Return None on critical file error
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return None # Return None on critical XML parse error
    except Exception as e:
        print(f"An unexpected error occurred while reading or parsing {file_path}: {e}")
        return None # Return None on any other unexpected reading/parsing error

    # --- Step 1: Parse global sections from the root <Model> element ---
    # These sections are typically direct children of the root <Model> and apply globally.

    # Basic Project settings from root
    project_name = root.findtext("name", default="Unnamed Project").strip()

    # Helper for parsing integer values from root, using None if invalid/missing
    def get_root_int(tag: str) -> Optional[int]:
         text = root.findtext(tag)
         if text is not None:
              try:
                   return int(text.strip())
              except ValueError:
                   # print(f"Warning: Invalid integer value '{text}' for root tag <{tag}>. Using None.") # Uncomment for warning
                   pass # Return None if conversion fails
         return None

    # Helper for parsing boolean values from root ('true'/'false'), using None if invalid/missing text
    def get_root_bool(tag: str) -> Optional[bool]:
         text = root.findtext(tag)
         if text is not None:
              text_lower = text.strip().lower()
              if text_lower == 'true': return True
              if text_lower == 'false': return False
              # print(f"Warning: Invalid boolean value '{text}' for root tag <{tag}>. Using None.") # Uncomment for warning
         return None


    # Parse the various settings using the helpers
    # Note: Default values are handled by the Project class constructor, not here.
    project_type = get_root_int("type")
    project_scale = get_root_int("scale")
    align_right = get_root_bool("alignRight")
    sim_param_sets = get_root_bool("simParamSets")
    sim_type = get_root_int("simType")
    cohort_size = get_root_int("cohortSize")
    crn = get_root_bool("CRN") # Tag name is "CRN" in the example
    crn_seed = get_root_int("crnSeed")
    display_ind_results = get_root_bool("displayIndResults")
    num_threads = get_root_int("numThreads") # Model default might be 1 if tag is missing
    report_subgroups = get_root_bool("reportSubgroups")


    # Parse standard global sections using their dedicated helper functions
    # These helper functions are defined before this function.
    parsed_dim_info = parse_dim_info(root) # <-- Call to parse DimInfo
    parsed_metadata = parse_metadata(root)
    parsed_tables = parse_tables(root)
    global_parameters = extract_parameters(root)
    global_variables = extract_variables(root)


    # --- Step 2: Parse each <markov> element into a DecisionTree using a helper function ---
    # Find all <markov> elements anywhere under the root.
    markov_elements = root.findall(".//markov")
    decision_trees: List[DecisionTree] = []

    if not markov_elements:
        print(f"Warning: No <markov> element found in {file_path}. Returning Project with no trees.")
    else:
        # Iterate through each <markov> element found in the file
        for markov_elem_index, markov_elem in enumerate(markov_elements):
            # Call the helper function to parse a single markov block into a DecisionTree
            # Pass the global parameters, variables, and DimInfo to the helper
            tree = _parse_single_decision_tree_block(
                markov_elem=markov_elem,
                markov_elem_index=markov_elem_index,
                global_parameters=global_parameters,
                global_variables=global_variables,
                dim_info=parsed_dim_info # <-- Pass the parsed DimInfo object here
            )
            if tree:
                 # Add the successfully parsed tree to the list
                 decision_trees.append(tree)
            # Note: Warnings/Errors during block parsing are handled within the helper function.


    # --- Step 3: Create the final Project object and return it ---
    # The Project object aggregates all the parsed information from the entire file.
    project = Project(
        name=project_name,
        model_type=project_type,
        scale=project_scale,
        align_right=align_right,
        sim_param_sets=sim_param_sets,
        sim_type=sim_type,
        cohort_size=cohort_size,
        crn=crn,
        crn_seed=crn_seed,
        display_ind_results=display_ind_results,
        num_threads=num_threads,
        report_subgroups=report_subgroups,
        metadata=parsed_metadata,
        dim_info=parsed_dim_info, # Assign the parsed DimInfo
        tables=parsed_tables,     # Assign the parsed tables dictionary
        global_parameters=global_parameters, # Assign global parameters dictionary
        global_variables=global_variables,   # Assign global variables dictionary
        decision_trees=decision_trees # Assign the list of built DecisionTrees
    )

    # Optional: Add final logging/summary
    print(f"Successfully parsed Project '{project.name if project.name else 'Unnamed'}'.")
    print(f"  Model Type: {project.model_type}, Sim Type: {project.sim_type}, Cohort Size: {project.cohort_size}")
    print(f"  Found {len(project.decision_trees)} Decision Tree(s).")
    # Print settings for the first tree if any found
    if project.decision_trees:
        first_tree = project.decision_trees[0]
        # Safely get root node name
        root_node = first_tree.get_node(first_tree.root_index)
        root_name = root_node.name if root_node else 'None'
        print(f"    First Tree (Root: '{root_name}', Index {first_tree.root_index}):")
        # Print some markov settings from the DecisionTree object
        print(f"      Markov Settings: max_cycles={first_tree.max_cycles}, cycles_per_year={first_tree.cycles_per_year}, discount_rewards={first_tree.discount_rewards}, ...")
        # Optional: Print costs/rewards for a state if possible (requires knowing a state index, or iterating)
        # For example, find the start state of the first MC in the first tree
        if first_tree.markov_chains:
            first_mc = first_tree.markov_chains[0]
            if first_mc.markov_states:
                 first_state = first_mc.markov_states[0]
                 print(f"    First State ('{first_state.name}', Index {first_state.index}):")
                 print(f"      Costs Parsed: {first_state.cost_expressions}")
                 print(f"      Rewards Parsed: {first_state.reward_expressions}")


    print(f"  Found {len(project.tables)} Table(s).")
    print(f"  Found {len(project.global_parameters)} Parameter(s) and {len(project.global_variables)} Variable(s).")
    if project.dim_info: print(f"  DimInfo parsed with {len(project.dim_info.dimensions)} dimensions.")
    if project.metadata: print(f"  Metadata parsed.")


    # Return the final Project object
    return project