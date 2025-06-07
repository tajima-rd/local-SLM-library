# amua_model.py
# -*- coding: utf-8 -*-
"""
Defines the data models (classes) for representing the AMUA Decision Tree and Markov Chain structure.
"""

import random # Needed by safe_eval, which is called by methods in this module
import re # Needed by safe_eval
from typing import Optional, List, Dict, Any
import tabulate
import json # <--- Add this import

# Lazy import utility functions to avoid circular dependency issues on initial load
# These will be assigned by _lazy_import_utils when first needed.
safe_eval = None
complete_direct_state_transition_complement = None

def _lazy_import_utils():
    """Lazy import utility functions."""
    global safe_eval, complete_direct_state_transition_complement
    if safe_eval is None: # Check if already imported
        try:
            # Use relative import if this is part of a package
            from .utils import safe_eval, complete_direct_state_transition_complement
        except ImportError:
             # Fallback for standalone execution or testing, though might fail if utils aren't in path
             print("Warning: Could not import amua_utils. Safe eval and complement functions may not work.")
             # Define dummy functions if needed to prevent crashes immediately
             safe_eval = lambda expr, context={}: None
             complete_direct_state_transition_complement = lambda state: None

# --- 新しく追加する Metadata クラス ---
class Metadata:
    """Represents the Metadata section of the AMUA file."""
    def __init__(
        self,
        author: Optional[str] = None,
        date_created: Optional[str] = None,
        version_created: Optional[str] = None,
        modifier: Optional[str] = None,
        date_modified: Optional[str] = None,
        version_modified: Optional[str] = None
    ):
        # Using Optional[str] because these tags might be missing in some files
        self.author: Optional[str] = author
        self.date_created: Optional[str] = date_created
        self.version_created: Optional[str] = version_created
        self.modifier: Optional[str] = modifier
        self.date_modified: Optional[str] = date_modified
        self.version_modified: Optional[str] = version_modified

    def __repr__(self):
        # A concise representation
        parts = []
        if self.author: parts.append(f"author='{self.author}'")
        if self.version_modified: parts.append(f"ver='{self.version_modified}'")
        elif self.version_created: parts.append(f"ver='{self.version_created}'")
        if self.date_modified: parts.append(f"mod_date='{self.date_modified}'")
        elif self.date_created: parts.append(f"cre_date='{self.date_created}'")

        return f"Metadata({', '.join(parts)})"

class Dimension:
    """Represents a single dimension (e.g., Cost, QALY) from the DimInfo section."""
    def __init__(self, name: str, symbol: str = "", decimals: int = 0):
        self.name = name
        self.symbol = symbol
        self.decimals = decimals

    def __repr__(self):
        return f"Dimension(name='{self.name}', symbol='{self.symbol}', decimals={self.decimals})"

class DimensionInfo:
    """Represents the DimInfo section defining analysis dimensions and settings."""
    def __init__(
        self,
        dimensions: List[Dimension], # <-- Dimensionオブジェクトのリストに変更
        analysis_type: int,
        objective: int,
        objective_dim_index: int, # <-- インデックスとして保持
        cost_dim_index: int,      # <-- インデックスとして保持
        effect_dim_index: int,    # <-- インデックスとして保持
        wtp: float,
        extended_dim_index: int   # <-- インデックスとして保持
    ):
        self.dimensions: List[Dimension] = dimensions
        self.analysis_type: int = analysis_type
        self.objective: int = objective
        self.objective_dim_index: int = objective_dim_index
        self.cost_dim_index: int = cost_dim_index
        self.effect_dim_index: int = effect_dim_index
        self.wtp: float = wtp
        self.extended_dim_index: int = extended_dim_index

        # 寸法情報の整合性を簡単なチェック (インデックスがリスト範囲内かなど)
        num_dims = len(self.dimensions)
        if not (0 <= self.cost_dim_index < num_dims):
             print(f"Warning: Cost dimension index {self.cost_dim_index} is out of bounds [0, {num_dims-1}].")
        if not (0 <= self.effect_dim_index < num_dims):
             print(f"Warning: Effect dimension index {self.effect_dim_index} is out of bounds [0, {num_dims-1}].")
        # 他のインデックスについても同様にチェック可能

    def get_dimension_by_index(self, index: int) -> Optional[Dimension]:
        """Returns a Dimension object by its index."""
        if 0 <= index < len(self.dimensions):
            return self.dimensions[index]
        return None

    def get_cost_dimension(self) -> Optional[Dimension]:
        """Returns the Dimension object representing the cost."""
        return self.get_dimension_by_index(self.cost_dim_index)

    def get_effect_dimension(self) -> Optional[Dimension]:
        """Returns the Dimension object representing the effect."""
        return self.get_dimension_by_index(self.effect_dim_index)

    def get_objective_dimension(self) -> Optional[Dimension]:
        """Returns the Dimension object related to the objective."""
        return self.get_dimension_by_index(self.objective_dim_index)

    # 必要に応じて、他のインデックスに対応する getter を追加

    def __repr__(self):
        dim_names = [d.name for d in self.dimensions]
        return (f"DimensionInfo(dim_names={dim_names}, analysis_type={self.analysis_type}, "
                f"cost_dim_index={self.cost_dim_index}, effect_dim_index={self.effect_dim_index})")

class Table:
    """Represents a Table section from the AMUA file."""
    def __init__(
        self,
        name: str,
        table_type: str, # Changed from 'type' to 'table_type' to avoid conflict with Python's built-in type()
        lookup_method: Optional[str], # Optional as type might not be Lookup
        num_rows: int,
        num_cols: int,
        headers: List[str],
        data: List[List[float]], # Table data as list of rows, each row is list of floats
        notes: Optional[str] = None
    ):
        self.name: str = name
        self.table_type: str = table_type
        self.lookup_method: Optional[str] = lookup_method
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.headers: List[str] = headers
        self.data: List[List[float]] = data
        self.notes: Optional[str] = notes

        # Basic validation (optional but recommended)
        if len(headers) != num_cols:
             print(f"Warning: Table '{name}' has numCols={num_cols} but {len(headers)} headers.")
        if len(data) != num_rows:
             print(f"Warning: Table '{name}' has numRows={num_rows} but {len(data)} data rows.")
        for i, row in enumerate(data):
             if len(row) != num_cols:
                  print(f"Warning: Table '{name}' row {i} has numCols={num_cols} but {len(row)} items.")


    def get_value(self, lookup_key: float, key_col_index: int = 0, value_col_index: int = 1) -> Optional[float]:
        """
        Performs a lookup based on lookup_method.
        Assumes the key column is sorted in ascending order.
        Currently only implements 'Truncate' method.
        """
        if self.table_type != 'Lookup':
             print(f"Error: Cannot perform lookup on table '{self.name}' with type '{self.table_type}'.")
             return None
        if not (0 <= key_col_index < self.num_cols) or not (0 <= value_col_index < self.num_cols):
             print(f"Error: Invalid key or value column index for table '{self.name}'.")
             return None
        if not self.data:
             print(f"Warning: Table '{self.name}' has no data for lookup.")
             return None

        if self.lookup_method == 'Truncate':
            # Find the row where the key column value is <= lookup_key, taking the largest such index
            best_match_row: Optional[List[float]] = None
            for row in self.data:
                try:
                    row_key = row[key_col_index]
                    if row_key <= lookup_key:
                        best_match_row = row
                    else:
                        # Since we assume the key column is sorted, we can stop once the key exceeds lookup_key
                        break
                except IndexError:
                    # Should not happen with validation, but defensive check
                    print(f"Error: Index out of bounds in table data row for '{self.name}'.")
                    return None
                except (ValueError, TypeError):
                    print(f"Error: Non-numeric value in key column for table '{self.name}'.")
                    return None

            if best_match_row:
                 try:
                      return best_match_row[value_col_index]
                 except IndexError:
                      print(f"Error: Value column index out of bounds in best match row for '{self.name}'.")
                      return None
                 except (ValueError, TypeError):
                      print(f"Error: Non-numeric value in value column for table '{self.name}'.")
                      return None
            else:
                 # If lookup_key is smaller than the first key in the table
                 print(f"Warning: Lookup key {lookup_key} is smaller than the first key in table '{self.name}'. No match found by Truncate.")
                 return None # Or return a default like 0 or the first value depending on desired behavior

        # Add other lookup methods here (e.g., 'Interpolate', 'Nearest', etc.)
        # elif self.lookup_method == 'Interpolate':
        #     # ... interpolation logic ...
        #     pass
        else:
            print(f"Error: Unsupported lookupMethod '{self.lookup_method}' for table '{self.name}'.")
            return None

    def __repr__(self):
        return (f"Table(name='{self.name}', type='{self.table_type}', "
                f"lookup_method='{self.lookup_method}', num_rows={self.num_rows}, "
                f"num_cols={self.num_cols}, headers={self.headers})") # Data omitted for brevity

class Parameter:
    """Represents a parameter defined in the AMUA model."""
    def __init__(self, name: str, expression: str, notes: str = ""):
        self.name = name
        self.expression = expression
        self.notes = notes
        # Value is not calculated at init, needs context later
        self.value: float | None = None

    def evaluate_expression(self, context: Dict[str, float] = {}) -> float | None:
        """Evaluates the parameter's expression using the given context."""
        _lazy_import_utils()
        if safe_eval:
            # Parameter expressions might depend on other parameters/variables, pass context
            self.value = safe_eval(self.expression, context)
            return self.value
        return None

    def __repr__(self):
        # Avoid evaluating just for repr, show expression
        return f"Parameter(name='{self.name}', expression='{self.expression}')"

class Variable:
    """Represents a variable defined in the AMUA model."""
    def __init__(self, name: str, expression: str, notes: str = ""):
        self.name = name
        self.expression = expression
        self.notes = notes
        # Value is not calculated at init, needs context later
        self.value: float | None = None

    def evaluate_expression(self, context: Dict[str, float] = {}) -> float | None:
        """Evaluates the variable's expression using the given context."""
        _lazy_import_utils()
        if safe_eval:
            # Variable expressions might depend on parameters/other variables, pass context
            self.value = safe_eval(self.expression, context)
            return self.value
        return None

    def __repr__(self):
        # Avoid evaluating just for repr, show expression
        return f"Variable(name='{self.name}', expression='{self.expression}')"

class TreeNode:
    """Base class for all nodes in the Decision Tree / Markov Chain structure."""
    def __init__(self, index: int, name: str, level: int, node_type: int, parent_type: int, data: Dict[str, str] = None):
        self.index = index
        self.name = name
        self.level = level
        self.node_type = node_type # 0:Decision, 1:MarkovChain, 2:MarkovState, 3:Chance, 4:StateTransition
        self.parent_type = parent_type # Type of the parent node
        self.data = data if data else {} # Raw data from XML
        self.parent: Optional[TreeNode] = None # Reference to parent node

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the node for JSON serialization."""
        # Basic attributes common to all nodes
        node_dict = {
            "type": self.__class__.__name__, # Include the class name for clarity in JSON
            "index": self.index,
            "name": self.name,
            "level": self.level,
            "node_type_id": self.node_type, # Use node_type_id to avoid conflict with 'type'
            "parent_type_id": self.parent_type,
            "data": self.data, # Include raw data
            "children": [] # Placeholder for child nodes
        }
        return node_dict

    def __repr__(self):
         return f"{self.__class__.__name__}(index={self.index}, name='{self.name}', type={self.node_type})"

class DecisionNode(TreeNode):
    """Represents a Decision Node (Type 0) in the tree."""
    def __init__(
        self,
        index: int,
        name: str,
        level: int,
        parent_type: int,
        markov_chains: List["MarkovChain"] = None,
        data: Dict[str, str] = None
    ):
        super().__init__(index, name, level, node_type=0, parent_type=parent_type, data=data)
        self.markov_chains: List[MarkovChain] = markov_chains if markov_chains is not None else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation including its Markov Chains."""
        node_dict = super().to_dict()
        # Recursively convert child MarkovChains
        node_dict["markov_chains"] = [mc.to_dict() for mc in self.markov_chains]
        # Note: We append markov_chains to the main dict, not the generic "children" list
        # as the structure is specific (DecisionNode -> markov_chains list)
        # You could append to "children" if you prefer a generic structure, but specific keys are clearer.
        # Let's keep specific keys like "markov_chains"
        return node_dict

class MarkovChain(TreeNode):
    """Represents a Markov Chain (Type 1)."""
    def __init__(
        self,
        index: int,
        name: str,
        level: int,
        parent_type: int,
        termination_condition: str = "t==1",
        markov_states: List["MarkovState"] = None,
        data: Dict[str, str] = None
    ):
        super().__init__(index, name, level, node_type=1, parent_type=parent_type, data=data)
        self.termination_condition: str = termination_condition
        self.markov_states: List[MarkovState] = markov_states if markov_states is not None else []
        # Parameters and variables are typically global in AMUA, but stored per MC instance here
        self.variables: Dict[str, Variable] = {}
        self.parameters: Dict[str, Parameter] = {}

    def get_context(self) -> Dict[str, float]:
        """
        Evaluates parameters and variables to create a context dictionary.
        Evaluation order might matter if variables depend on parameters or each other.
        A simple iterative approach is used here.
        """
        _lazy_import_utils()
        context = {}
        # Combine parameters and variables for evaluation
        eval_items = list(self.parameters.values()) + list(self.variables.values())
        evaluated_names = set()
        # Simple iterative evaluation (handles basic dependencies)
        # More complex dependencies might require topological sort
        for _ in range(len(eval_items) + 1): # Iterate slightly more than needed to catch dependencies
             updated_in_iter = False
             for item in eval_items:
                 if item.name not in evaluated_names:
                      val = item.evaluate_expression(context)
                      if val is not None:
                          context[item.name] = val
                          evaluated_names.add(item.name)
                          updated_in_iter = True
             if not updated_in_iter and len(evaluated_names) == len(eval_items):
                  break # No new items evaluated, all possible are done

        # Report items that failed to evaluate
        # for item in eval_items:
        #      if item.name not in evaluated_names:
        #           print(f"Warning: Could not evaluate '{item.name}' expression '{item.expression}'.")

        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation including its states."""
        node_dict = super().to_dict()
        node_dict["termination_condition"] = self.termination_condition
        # Optionally include parameter/variable info (expressions, maybe initial values)
        # Omitting evaluated values as they change during simulation and can be large objects
        node_dict["parameters"] = {name: p.expression for name, p in self.parameters.items()}
        node_dict["variables"] = {name: v.expression for name, v in self.variables.items()}

        # Recursively convert child MarkovStates
        node_dict["markov_states"] = [ms.to_dict() for ms in self.markov_states]
        # Again, using specific key "markov_states"
        return node_dict

    def resolve_transitions(self):
        """
        Evaluates all transition probabilities and resolves next state references
        within this Markov Chain.
        Should be called before simulation.
        """
        _lazy_import_utils()
        if safe_eval is None or complete_direct_state_transition_complement is None:
             print("Error: amua_utils not loaded. Cannot resolve transitions.")
             return

        context = self.get_context()
        # print(f"Resolution context for {self.name}: {context}")

        # Map state names to state objects for quick lookup
        state_map = {state.name: state for state in self.markov_states}

        # Resolve probabilities and next state references
        for state in self.markov_states:
            # Direct transitions from state
            for st in state.state_transitions:
                st.next_state = state_map.get(st.transition)
                if st.next_state is None:
                     print(f"Warning: StateTransition '{st.name}' targets unknown state '{st.transition}'.")
                st.resolve_probability(context) # Evaluate probability expression

            # Apply complement logic for direct transitions
            complete_direct_state_transition_complement(state)

            # Transitions via chance nodes
            for cn in state.chance_nodes:
                for st in cn.state_transitions:
                     st.next_state = state_map.get(st.transition)
                     if st.next_state is None:
                         print(f"Warning: StateTransition '{st.name}' targets unknown state '{st.transition}'.")
                     st.resolve_probability(context) # Evaluate probability expression
                # Apply complement logic for chance node transitions
                cn.complete_complement_probability()

class MarkovState(TreeNode):
    """Represents a Markov State (Type 2)."""
    def __init__(
        self,
        index: int,
        name: str,
        level: int,
        parent_type: int,
        chance_nodes: List["ChanceNode"] = None,
        state_transitions: List["StateTransition"] = None, # Direct transitions from this state
        data: Dict[str, str] = None # data will still contain other miscellaneous raw data
    ):
        super().__init__(index, name, level, node_type=2, parent_type=parent_type, data=data)
        self.chance_nodes: List[ChanceNode] = chance_nodes if chance_nodes is not None else []
        self.state_transitions: List[StateTransition] = state_transitions if state_transitions is not None else [] # Direct transitions

        # --- 新しく追加する属性 ---
        # DimInfo の次元名をキーとするコスト式の辞書 (値は文字列式)
        self.cost_expressions: Dict[str, str] = {}
        # DimInfo の次元名をキーとする報酬式の辞書 (値は文字列式)
        self.reward_expressions: Dict[str, str] = {}
        # 変数更新式のリスト (値は文字列式)
        self.variable_updates: List[str] = []

        # Note: Position/Visual data (xPos, yPos, etc.) can remain in the 'data' dict or be moved to dedicated attributes.
        # For simulation purposes, the expressions attributes are key.
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation including its costs, rewards, updates, and children."""
        node_dict = super().to_dict()
        node_dict["cost_expressions"] = self.cost_expressions
        node_dict["reward_expressions"] = self.reward_expressions
        node_dict["variable_updates"] = self.variable_updates # List of strings

        # Recursively convert child ChanceNodes and direct StateTransitions
        # These are the structural children in the XML hierarchy sense
        node_dict["children"] = [cn.to_dict() for cn in self.chance_nodes] + \
                                [st.to_dict() for st in self.state_transitions]
        return node_dict

class ChanceNode(TreeNode):
    """Represents a Chance Node (Type 3) within a Markov State."""
    def __init__(
        self,
        index: int,
        name: str,
        level: int,
        parent_type: int,
        state_transitions: List["StateTransition"] = None,
        data: Dict[str, str] = None
    ):
        super().__init__(index, name, level, node_type=3, parent_type=parent_type, data=data)
        self.state_transitions: List[StateTransition] = state_transitions if state_transitions is not None else []

    def complete_complement_probability(self):
        """Distributes remaining probability among complement transitions in this ChanceNode."""
        # Note: This logic is also in amua_utils.complete_direct_state_transition_complement
        # Consider if this method should just call the utils function, or if CN needs its own.
        # Given it's identical logic but applies to ChanceNode's state_transitions,
        # having it here as a method is also reasonable. Let's keep it here for now.
        total = 0.0
        complement_nodes = []

        for st in self.state_transitions:
            if st.data.get("prob") == "C":
                complement_nodes.append(st)
            elif st.prob is not None:
                total += st.prob

        remaining = 1.0 - total

        # Handle floating point errors and negative remainders
        if remaining < -1e-9:
            print(f"[!] Warning: total transition probabilities ({total:.4f}) exceed 1.0 in ChanceNode '{self.name}' (index {self.index}). Remaining set to 0.")
            remaining = 0.0
        elif remaining < 0: # Treat small negative as 0
             remaining = 0.0

        if complement_nodes:
            if remaining > 1e-9: # Only distribute if there's something left
                 per_node = remaining / len(complement_nodes)
                 for st in complement_nodes:
                     st.prob = per_node
            else: # No remaining probability, set complement nodes to 0
                 for st in complement_nodes:
                     st.prob = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation including its transitions."""
        node_dict = super().to_dict()
        # Recursively convert child StateTransitions
        node_dict["children"] = [st.to_dict() for st in self.state_transitions]
        return node_dict

class StateTransition(TreeNode):
    """Represents a State Transition (Type 4) from a State or Chance node."""
    def __init__(
        self,
        index: int,
        name: str,
        level: int,
        parent_type: int,
        transition: str = "", # Name of the target MarkovState
        data: Dict[str, str] = None
    ):
        super().__init__(index, name, level, node_type=4, parent_type=parent_type, data=data)
        self.transition: str = transition  # The name of the target MarkovState
        self.next_state: Optional[MarkovState] = None  # Resolved reference to the target MarkovState object
        self.prob: Optional[float] = None  # Resolved probability after evaluation and complement

    def resolve_probability(self, context: Dict[str, float]) -> None:
        """Evaluates the probability expression using the given context."""
        _lazy_import_utils()
        prob_expr = self.data.get("prob", "")
        if prob_expr == "C":
            self.prob = None  # Mark as complement, to be filled later
            return

        if safe_eval:
             val = safe_eval(prob_expr, context)
             if val is not None:
                 # Ensure probability is between 0 and 1
                 self.prob = max(0.0, min(1.0, val))
             else:
                 print(f"[!] Failed to evaluate prob expression '{prob_expr}' for StateTransition '{self.name}' (index {self.index}). Setting prob to 0.")
                 self.prob = 0.0 # Default to 0 if evaluation fails
        else:
            print(f"[!] amua_utils not loaded. Cannot evaluate probability '{prob_expr}' for StateTransition '{self.name}'. Setting prob to 0.")
            self.prob = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation including its target and probability."""
        # StateTransitions don't have children in the structural tree sense (they reference back to states)
        # So, we use the base to_dict but override the "children" list potentially added by the base.
        node_dict = super().to_dict()
        # Remove the empty "children" list if the base adds it and this node type doesn't have structural children
        node_dict.pop("children", None)

        node_dict["transition_target_name"] = self.transition # Target state name (string)
        node_dict["resolved_probability"] = self.prob # Resolved probability (float or None)

        # Optionally include the raw probability expression if useful
        # node_dict["probability_expression"] = self.data.get("prob", "")

        return node_dict

class DecisionTree:
    """
    Represents a Decision Tree structure, typically starting with a DecisionNode
    or a MarkovChain, contained within a Project.
    """
    def __init__(
        self,
        root_index: Optional[int] = None,
        nodes: Dict[int, "TreeNode"] = None,
        # Simulation settings specific to the <markov> block this tree came from
        max_cycles: Optional[int] = None,
        state_decimals: Optional[int] = None,
        half_cycle_correction: Optional[bool] = None,
        discount_rewards: Optional[bool] = None,
        discount_start_cycle: Optional[int] = None,
        cycles_per_year: Optional[float] = None,
        show_trace: Optional[bool] = None,
        compile_traces: Optional[bool] = None,
    ):
        self.nodes: Dict[int, TreeNode] = nodes if nodes is not None else {}
        self.root_index: Optional[int] = root_index
        self.markov_chains: List[MarkovChain] = [] # Populated during parsing

        # Simulation settings
        self.max_cycles: Optional[int] = max_cycles
        self.state_decimals: Optional[int] = state_decimals
        self.half_cycle_correction: Optional[bool] = half_cycle_correction
        self.discount_rewards: Optional[bool] = discount_rewards
        self.discount_start_cycle: Optional[int] = discount_start_cycle
        self.cycles_per_year: Optional[float] = cycles_per_year
        self.show_trace: Optional[bool] = show_trace
        self.compile_traces: Optional[bool] = compile_traces


    def add_node(self, node: "TreeNode"):
        """Adds a node to the tree's node dictionary."""
        self.nodes[node.index] = node

    def get_node(self, index: int) -> Optional["TreeNode"]:
        """Retrieves a node by its index from THIS tree's nodes."""
        return self.nodes.get(index)

    def get_markov_chains(self) -> List["MarkovChain"]:
        """
        Returns the top-level MarkovChains associated with this tree structure.
        If the root is a DecisionNode, it returns its children MarkovChains.
        If the root is a MarkovChain, it returns itself.
        """
        # This list is populated by the parser during tree building
        return self.markov_chains

    def __repr__(self):
        root_name = self.get_node(self.root_index).name if self.root_index is not None and self.get_node(self.root_index) else 'None'
        return (f"DecisionTree(root='{root_name}', num_nodes={len(self.nodes)}, "
                f"max_cycles={self.max_cycles}, cycles_per_year={self.cycles_per_year})")

class Project:
    """
    Represents the entire AMUA file structure, corresponding to the <Model> root element.
    Contains global settings, metadata, dimensions, tables, and the decision tree(s).
    """
    def __init__(
        self,
        name: Optional[str] = None,
        model_type: Optional[int] = None, # Changed from 'type' to 'model_type'
        scale: Optional[int] = None,
        align_right: Optional[bool] = None, # Using bool for boolean value
        sim_param_sets: Optional[bool] = None,
        sim_type: Optional[int] = None,
        cohort_size: Optional[int] = None,
        crn: Optional[bool] = None,
        crn_seed: Optional[int] = None,
        display_ind_results: Optional[bool] = None,
        num_threads: Optional[int] = None,
        report_subgroups: Optional[bool] = None,
        # References to other top-level parsed objects
        metadata: Optional["Metadata"] = None,
        dim_info: Optional["DimensionInfo"] = None,
        tables: Dict[str, "Table"] = None,
        global_parameters: Dict[str, "Parameter"] = None,
        global_variables: Dict[str, "Variable"] = None,
        # The main model structure (can be one or more trees/chains)
        decision_trees: List["DecisionTree"] = None # Project can contain one or more trees
    ):
        self.name: Optional[str] = name
        self.model_type: Optional[int] = model_type
        self.scale: Optional[int] = scale
        self.align_right: Optional[bool] = align_right
        self.sim_param_sets: Optional[bool] = sim_param_sets
        self.sim_type: Optional[int] = sim_type
        self.cohort_size: Optional[int] = cohort_size
        self.crn: Optional[bool] = crn
        self.crn_seed: Optional[int] = crn_seed
        self.display_ind_results: Optional[bool] = display_ind_results
        self.num_threads: Optional[int] = num_threads
        self.report_subgroups: Optional[bool] = report_subgroups

        # Global sections data
        self.metadata: Optional[Metadata] = metadata
        self.dim_info: Optional[DimensionInfo] = dim_info
        self.tables: Dict[str, Table] = tables if tables is not None else {}
        self.global_parameters: Dict[str, Parameter] = global_parameters if global_parameters is not None else {}
        self.global_variables: Dict[str, Variable] = global_variables if global_variables is not None else {}

        # Model structure
        self.decision_trees: List[DecisionTree] = decision_trees if decision_trees is not None else []

    def get_markov_chains(self) -> List["MarkovChain"]:
        """Helper to get all MarkovChains from all trees in this project."""
        all_mcs = []
        for tree in self.decision_trees:
             all_mcs.extend(tree.get_markov_chains())
        return all_mcs
    
    def get_markov_chain_by_name(self, name: str) -> Optional["MarkovChain"]:
        """
        Finds a MarkovChain object in the project by its name.

        Args:
            name: The name of the MarkovChain to find.

        Returns:
            The MarkovChain object if found, otherwise None.
        """
        # Use the existing helper method to get all MC objects
        all_mcs = self.get_markov_chains()

        # Iterate through the list to find the one with the matching name
        for mc in all_mcs: # 変数名を mc に変更
            if mc.name == name:
                return mc # Return the object once found

        # If the loop completes without finding a match, return None explicitly
        return None # None を返り値に含めるため Optional を使用

    def display_parameters(self):
        """
        Displays the project's global parameters in a table format.
        """
        print(f"--- Global Parameters for Project '{self.name if self.name else 'Unnamed'}' ---")
        parameters_list = list(self.global_parameters.values())

        if not parameters_list:
            print("  No global parameters found.")
        else:
            parameter_data = []
            for param in parameters_list:
                # Use the stored 'value' if available, otherwise show "N/A"
                value_str = f"{param.value:.4f}" if isinstance(param.value, (int, float)) else "N/A"
                parameter_data.append([
                    param.name,
                    param.expression,
                    param.notes,
                    value_str
                ])

            headers = ["Name", "Expression", "Notes", "Value"]
            # Use tabulate to print the table
            print(tabulate.tabulate(parameter_data, headers=headers, tablefmt="grid"))
        print("-" * (len("--- Global Parameters for Project '' ---") + len(self.name if self.name else 'Unnamed')))

    def display_variables(self):
        """
        Displays the project's global variables in a table format.
        """
        print(f"--- Global Variables for Project '{self.name if self.name else 'Unnamed'}' ---")
        variables_list = list(self.global_variables.values())

        if not variables_list:
            print("  No global variables found.")
        else:
            variable_data = []
            for var in variables_list:
                # Use the stored 'value' if available, otherwise show "N/A"
                # Note: This value is the *initial* evaluated value before simulation cycles change it.
                value_str = f"{var.value:.4f}" if isinstance(var.value, (int, float)) else "N/A"
                variable_data.append([
                    var.name,
                    var.expression, # Shows the *initial* expression
                    var.notes,
                    value_str # Shows the *initial* evaluated value
                ])

            headers = ["Name", "Expression", "Notes", "Initial Value"] # Clarify "Initial Value"
            # Use tabulate to print the table
            print(tabulate.tabulate(variable_data, headers=headers, tablefmt="grid"))
        print("-" * (len("--- Global Variables for Project '' ---") + len(self.name if self.name else 'Unnamed')))

    # ここから新しく追加するメソッド
    def list_all_markov_chain_names(self) -> List[str]:
        """
        Gets a list of names of all MarkovChains across all DecisionTrees
        in this project.

        Returns:
            A list of strings, where each string is the name of a MarkovChain.
        """
        all_mcs = self.get_markov_chains() # Use the existing helper method to get all MC objects
        chain_names = [mc.name for mc in all_mcs] # Extract names using a list comprehension
        return chain_names
    
    def __repr__(self):
         return f"Project(name='{self.name}', sim_type={self.sim_type}, num_trees={len(self.decision_trees)}, num_tables={len(self.tables)})"

