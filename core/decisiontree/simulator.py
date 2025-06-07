# amua_simulator.py
# -*- coding: utf-8 -*-
"""
Handles the simulation of Markov Chains based on the parsed model.
Simulates individual paths, evaluates costs/rewards/variable updates per cycle,
and accumulates results per dimension. Handles table lookups in expressions.
"""

import random
import re
from typing import List, Dict, Optional, Any, Tuple, Union
import math # math functions might be used in expressions (e.g., math.log, math.exp)
import sys # For error handling
import tabulate
import statistics

# Import necessary model classes
# Assuming model.py is in the same directory or sys.path is set up correctly
try:
    # Need DecisionTree class here to access block-level settings
    from .model import MarkovChain, MarkovState, StateTransition, ChanceNode, Project, DimensionInfo, Table, Parameter, Variable, DecisionTree
except ImportError as e:
    print(f"Fatal Error importing model classes in simulator.py: {e}")
    print("Please ensure model.py is accessible via sys.path.")
    # Exit hard if essential model classes can't be imported
    sys.exit(1)

# Lazy import utility functions (safe_eval and complete_direct_state_transition_complement)
# These should be defined in utils.py
safe_eval = None
complete_direct_state_transition_complement = None

def _lazy_import_utils():
    """Lazy import utility functions."""
    global safe_eval, complete_direct_state_transition_complement
    if safe_eval is None: # Check if already imported
        try:
            # Use relative import if part of a package, or direct if utils.py is in the same dir
            from .utils import safe_eval, complete_direct_state_transition_complement
        except ImportError:
             print("Fatal Error: Could not import amua_utils in simulator. Safe eval and complement functions are essential.")
             # Define dummy functions that will cause clear errors if called
             safe_eval = lambda expr, context={}: (_ for _ in ()).throw(ImportError("safe_eval not loaded")), None # Force an error if called
             complete_direct_state_transition_complement = lambda state, context: (_ for _ in ()).throw(ImportError("complement_direct_state_transition_complement not loaded"))
             # Exit if essential utils are missing
             sys.exit(1)

class DimensionStats:
    """
    Holds aggregated statistics for a single dimension across all individuals
    in a Markov Chain simulation.
    """
    def __init__(
        self,
        dimension_name: str,
        average: float,
        std_dev: float,
        min_val: float,
        max_val: float,
        # Add other stats here if calculated, e.g., percentiles: Dict[float, float] = None
    ):
        self.dimension_name: str = dimension_name
        self.average: float = average
        self.std_dev: float = std_dev
        self.min_val: float = min_val
        self.max_val: float = max_val
        # self.percentiles = percentiles if percentiles is not None else {}

    def __repr__(self):
        return (f"DimensionStats(name='{self.dimension_name}', avg={self.average:.4f}, "
                f"std={self.std_dev:.4f}, min={self.min_val:.4f}, max={self.max_val:.4f})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easier display/serialization."""
        return {
            "dimension_name": self.dimension_name,
            "average": self.average,
            "std_dev": self.std_dev,
            "min_val": self.min_val,
            "max_val": self.max_val,
            # Include percentiles if added: "percentiles": self.percentiles
        }

class IndividualState:
    """
    Represents the state and cumulative results for a single individual
    simulated through a Markov Chain.
    """
    def __init__(self, start_state_obj: MarkovState, initial_variables: Dict[str, float], dim_info: Optional[DimensionInfo]):
        self.current_state_obj: MarkovState = start_state_obj # Reference to the state object
        self.current_state_name: str = start_state_obj.name
        self.current_cycle: int = 0 # Current cycle number (0-based)
        self.is_terminated: bool = False # Flag to stop simulation for this individual

        # Variables that change during simulation (e.g., timeSick, timeChemo)
        # Start with initial global variable values provided at the start of simulation
        self.variables: Dict[str, float] = initial_variables.copy()

        # Cumulative costs and rewards for each dimension
        # Keys are dimension names (from DimInfo), values are accumulated totals
        self.cumulative_values: Dict[str, float] = {}
        if dim_info:
             # Initialize cumulative values for all dimensions to 0.0
             for dim in dim_info.dimensions:
                 self.cumulative_values[dim.name] = 0.0
        
        self.cycles_simulated: int = 0
        self.state_occupancy: Dict[str, int] = {start_state_obj.name: 1}

    def get_evaluation_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        eval_context = base_context.copy()
        eval_context['t'] = self.current_cycle
        eval_context.update(self.variables)
        eval_context['currentState'] = self.current_state_name
        return eval_context


    def add_cycle_values(self, dim_info: Optional[DimensionInfo], evaluation_context: Dict[str, Any]) -> None:
         """
         Evaluates costs and rewards expressions for the current state
         and adds the results to the cumulative totals.
         Assumes evaluation_context contains necessary variables, t, lookup func, etc.
         """
         _lazy_import_utils() # Ensure safe_eval is loaded
         if safe_eval is None or dim_info is None:
              # Error message printed during import or project parsing
              return # Cannot evaluate without safe_eval or DimInfo

         current_state = self.current_state_obj
         # Use .get() with a default expression "0" to handle states missing cost/reward nodes in XML
         cost_exprs = current_state.cost_expressions # Dict[DimName, ExprString]
         reward_exprs = current_state.reward_expressions # Dict[DimName, ExprString]

         # Evaluate cost expressions
         # Iterate over dimension names from cumulative_values keys (ensures we cover all dimensions initialized)
         for dim_name in self.cumulative_values.keys():
             expr = cost_exprs.get(dim_name, "0") # Get expression for this dimension, default to "0" if missing

             if expr and expr.strip() != "0": # Only evaluate if expression is not empty or literal "0"
                 try:
                     value = safe_eval(expr, evaluation_context) # Evaluate the string expression
                     if isinstance(value, (int, float)):
                         # Add evaluated value to cumulative total for this dimension
                         self.cumulative_values[dim_name] += value
                         # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Added cost '{expr}' ({value:.4f}) to {dim_name}. Total {dim_name}: {self.cumulative_values[dim_name]:.4f}") # Debugging
                     # else: Optionally warn about non-numeric result if not None
                 except Exception as e:
                     # Log error if expression evaluation fails
                     print(f"  [!] Error evaluating cost expression '{expr}' for dimension '{dim_name}' in state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Using 0.0 for this cycle.")


         # Evaluate reward expressions
         # Iterate over dimension names from cumulative_values keys
         for dim_name in self.cumulative_values.keys():
              expr = reward_exprs.get(dim_name, "0") # Get expression for this dimension, default to "0"

              if expr and expr.strip() != "0": # Only evaluate if expression is not empty or literal "0"
                 try:
                     value = safe_eval(expr, evaluation_context) # Evaluate the string expression
                     if isinstance(value, (int, float)):
                         # Add evaluated value to cumulative total for this dimension
                         self.cumulative_values[dim_name] += value
                         # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Added reward '{expr}' ({value:.4f}) to {dim_name}. Total {dim_name}: {self.cumulative_values[dim_name]:.4f}") # Debugging
                     # else: Optionally warn about non-numeric result if not None
                 except Exception as e:
                      # Log error if expression evaluation fails
                      print(f"  [!] Error evaluating reward expression '{expr}' for dimension '{dim_name}' in state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Using 0.0 for this cycle.")

    def apply_variable_updates(self, base_evaluation_context: Dict[str, Any]) -> None:
        """
        Evaluates and applies variable update expressions defined for the current state.
        Handles basic 'varName++' syntax and relies on safe_eval for general expressions.
        Updates values in self.variables.
        Crucially, this method updates self.variables *in place*. The context for
        subsequent cycles (or potentially later expressions in the same update list,
        depending on AMUA's exact semantics) will reflect these changes.
        """
        _lazy_import_utils() # Ensure safe_eval is loaded
        if safe_eval is None:
             # Error message printed during import or project parsing
             return

        current_state = self.current_state_obj
        update_expressions = current_state.variable_updates # List of ExprStrings

        if not update_expressions:
             return # No updates to apply for this state

        # Create a *working copy* of the evaluation context for this cycle's updates.
        # This context will be updated *within* this loop as variables are assigned.
        # It must reflect 't' and variable values *at the start* of this cycle's updates.
        # Note: This context is used *only* within this update loop.
        # The context for the *next* cycle's calculations will be rebuilt based on `self.variables`.
        update_context = self.get_evaluation_context(base_evaluation_context)


        for expr in update_expressions:
            if not expr or not expr.strip(): continue # Skip empty or whitespace-only strings

            try:
                # --- Handling variable update expressions ---
                # These are assignments. Standard `eval` is unsafe.
                # safe_eval needs to support evaluation of the RHS, and we apply the assignment.

                match_assign = re.match(r'^\s*([\w_]+)\s*=\s*(.+)$', expr)
                match_increment = re.match(r'^\s*([\w_]+)\s*\+\+\s*$', expr) # Pattern for 'varName++'

                if match_increment:
                     var_name = match_increment.group(1)
                     # Need to check if the variable exists *in the context* (i.e., is a known variable)
                     if var_name in self.variables: # Check individual variables
                          self.variables[var_name] += 1.0 # Increment by 1.0
                          # Update the context *immediately* for subsequent expressions in this list
                          update_context[var_name] = self.variables[var_name]
                          # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Incremented variable {var_name} to {self.variables[var_name]}.") # Debugging
                     else:
                          # If variable not in self.variables, check if it's in the base context (e.g., global param)
                          if var_name in base_evaluation_context:
                               print(f"  Warning: Attempted to increment global parameter/unknown variable '{var_name}' in update '{expr}' for state '{self.current_state_name}', cycle {self.current_cycle}. Global parameters cannot be incremented. Update ignored.")
                          else:
                               # Decide behavior for unknown variable: Initialize to 1.0? Ignore?
                               # Amua might auto-initialize. Let's initialize to 1.0 and warn.
                               print(f"  Warning: Attempted to increment unknown variable '{var_name}' in update '{expr}' for state '{self.current_state_name}', cycle {self.current_cycle}. Initializing to 1.0.")
                               self.variables[var_name] = 1.0 # Initialize
                               update_context[var_name] = self.variables[var_name] # Update context
                     continue # Processed this expression


                elif match_assign:
                     var_name = match_assign.group(1)
                     value_expr = match_assign.group(2) # The expression on the right-hand side

                     # Evaluate the RHS expression using the *current* update_context
                     # This context reflects variables updated by *previous* expressions in this list.
                     try:
                          value = safe_eval(value_expr, update_context) # Evaluate the right-hand side
                          if isinstance(value, (int, float)):
                               # Check if it's a known variable (must be in initial variables)
                               if var_name in self.variables:
                                   self.variables[var_name] = value # Apply the assignment
                                   # Update the context *immediately* for subsequent expressions in this list
                                   update_context[var_name] = value
                                   # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Assigned variable {var_name} to {value} based on '{value_expr}'.") # Debugging
                               elif var_name in base_evaluation_context:
                                    print(f"  Warning: Attempted to assign value to global parameter '{var_name}' in update '{expr}' for state '{self.current_state_name}', cycle {self.current_cycle}. Global parameters cannot be assigned. Update ignored.")
                               else:
                                    # Decide behavior for unknown variable: Initialize? Ignore?
                                    # Amua might auto-initialize. Let's initialize and warn.
                                    print(f"  Warning: Attempted to assign value to unknown variable '{var_name}' in update '{expr}' for state '{self.current_state_name}', cycle {self.current_cycle}. Initializing with value {value}.")
                                    self.variables[var_name] = value # Initialize
                                    update_context[var_name] = value # Update context


                          else:
                               print(f"  Warning: Assignment expression '{expr}' for variable '{var_name}' evaluated to non-numeric value: {value}. Skipping update for state '{self.current_state_name}', cycle {self.current_cycle}.")
                     except Exception as e:
                          print(f"  [!] Error evaluating assignment expression '{value_expr}' for variable '{var_name}' in state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Skipping update.")
                     continue # Processed this expression

                else:
                     # If the expression didn't match known update patterns (like ++ or =)
                     print(f"  Warning: Unhandled variable update expression syntax for state '{self.current_state_name}', cycle {self.current_cycle}: '{expr}'. Skipping update.")

            except Exception as e:
                # Catch unexpected errors during the update processing loop itself
                print(f"  [!] Critical error during variable update processing for state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Skipping remaining updates for this state.")
                break # Exit loop for updates for this state

class ChainResult:
    """
    Holds simulation results and statistics for a single Markov Chain.
    """
    def __init__(
        self,
        chain_name: str,
        simulated_individuals_count: int,
        successful_simulated_count: int,
        max_cycles: int,
        start_state_name: str,
        dimension_stats: Dict[str, DimensionStats], # Dict[DimName, DimensionStats object]
        average_cycles_simulated: float, # 平均生存サイクル数
        average_state_occupancy: Dict[str, float], # 各状態の平均滞在サイクル数 Dict[StateName, AverageCycles]
        cycles_per_year: float,
        termination_reason_counts: Dict[str, int] = None # 終了理由の内訳Dict[Reason, Count] (今回は保留)
    ):
        self.chain_name: str = chain_name
        self.simulated_individuals_count: int = simulated_individuals_count
        self.successful_simulated_count: int = successful_simulated_count
        self.max_cycles: int = max_cycles
        self.start_state_name: str = start_state_name
        self.dimension_stats: Dict[str, DimensionStats] = dimension_stats
        self.average_cycles_simulated: float = average_cycles_simulated
        self.average_state_occupancy: Dict[str, float] = average_state_occupancy
        self.cycles_per_year: float = cycles_per_year # Store cycles per year
        self.termination_reason_counts = termination_reason_counts if termination_reason_counts is not None else {}


        # --- Methods for Analysis (Optional) ---
        # Example: Get stats for a specific dimension
    def get_stats_for_dimension(self, dim_name: str) -> Optional[DimensionStats]:
        return self.dimension_stats.get(dim_name)

        # Example: Calculate ICER against a comparator chain (requires cost and effect dimensions)
        # This is more complex as it needs another ChainResult object.
        # def calculate_icer(self, comparator_result: 'ChainResult', cost_dim_name: str, effect_dim_name: str):
        #     # ... ICER calculation logic using self.dimension_stats and comparator_result.dimension_stats ...
        #     pass

    def __repr__(self):
         num_dims = len(self.dimension_stats)
         return (f"ChainResult(name='{self.chain_name}', simulated={self.simulated_individuals_count}, "
                 f"successful={self.successful_simulated_count}, max_cycles={self.max_cycles}, "
                 f"start_state='{self.start_state_name}', num_dimensions_with_stats={num_dims})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easier display/serialization."""
        return {
            "chain_name": self.chain_name,
            "simulated_individuals_count": self.simulated_individuals_count,
            "successful_simulated_count": self.successful_simulated_count,
            "max_cycles": self.max_cycles,
            "start_state_name": self.start_state_name,
            "dimension_stats": {name: stats.to_dict() for name, stats in self.dimension_stats.items()},
            "average_cycles_simulated": self.average_cycles_simulated,
            "average_state_occupancy": self.average_state_occupancy,
            "cycles_per_year": self.cycles_per_year,
            "termination_reason_counts": self.termination_reason_counts,
        }

class SimulationResult:
    """
    Holds the aggregated simulation results and statistics for one or more
    Markov Chains within a Project simulation run. This is the top-level result object.
    """
    def __init__(
        self,
        project_name: str,
        # Store ChainResult objects keyed by chain name
        chain_results: Dict[str, ChainResult],
        project_dim_info: Optional[DimensionInfo] = None, # Keep a reference to DimInfo for formatting
        # Add other overall simulation info, e.g., timestamp, run settings summary
    ):
        self.project_name: str = project_name
        self.chain_results: Dict[str, ChainResult] = chain_results
        self.project_dim_info: Optional[DimensionInfo] = project_dim_info

    def get_chain_result(self, chain_name: str) -> Optional[ChainResult]:
         return self.chain_results.get(chain_name)

    # Example: Display summary for all chains
    # def display_summary(self):
    #     print(f"--- Simulation Results for Project: {self.project_name} ---")
    #     for chain_name, result in self.chain_results.items():
    #         print(f"\nChain: {chain_name}")
    #         print(f"  Simulated: {result.simulated_individuals_count}, Successful: {result.successful_simulated_count}, Max Cycles: {result.max_cycles}")
    #         print("  Dimension Statistics:")
    #         for dim_name, stats in result.dimension_stats.items():
    #             print(f"    {dim_name}: Avg={stats.average:.4f}, Std={stats.std_dev:.4f}, Min={stats.min_val:.4f}, Max={stats.max_val:.4f}")
    #     print("-" * 40)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire simulation result structure to a dictionary."""
        return {
            "project_name": self.project_name,
            "chain_results": {name: result.to_dict() for name, result in self.chain_results.items()},
            # Optional: Include a summary of DimInfo, not the whole object
            "dim_info_summary": {
                "dimensions": [d.to_dict() for d in self.project_dim_info.dimensions] if self.project_dim_info else [],
                "cost_dim_index": self.project_dim_info.cost_dim_index if self.project_dim_info else -1,
                "effect_dim_index": self.project_dim_info.effect_dim_index if self.project_dim_info else -1,
                # ... add other DimInfo summary fields ...
            }
            # Include other simulation-level attributes if added
        }

def simulate_individual_path(
    markov_chain: MarkovChain,
    start_state_obj: MarkovState,
    max_cycles: int,
    project_dim_info: Optional[DimensionInfo],
    project_tables: Dict[str, Table],
    initial_global_parameters: Dict[str, float],
    initial_global_variables: Dict[str, float]
) -> Optional[IndividualState]: # <-- Now returns IndividualState object
    """
    Simulates a single path through the Markov Chain for one individual.
    Records cumulative values, cycles simulated, state occupancy, and termination reason.

    Args:
        ... (既存の引数) ...

    Returns:
        The completed IndividualState object if simulation starts successfully and finishes.
        Returns None on critical error during initial setup.
    """
    _lazy_import_utils()
    if safe_eval is None:
        print("Error: safe_eval is not loaded. Cannot simulate individual path.")
        return None

    state_map: Dict[str, MarkovState] = {s.name: s for s in markov_chain.markov_states}

    if start_state_obj is None:
        print(f"Error: Invalid start state object provided. Cannot start simulation path.")
        return None

    individual_state = IndividualState(start_state_obj, initial_global_variables.copy(), project_dim_info)

    base_evaluation_context: Dict[str, Any] = {}
    base_evaluation_context.update(initial_global_parameters)
    base_evaluation_context.update(initial_global_variables) # Add initial variables to base context

    def table_lookup_func(table_name: str, lookup_key: float, key_col: Union[int, str] = 0, value_col: Union[int, str] = 1) -> Optional[float]:
         table = project_tables.get(table_name)
         if not table: return None
         resolved_key_col = key_col
         if isinstance(key_col, str):
              try: resolved_key_col = table.headers.index(key_col)
              except ValueError: return None
         resolved_value_col = value_col
         if isinstance(value_col, str):
             try: resolved_value_col = table.headers.index(value_col)
             except ValueError: return None
         return table.get_value(lookup_key, key_col_index=resolved_key_col, value_col_index=resolved_value_col)

    base_evaluation_context['lookup'] = table_lookup_func
    base_evaluation_context['math'] = math
    base_evaluation_context['random'] = random

    # --- Simulation Loop (Cycles) ---
    while individual_state.current_cycle < max_cycles and not individual_state.is_terminated:

        current_state_obj = individual_state.current_state_obj
        current_state_name = individual_state.current_state_name

        if current_state_obj is None:
             print(f"  Error: Current state object is None for individual path in cycle {individual_state.current_cycle}. Terminating path.")
             individual_state.is_terminated = True
             # Set termination reason for error
             individual_state.termination_reason = "Simulation Error (Current State None)"
             individual_state.cycles_simulated = individual_state.current_cycle + 1
             break

        cycle_evaluation_context = individual_state.get_evaluation_context(base_evaluation_context)
        individual_state.add_cycle_values(project_dim_info, cycle_evaluation_context)
        individual_state.apply_variable_updates(base_evaluation_context)

        possible_transitions: List[StateTransition] = []
        if current_state_obj.state_transitions:
             possible_transitions = current_state_obj.state_transitions
        elif current_state_obj.chance_nodes:
             if current_state_obj.chance_nodes:
                 first_chance_node = current_state_obj.chance_nodes[0]
                 possible_transitions = first_chance_node.state_transitions

        transitions_for_this_cycle: List[Tuple[StateTransition, float]] = []
        unresolved_complement_transitions: List[StateTransition] = []

        current_cycle_context_for_transitions = individual_state.get_evaluation_context(base_evaluation_context)

        for st in possible_transitions:
             prob_expr = st.data.get("prob", "0")
             if prob_expr == "C":
                  unresolved_complement_transitions.append(st)
             elif prob_expr and prob_expr.strip() != "0":
                 try:
                      evaluated_prob = safe_eval(prob_expr, current_cycle_context_for_transitions)
                      if isinstance(evaluated_prob, (int, float)):
                           evaluated_prob = max(0.0, min(1.0, evaluated_prob))
                           transitions_for_this_cycle.append((st, evaluated_prob))
                      else:
                           print(f"  [!] Warning: Transition probability expression '{prob_expr}' for '{st.name}' in state '{current_state_name}', cycle {individual_state.current_cycle} evaluated to non-numeric: {evaluated_prob}. Using 0.0.")
                           transitions_for_this_cycle.append((st, 0.0))
                 except Exception as e:
                      print(f"  [!] Error evaluating transition probability '{prob_expr}' for '{st.name}' in state '{current_state_name}', cycle {individual_state.current_cycle}: {e}. Using 0.0.")
                      transitions_for_this_cycle.append((st, 0.0))
             else:
                  transitions_for_this_cycle.append((st, 0.0))

        sum_resolved_probs = sum(prob for _, prob in transitions_for_this_cycle)
        if unresolved_complement_transitions:
            remaining_prob = 1.0 - sum_resolved_probs
            if remaining_prob < -1e-9:
                 print(f"[!] Warning: Total resolved probabilities ({sum_resolved_probs:.4f}) exceed 1.0 in state '{current_state_name}' / ChanceNode in cycle {individual_state.current_cycle}. Remaining for complement set to 0.")
                 remaining_prob = 0.0
            elif remaining_prob < 0:
                 remaining_prob = 0.0

            if remaining_prob > 1e-9:
                 per_node = remaining_prob / len(unresolved_complement_transitions)
                 for st in unresolved_complement_transitions:
                     transitions_for_this_cycle.append((st, per_node))
            else:
                 for st in unresolved_complement_transitions:
                     transitions_for_this_cycle.append((st, 0.0))

        valid_transitions_and_probs = []
        for st, prob in transitions_for_this_cycle:
            if prob > 1e-9:
                 target_state_obj = state_map.get(st.transition)
                 if target_state_obj:
                      valid_transitions_and_probs.append((target_state_obj, prob))
                 else:
                      print(f"  [!] Warning: Transition '{st.name}' from '{current_state_name}' at cycle {individual_state.current_cycle} targets unknown state '{st.transition}'. Ignoring this transition.")

        # If no valid transitions, the individual terminates in the *current* state
        if not valid_transitions_and_probs:
            individual_state.is_terminated = True
            # Set termination reason
            individual_state.termination_reason = f"No Valid Transitions from State '{current_state_name}'"
            individual_state.cycles_simulated = individual_state.current_cycle + 1
            # State occupancy for the current cycle is already counted in __init__ or loop
            continue


        next_state_options = [obj for obj, prob in valid_transitions_and_probs]
        probs = [prob for obj, prob in valid_transitions_and_probs]
        sum_of_valid_probs = sum(probs)

        if abs(sum_of_valid_probs - 1.0) > 1e-6 and sum_of_valid_probs > 1e-9:
             if sum_of_valid_probs > 1e-9:
                 normalized_probs = [p / sum_of_valid_probs for p in probs]
             else: # Sum is positive but tiny, means something is wrong, terminate.
                 print(f"  [!] Error: Sum of valid transition probabilities from '{current_state_name}' at cycle {individual_state.current_cycle} is near zero ({sum_of_valid_probs:.4f}). Cannot normalize or choose state. Terminating path.")
                 individual_state.is_terminated = True
                 individual_state.termination_reason = "Simulation Error (Probabilities Near Zero)"
                 individual_state.cycles_simulated = individual_state.current_cycle + 1
                 continue
        elif sum_of_valid_probs <= 1e-9: # Sum is non-positive, terminate.
             print(f"  [!] Error: Sum of valid transition probabilities from '{current_state_name}' at cycle {individual_state.current_cycle} is non-positive ({sum_of_valid_probs:.4f}) after recalculation. Cannot choose next state. Terminating path.")
             individual_state.is_terminated = True
             individual_state.termination_reason = "Simulation Error (Probabilities Non-Positive)"
             individual_state.cycles_simulated = individual_state.current_cycle + 1
             continue
        else:
             normalized_probs = probs


        try:
            chosen_next_state_obj = random.choices(next_state_options, weights=normalized_probs, k=1)[0]
            chosen_next_state_name = chosen_next_state_obj.name

            # Move to the next state
            individual_state.current_state_obj = chosen_next_state_obj
            individual_state.current_state_name = chosen_next_state_name

            # --- Step 6: Check for termination AFTER transitioning ---
            # Check for immediate termination states like "Death" name AFTER transitioning
            if individual_state.current_state_name == "Death":
                 individual_state.is_terminated = True
                 # Set termination reason
                 individual_state.termination_reason = "Death State Reached"
                 individual_state.cycles_simulated = individual_state.current_cycle + 1
                 # Note: If Death is the ONLY hard terminal state, this is sufficient.
                 # If there are other named terminal states, check for them here too.
                 # Or, check a list of terminal state names if provided in Project/MC.

            # Check Markov Chain Termination Condition (for the *end* of the current cycle)
            # This condition uses 't' (current_cycle), which is the cycle just completed.
            term_context = individual_state.get_evaluation_context(base_evaluation_context)
            termination_expr = markov_chain.termination_condition

            # If not already terminated by state check...
            if not individual_state.is_terminated and termination_expr and safe_eval:
                 try:
                     is_terminated_by_condition = safe_eval(termination_expr, term_context)
                     if is_terminated_by_condition is True:
                          individual_state.is_terminated = True
                          # Set termination reason
                          individual_state.termination_reason = f"Termination Condition Met ('{termination_expr}')"
                          individual_state.cycles_simulated = individual_state.current_cycle + 1

                 except Exception as e:
                      print(f"  [!] Error evaluating termination condition '{termination_expr}' in cycle {individual_state.current_cycle}: {e}. Simulation will continue up to max_cycles.")
                      # individual_state.is_terminated remains False, no reason set for this condition type error


        except ValueError as e:
             print(f"  [!] Error choosing next state from '{current_state_name}' at cycle {individual_state.current_cycle} with normalized probabilities {normalized_probs}: {e}. Terminating path.")
             individual_state.is_terminated = True
             individual_state.termination_reason = "Simulation Error (State Selection)"
             individual_state.cycles_simulated = individual_state.current_cycle + 1
             break
        except Exception as e:
             print(f"  [!] Error during next state selection from '{current_state_name}' in cycle {individual_state.current_cycle}: {e}. Terminating path.")
             individual_state.is_terminated = True
             individual_state.termination_reason = "Simulation Error (Unexpected)"
             individual_state.cycles_simulated = individual_state.current_cycle + 1
             break

        # --- Step 7: Prepare for Next Cycle & Update State Occupancy ---
        individual_state.current_cycle += 1

        # Update State Occupancy for the state the individual just transitioned *into*.
        # This state will be the current state at the start of the *next* cycle.
        # Add 1 cycle to the new current state's occupancy.
        # The initial state's occupancy for cycle 0 is handled in __init__ (state_occupancy = {start_state_obj.name: 1}).
        # This update is for cycles 1, 2, ...
        # NOTE: This logic counts the cycle *after* entering the state.
        # If an individual is in StateA for cycle 0, transitions to StateB at the end of cycle 0,
        # they spend cycle 1 in StateB. Occupancy should be StateA:1, StateB:1 after cycle 1 processing.
        # Let's clarify the counting:
        # In __init__: state_occupancy[start_state_obj.name] = 1 (for cycle 0)
        # Loop for cycle t:
        #   Spend cycle t in current_state_obj. Calculate costs/rewards for cycle t. Apply updates for cycle t.
        #   Transition to next_state_obj.
        #   Next state will be current state for cycle t+1.
        #   Increment current_cycle to t+1.
        #   Update occupancy for the state *just entered* (individual_state.current_state_name) for cycle t+1.
        #   This logic seems correct. StateA: 1 (cycle 0), StateB: 1 (cycle 1), ...
        # Total cycles simulated = sum of state occupancies.

        new_current_state_name = individual_state.current_state_name # The state they just entered
        individual_state.state_occupancy[new_current_state_name] = individual_state.state_occupancy.get(new_current_state_name, 0) + 1


    # --- Simulation path finished ---
    # The loop exited because max_cycles was reached OR individual_state.is_terminated became True.

    # Finalize cycles_simulated and termination_reason if loop completed without specific termination event
    # This handles the case where the loop simply reaches max_cycles without hitting Death or condition
    if not individual_state.is_terminated:
         # Loop exited because current_cycle reached max_cycles.
         individual_state.cycles_simulated = max_cycles # Number of cycles completed (0 to max_cycles-1, total max_cycles cycles)
         individual_state.termination_reason = "Max Cycles Reached" # Reason is reaching max cycles

    # One final check/adjustment for state occupancy sum vs cycles_simulated
    # The sum of state occupancies should equal cycles_simulated.
    # sum_occupancy = sum(individual_state.state_occupancy.values())
    # if sum_occupancy != individual_state.cycles_simulated:
    #      print(f"  Warning: Occupancy sum ({sum_occupancy}) != Cycles Simulated ({individual_state.cycles_simulated}) for an individual path.")
    #      # Decide how to handle - maybe adjust one to match the other, or log and ignore.
    #      # Based on the logic (init + loop increment), they should be equal.

    # Return the completed IndividualState object
    return individual_state

def run_cohort_simulation(
    markov_chain: MarkovChain,
    project_model: Project, # Pass the full project model
    num_individuals: Optional[int] = None, # Override project setting
    max_cycles: Optional[int] = None,      # Override MC setting
    start_state_name: Optional[str] = None # Override default start state
) -> Optional[ChainResult]:
    """
    Runs a cohort simulation for a single Markov Chain within a Project.
    Evaluates initial globals, resolves transitions, simulates multiple individuals,
    and calculates statistics and simulation characteristics, returning a ChainResult object.

    Args:
        ... (既存の引数) ...

    Returns:
        A ChainResult object containing all aggregated statistics and characteristics
        for the chain, or None if simulation cannot start or fails critically.
    """
    _lazy_import_utils()
    if safe_eval is None:
         print("Error: safe_eval is not loaded. Cannot run cohort simulation.")
         return None

    print(f"\n--- Running Cohort Simulation for '{markov_chain.name}' ---")

    # --- 1. Evaluate Initial Global Parameters and Variables ---
    # ... (既存のコードそのまま) ...
    print("Evaluating initial global parameters and variables...")
    initial_global_parameters: Dict[str, float] = {}
    initial_global_variables: Dict[str, float] = {}

    global_eval_context: Dict[str, Any] = {'math': math, 'random': random}

    def global_table_lookup(table_name: str, lookup_key: float, key_col: Union[int, str] = 0, value_col: Union[int, str] = 1) -> Optional[float]:
         table = project_model.tables.get(table_name)
         if not table: return None
         resolved_key_col = key_col
         if isinstance(key_col, str):
              try: resolved_key_col = table.headers.index(key_col)
              except ValueError: return None
         resolved_value_col = value_col
         if isinstance(value_col, str):
             try: resolved_value_col = table.headers.index(value_col)
             except ValueError: return None
         return table.get_value(lookup_key, key_col_index=resolved_key_col, value_col_index=resolved_value_col)

    global_eval_context['lookup'] = global_table_lookup

    for param_name, param_obj in project_model.global_parameters.items():
        try:
            value = safe_eval(param_obj.expression, global_eval_context)
            if isinstance(value, (int, float)):
                initial_global_parameters[param_name] = value
                global_eval_context[param_name] = value
                param_obj.value = value
            else:
                 print(f"  Warning: Global parameter '{param_name}' expression '{param_obj.expression}' evaluated to non-numeric: {value}. Using 0.0")
                 initial_global_parameters[param_name] = 0.0
                 global_eval_context[param_name] = 0.0
                 param_obj.value = 0.0
        except Exception as e:
            print(f"  [!] Error evaluating global parameter '{param_name}' expression '{param_obj.expression}': {e}. Using 0.0")
            initial_global_parameters[param_name] = 0.0
            global_eval_context[param_name] = 0.0
            param_obj.value = 0.0

    evaluated_vars_count = 0
    max_eval_attempts = len(project_model.global_variables) + 5
    variables_to_evaluate = list(project_model.global_variables.items())
    successfully_evaluated_globals: Dict[str, float] = {}

    for attempt in range(max_eval_attempts):
         updated_in_attempt = False
         unresolved_in_attempt = []
         for var_name, var_obj in variables_to_evaluate:
             if var_name not in successfully_evaluated_globals:
                  try:
                       value = safe_eval(var_obj.expression, global_eval_context)
                       if isinstance(value, (int, float)):
                           successfully_evaluated_globals[var_name] = value
                           global_eval_context[var_name] = value
                           var_obj.value = value
                           updated_in_attempt = True
                           evaluated_vars_count += 1
                       else:
                            unresolved_in_attempt.append((var_name, var_obj))
                  except Exception as e:
                       unresolved_in_attempt.append((var_name, var_obj))
         variables_to_evaluate = unresolved_in_attempt
         if not updated_in_attempt and variables_to_evaluate:
              break

    initial_global_variables.update(successfully_evaluated_globals)
    for var_name, var_obj in project_model.global_variables.items():
         if var_name not in initial_global_variables:
              print(f"  Warning: Global variable '{var_name}' expression '{var_obj.expression}' failed to evaluate after multiple attempts. Using 0.0 as initial value.")
              initial_global_variables[var_name] = 0.0
              var_obj.value = 0.0

    print(f"Initial global evaluation complete. {len(initial_global_parameters)} parameters, {len(initial_global_variables)} variables evaluated.")

    # --- 2. Resolve Transitions within the Markov Chain ---
    print(f"\nResolving initial transitions within '{markov_chain.name}' using initial global context...")
    markov_chain.resolve_transitions() # This uses mc.get_context() internally
    print("Transitions resolved.")

    # --- 3. Determine Simulation Settings ---
    # ... (既存のコードそのまま) ...
    containing_tree: Optional[DecisionTree] = None
    for tree in project_model.decision_trees:
        if markov_chain.index in tree.nodes and tree.nodes[markov_chain.index] is markov_chain:
            containing_tree = tree
            break

    if containing_tree is None:
        print(f"Warning: Cannot find the DecisionTree containing Markov Chain '{markov_chain.name}' (index {markov_chain.index}). Will use Project-level or default simulation settings.")
        mc_max_cycles_setting = None
        mc_cycles_per_year_setting = None
        mc_state_decimals_setting = None
        mc_half_cycle_correction_setting = None
        mc_discount_rewards_setting = None
        mc_discount_start_cycle_setting = None
        mc_show_trace_setting = None
        mc_compile_traces_setting = None
    else:
        mc_max_cycles_setting = containing_tree.max_cycles
        mc_cycles_per_year_setting = containing_tree.cycles_per_year
        mc_state_decimals_setting = containing_tree.state_decimals
        mc_half_cycle_correction_setting = containing_tree.half_cycle_correction
        mc_discount_rewards_setting = containing_tree.discount_rewards
        mc_discount_start_cycle_setting = containing_tree.discount_start_cycle
        mc_show_trace_setting = containing_tree.show_trace
        mc_compile_traces_setting = containing_tree.compile_traces


    sim_num_individuals = num_individuals if num_individuals is not None else (project_model.cohort_size if project_model.cohort_size is not None else 1000)
    sim_max_cycles = max_cycles if max_cycles is not None else (mc_max_cycles_setting if mc_max_cycles_setting is not None else (project_model.max_cycles if project_model.max_cycles is not None else 10))
    sim_cycles_per_year = mc_cycles_per_year_setting if mc_cycles_per_year_setting is not None else (project_model.cycles_per_year if project_model.cycles_per_year is not None else 1.0)
    sim_state_decimals = mc_state_decimals_setting if mc_state_decimals_setting is not None else (project_model.state_decimals if project_model.state_decimals is not None else 0)
    sim_half_cycle_correction = mc_half_cycle_correction_setting if mc_half_cycle_correction_setting is not None else (project_model.half_cycle_correction if project_model.half_cycle_correction is not None else False)
    sim_discount_rewards = mc_discount_rewards_setting if mc_discount_rewards_setting is not None else (project_model.discount_rewards if project_model.discount_rewards is not None else True)
    sim_discount_start_cycle = mc_discount_start_cycle_setting if mc_discount_start_cycle_setting is not None else (project_model.discount_start_cycle if project_model.discount_start_cycle is not None else 0)
    sim_show_trace = mc_show_trace_setting if mc_show_trace_setting is not None else (project_model.show_trace if project_model.show_trace is not None else False)
    sim_compile_traces = mc_compile_traces_setting if mc_compile_traces_setting is not None else (project_model.compile_traces if project_model.compile_traces is not None else False)


    sim_start_state_name = start_state_name
    if sim_start_state_name is None:
         if markov_chain.markov_states:
             sim_start_state_name = markov_chain.markov_states[0].name
         else:
              print(f"Error: Markov Chain '{markov_chain.name}' has no states and no start_state_name provided. Cannot simulate.")
              return None

    mc_state_map: Dict[str, MarkovState] = {s.name: s for s in markov_chain.markov_states}
    start_state_obj = mc_state_map.get(sim_start_state_name)

    if start_state_obj is None:
        print(f"Error: Start state '{sim_start_state_name}' not found in Markov Chain '{markov_chain.name}'. Cannot simulate.")
        available_states = list(mc_state_map.keys())
        print(f"Available states in '{markov_chain.name}': {available_states}")
        return None


    print(f"\nSimulation Settings:")
    print(f"  Start State: '{sim_start_state_name}'")
    print(f"  Number of Individuals: {sim_num_individuals}")
    print(f"  Maximum Cycles: {sim_max_cycles}")
    print(f"  Cycles Per Year: {sim_cycles_per_year}") # Display Cycles Per Year setting
    print(f"  State Decimals: {sim_state_decimals}")
    print(f"  Half Cycle Correction: {sim_half_cycle_correction}")
    print(f"  Discount Rewards: {sim_discount_rewards}")
    print(f"  Discount Start Cycle: {sim_discount_start_cycle}")
    if project_model.crn and project_model.crn_seed is not None:
        print(f"  CRN Enabled with Seed: {project_model.crn_seed}")


    # --- 4. Run Simulations for Multiple Individuals ---
    completed_individual_states: List[IndividualState] = []
    successful_sim_count = 0

    if project_model.crn and project_model.crn_seed is not None:
         print(f"Setting random seed to {project_model.crn_seed}")
         random.seed(project_model.crn_seed)

    print(f"\nRunning {sim_num_individuals} individual simulations...")

    for i in range(sim_num_individuals):
        individual_state_result = simulate_individual_path(
            markov_chain=markov_chain,
            start_state_obj=start_state_obj,
            max_cycles=sim_max_cycles,
            project_dim_info=project_model.dim_info,
            project_tables=project_model.tables,
            initial_global_parameters=initial_global_parameters,
            initial_global_variables=initial_global_variables.copy()
        )

        if individual_state_result is not None:
            completed_individual_states.append(individual_state_result)
            successful_sim_count += 1

    print(f"\nIndividual simulations complete.")
    print(f"Successfully simulated paths: {successful_sim_count} / {sim_num_individuals}")
    failed_sim_count = sim_num_individuals - successful_sim_count
    if failed_sim_count > 0:
        print(f"Failed simulation paths: {failed_sim_count}")


    # --- 5. Calculate Aggregated Statistics and Characteristics and create ChainResult ---
    if not completed_individual_states:
        print("No successful simulation results to aggregate.")
        return None

    print("\nCalculating aggregated statistics and characteristics...")

    # Cumulative Statistics (same as before)
    if project_model.dim_info:
        all_dim_names = [dim.name for dim in project_model.dim_info.dimensions]
    elif completed_individual_states:
        first_result_keys = completed_individual_states[0].cumulative_values.keys()
        all_dim_names = list(first_result_keys) if first_result_keys else []
        if not project_model.dim_info:
            print("Warning: DimInfo missing, inferring dimension names from first individual result for statistics.")
    else:
         print("Cannot calculate statistics: No DimInfo and no successful simulations.")
         return None

    dimension_stats_dict: Dict[str, DimensionStats] = {}

    for dim_name in all_dim_names:
        dim_values = [ind_state.cumulative_values.get(dim_name, 0.0) for ind_state in completed_individual_states]

        if not dim_values or successful_sim_count == 0:
             avg = 0.0; std_dev = 0.0; min_val = 0.0; max_val = 0.0
             if dim_name in [d.name for d in project_model.dim_info.dimensions] if project_model.dim_info else True:
                  print(f"Warning: No cumulative data found for dimension '{dim_name}' across {successful_sim_count} successful simulations. Statistics set to 0.")
        else:
            avg = sum(dim_values) / successful_sim_count
            min_val = min(dim_values)
            max_val = max(dim_values)
            if successful_sim_count > 1:
                 try:
                      std_dev = statistics.stdev(dim_values)
                 except statistics.StatisticsError:
                      std_dev = 0.0
            else:
                 std_dev = 0.0

        dimension_stats_dict[dim_name] = DimensionStats(
            dimension_name=dim_name, average=avg, std_dev=std_dev, min_val=min_val, max_val=max_val
        )

    print("Cumulative statistics calculated.")

    # --- Calculate Simulation Characteristics (Average Cycles, State Occupancy, Termination Reasons) ---

    # Calculate Average Cycles Simulated (same as before)
    total_cycles_simulated = sum(ind_state.cycles_simulated for ind_state in completed_individual_states)
    average_cycles_simulated = total_cycles_simulated / successful_sim_count if successful_sim_count > 0 else 0.0
    # print(f"Average cycles simulated: {average_cycles_simulated:.2f}") # Print moved to display function

    # Calculate Average State Occupancy (same as before)
    total_state_occupancy: Dict[str, int] = {}
    for ind_state in completed_individual_states:
        for state_name, cycles_in_state in ind_state.state_occupancy.items():
             total_state_occupancy[state_name] = total_state_occupancy.get(state_name, 0) + cycles_in_state

    average_state_occupancy: Dict[str, float] = {}
    if successful_sim_count > 0:
        all_mc_state_names = {state.name for state in markov_chain.markov_states}
        for state_name in all_mc_state_names:
             total_cycles = total_state_occupancy.get(state_name, 0)
             average_state_occupancy[state_name] = total_cycles / successful_sim_count
        for state_name in total_state_occupancy.keys() - all_mc_state_names:
             print(f"Warning: State '{state_name}' found in occupancy results but not in MC definition.")
             average_state_occupancy[state_name] = total_state_occupancy[state_name] / successful_sim_count
    # print("Average state occupancy calculated.") # Print moved to display function


    # --- Calculate Termination Reason Counts ---
    termination_reason_counts: Dict[str, int] = {}
    for ind_state in completed_individual_states:
        reason = ind_state.termination_reason if ind_state.termination_reason is not None else "Unknown Reason" # Handle None reason
        termination_reason_counts[reason] = termination_reason_counts.get(reason, 0) + 1

    print("Termination reason counts calculated.")
    # Optional: Print counts for debugging
    # print("Termination Reason Counts:", termination_reason_counts)

    # Create and return the ChainResult object for this Markov Chain
    chain_result = ChainResult(
        chain_name=markov_chain.name,
        simulated_individuals_count=sim_num_individuals,
        successful_simulated_count=successful_sim_count,
        max_cycles=sim_max_cycles,
        start_state_name=sim_start_state_name,
        dimension_stats=dimension_stats_dict,
        average_cycles_simulated=average_cycles_simulated,
        average_state_occupancy=average_state_occupancy,
        cycles_per_year=sim_cycles_per_year, # Pass cycles per year
        termination_reason_counts=termination_reason_counts # Pass termination counts
    )

    print(f"ChainResult object created for '{markov_chain.name}' with characteristics.")
    return chain_result

def display_aggregated_results(simulation_result: Optional[SimulationResult]):
    # ... (既存の定義そのまま) ...
    """
    Displays aggregated simulation statistics and characteristics stored in a SimulationResult object.
    Includes Average, StdDev, Min, Max per dimension, plus average cycles, cycles per year,
    state occupancy, and termination reason breakdown.
    """
    if simulation_result is None or not simulation_result.chain_results:
        print("\n--- No Simulation Results to Display ---")
        return

    print(f"\n--- Aggregated Simulation Results for Project '{simulation_result.project_name}' ---")

    # Display cumulative statistics table first (same as before)
    print("\n--- Cumulative Dimension Statistics ---")

    all_dim_names_in_results = set()
    for chain_result in simulation_result.chain_results.values():
        all_dim_names_in_results.update(chain_result.dimension_stats.keys())

    project_dim_info = simulation_result.project_dim_info
    if project_dim_info:
        dim_names_ordered = [dim.name for dim in project_dim_info.dimensions if dim.name in all_dim_names_in_results]
        dim_names_ordered.extend(sorted(list(all_dim_names_in_results - set(dim_names_ordered))))
    else:
        dim_names_ordered = sorted(list(all_dim_names_in_results))
        if dim_names_ordered:
             print("Warning: DimInfo missing. Displaying dimensions in alphabetical order.")

    if not dim_names_ordered:
         print("No dimensions found in the simulation results to display statistics for.")
    else:
        stats_to_display = ["Average", "StdDev", "Min", "Max"]
        headers = ["Markov Chain"]
        for dim_name in dim_names_ordered:
            for stat_name in stats_to_display:
                 headers.append(f"{dim_name} [{stat_name}]")

        table_data = []
        for mc_name, chain_result in simulation_result.chain_results.items():
            row = [mc_name]
            for dim_name in dim_names_ordered:
                dim_stats = chain_result.get_stats_for_dimension(dim_name)

                format_string = ".4f"
                symbol = ""
                if project_dim_info:
                     dim_obj = next((d for d in project_dim_info.dimensions if d.name == dim_name), None)
                     if dim_obj:
                         format_string = f".{dim_obj.decimals}f"
                         symbol = dim_obj.symbol

                for stat_name in stats_to_display:
                    attribute_name = stat_name.lower().replace(' ', '_')
                    value = getattr(dim_stats, attribute_name, 0.0) if dim_stats else 0.0
                    formatted_value = f"{value:{format_string}}{symbol}"
                    row.append(formatted_value)
            table_data.append(row)

        if table_data:
            print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("No cumulative statistics data rows to display.")

    # --- Display Simulation Characteristics ---
    print("\n--- Simulation Characteristics ---")
    # Add Cycles Per Year to the characteristics header
    char_headers = ["Markov Chain", "Simulated Individuals", "Successful Simulations", "Max Cycles", "Cycles Per Year", "Average Cycles Simulated"]
    char_data = []

    # Collect all unique state names across all chains' state occupancy results for consistent columns
    all_state_names_in_occupancy = set()
    for chain_result in simulation_result.chain_results.values():
        all_state_names_in_occupancy.update(chain_result.average_state_occupancy.keys())

    state_names_ordered = sorted(list(all_state_names_in_occupancy))

    # Add headers for average state occupancy
    for state_name in state_names_ordered:
        char_headers.append(f"Avg Occupancy [{state_name}]")


    for mc_name, chain_result in simulation_result.chain_results.items():
        row = [
            mc_name,
            chain_result.simulated_individuals_count,
            chain_result.successful_simulated_count,
            chain_result.max_cycles,
            f"{chain_result.cycles_per_year:.2f}", # Display Cycles Per Year
            f"{chain_result.average_cycles_simulated:.2f}" # Display Average Cycles Simulated
        ]
        # Add average state occupancy values in the determined order
        for state_name in state_names_ordered:
            avg_occupancy = chain_result.average_state_occupancy.get(state_name, 0.0)
            row.append(f"{avg_occupancy:.2f}")

        char_data.append(row)

    if char_data:
         print(tabulate.tabulate(char_data, headers=char_headers, tablefmt="grid"))
    else:
         print("No simulation characteristics data to display.")

    # --- Display Termination Reason Breakdown ---
    print("\n--- Termination Reason Breakdown ---")
    # Iterate through each chain's results
    for mc_name, chain_result in simulation_result.chain_results.items():
        print(f"  '{mc_name}':")
        total_successful = chain_result.successful_simulated_count
        reason_counts = chain_result.termination_reason_counts

        if total_successful > 0 and reason_counts:
            # Sort reasons alphabetically for consistent output
            sorted_reasons = sorted(reason_counts.keys())
            for reason in sorted_reasons:
                count = reason_counts[reason]
                percentage = (count / total_successful) * 100 if total_successful > 0 else 0.0
                print(f"    - {reason}: {count} individuals ({percentage:.1f}%)")
        elif total_successful > 0:
            print("    - No specific termination reasons recorded for successful simulations.")
        else:
             print("    - No successful simulations to report termination reasons.")


    # Print the final footer line
    footer_text = f"--- Aggregated Simulation Results for Project '{simulation_result.project_name}' ---"
    print("-" * len(footer_text))