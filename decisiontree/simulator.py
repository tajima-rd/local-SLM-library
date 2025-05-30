# amua_simulator.py
# -*- coding: utf-8 -*-
"""
Handles the simulation of Markov Chains based on the parsed model.
Simulates individual paths, evaluates costs/rewards/variable updates per cycle,
and accumulates results per dimension. Handles table lookups in expressions.
"""

import random
import re
from typing import List, Dict, Optional, Any, Tuple, Union # Added Union, Tuple
import math # math functions might be used in expressions (e.g., math.log, math.exp)

# Import necessary model classes
# Assuming model.py is in the same directory or sys.path is set up correctly
try:
    from model import MarkovChain, MarkovState, StateTransition, ChanceNode, Project, DimensionInfo, Table, Parameter, Variable
except ImportError as e:
    print(f"Error importing model classes in simulator.py: {e}")
    print("Please ensure model.py is accessible via sys.path.")
    # Exit or handle error appropriately if models are critical
    # For this example, we'll assume the calling script handles this if it fails.
    # Simulation functions will return None or empty results on critical errors.
    # sys.exit(1) # Uncomment to exit hard on import failure
    pass # Allow script to proceed but log error


# Lazy import utility functions (safe_eval and complete_direct_state_transition_complement)
# These should be defined in utils.py
# Using a simple check; _lazy_import_utils is better for avoiding circular deps
# but direct import here is simpler if utils doesn't depend on model/simulator.
# Let's assume direct import is okay or utils has no heavy deps.
# from utils import safe_eval, complete_direct_state_transition_complement # Original thought
# Revert to lazy import pattern as it's more robust for potentially complex projects
safe_eval = None
complete_direct_state_transition_complement = None

def _lazy_import_utils():
    """Lazy import utility functions."""
    global safe_eval, complete_direct_state_transition_complement
    if safe_eval is None: # Check if already imported
        try:
            from utils import safe_eval, complete_direct_state_transition_complement
        except ImportError:
             print("Warning: Could not import amua_utils in simulator. Safe eval and complement functions may not work.")
             # Define dummy functions that will likely fail if used, prompting user to fix imports
             safe_eval = lambda expr, context={}: None # Dummy that always returns None
             complete_direct_state_transition_complement = lambda state: None


# --- Helper class to track the state of a single individual ---
# Defined here in simulator.py
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
        # If dim_info is None, cumulative_values remains empty, which might require
        # handling in the analysis script if it expects specific dimension names.


    def get_evaluation_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a context dictionary for evaluating expressions (costs, rewards, var updates, termination).
        Includes base globals, current cycle 't', and current individual variables.
        """
        eval_context = base_context.copy() # Start with base context (globals + tables func)
        eval_context['t'] = self.current_cycle # Add current cycle number
        # Add current individual variables to the context. This allows expressions like 'timeSick > 5'.
        eval_context.update(self.variables)
        # You might also add state-specific context if needed, e.g., current state name
        # eval_context['currentStateName'] = self.current_state_name
        # eval_context['currentStateObj'] = self.current_state_obj # Needs safe_eval to handle object access
        return eval_context


    def add_cycle_values(self, dim_info: Optional[DimensionInfo], evaluation_context: Dict[str, Any]) -> None:
         """
         Evaluates costs and rewards expressions for the current state
         and adds the results to the cumulative totals.
         """
         _lazy_import_utils() # Ensure safe_eval is loaded
         if safe_eval is None or dim_info is None:
              # Error message printed during import or project parsing
              return # Cannot evaluate without safe_eval or DimInfo

         current_state = self.current_state_obj
         cost_exprs = current_state.cost_expressions # Dict[DimName, ExprString]
         reward_exprs = current_state.reward_expressions # Dict[DimName, ExprString]

         # Evaluate cost expressions
         # Iterate over dimension names from cumulative_values keys (ensures we cover all dimensions)
         for dim_name in self.cumulative_values.keys():
             expr = cost_exprs.get(dim_name, "0") # Get expression for this dimension, default to "0" if missing

             if expr: # Only evaluate if expression is not empty or default "0"
                 try:
                     value = safe_eval(expr, evaluation_context) # Evaluate the string expression
                     if isinstance(value, (int, float)):
                         # Add evaluated value to cumulative total for this dimension
                         self.cumulative_values[dim_name] += value
                         # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Added cost '{expr}' ({value:.4f}) to {dim_name}. Total {dim_name}: {self.cumulative_values[dim_name]:.4f}") # Debugging
                     # else: Warning about non-numeric evaluation result...
                 except Exception as e:
                     # Log error if expression evaluation fails
                     print(f"  Error evaluating cost expression '{expr}' for dimension '{dim_name}' in state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Using 0.0 for this cycle.")
                     # Cumulative value is already initialized to 0.0, so no change needed


         # Evaluate reward expressions
         # Iterate over dimension names from cumulative_values keys
         for dim_name in self.cumulative_values.keys():
              expr = reward_exprs.get(dim_name, "0") # Get expression for this dimension, default to "0"

              if expr: # Only evaluate if expression is not empty or default "0"
                 try:
                     value = safe_eval(expr, evaluation_context) # Evaluate the string expression
                     if isinstance(value, (int, float)):
                         # Add evaluated value to cumulative total for this dimension
                         self.cumulative_values[dim_name] += value
                         # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Added reward '{expr}' ({value:.4f}) to {dim_name}. Total {dim_name}: {self.cumulative_values[dim_name]:.4f}") # Debugging
                     # else: Warning about non-numeric evaluation result...
                 except Exception as e:
                      # Log error if expression evaluation fails
                      print(f"  Error evaluating reward expression '{expr}' for dimension '{dim_name}' in state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Using 0.0 for this cycle.")


    def apply_variable_updates(self, evaluation_context: Dict[str, Any]) -> None:
        """
        Evaluates and applies variable update expressions defined for the current state.
        Handles basic 'varName++' syntax and relies on safe_eval for general expressions.
        Updates values in self.variables.
        """
        _lazy_import_utils() # Ensure safe_eval is loaded
        if safe_eval is None:
             # Error message printed during import or project parsing
             return

        current_state = self.current_state_obj
        update_expressions = current_state.variable_updates # List of ExprStrings

        if not update_expressions:
             return # No updates to apply for this state

        # Evaluate expressions sequentially and apply updates
        # Need to be careful if later expressions in the list depend on variables
        # updated by earlier expressions in the same list.
        # Amua's evaluation order for multiple varUpdates might be important.
        # Assuming sequential evaluation in the order they appear in the list.

        for expr in update_expressions:
            if not expr.strip(): continue # Skip empty strings

            try:
                # --- Handling variable update expressions ---
                # These are assignments. Standard `eval` is unsafe.
                # safe_eval needs to support assignment or we need specific pattern matching.
                # Let's enhance this to support simple assignments `var = expr` using safe_eval.
                # This requires safe_eval to return the result of the RHS and we apply it.

                match_assign = re.match(r'^\s*([\w_]+)\s*=\s*(.+)$', expr)
                match_increment = re.match(r'^\s*([\w_]+)\s*\+\+\s*$', expr) # Pattern for 'varName++'

                if match_increment:
                     var_name = match_increment.group(1)
                     if var_name in self.variables:
                          self.variables[var_name] += 1.0 # Increment by 1.0
                          # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Incremented variable {var_name} to {self.variables[var_name]}.") # Debugging
                     else:
                          print(f"  Warning: Attempted to increment unknown variable '{var_name}' in update '{expr}' for state '{self.current_state_name}', cycle {self.current_cycle}. Variable not initialized.")
                          # Decide behavior for unknown variable: Initialize to 1.0? Ignore?
                          # self.variables[var_name] = 1.0 # Option: Initialize
                     continue # Processed this expression


                elif match_assign:
                     var_name = match_assign.group(1)
                     value_expr = match_assign.group(2) # The expression on the right-hand side

                     # Evaluate the RHS expression using the current context (includes updated vars from THIS state)
                     # The context used here should reflect variables already updated by PREVIOUS expressions in the same update list.
                     # So, the context must be based on self.variables updated sequentially.
                     # Let's regenerate eval_context inside the loop for accuracy, though less performant.
                     # Or, ensure evaluation_context is updated *after* each assignment.
                     # Simpler approach: Use the same evaluation_context, but update self.variables directly.
                     # safe_eval needs to return the value correctly.

                     try:
                          value = safe_eval(value_expr, evaluation_context) # Evaluate the right-hand side
                          if isinstance(value, (int, float)):
                               self.variables[var_name] = value # Apply the assignment
                               # Update the evaluation_context itself *immediately* so subsequent expressions in the same list
                               # use the new value. This is crucial for dependencies like `a = 1; b = a + 1`.
                               evaluation_context[var_name] = value
                               # print(f"  Cycle {self.current_cycle}, State '{self.current_state_name}': Assigned variable {var_name} to {value} based on '{value_expr}'.") # Debugging
                          else:
                               print(f"  Warning: Assignment expression '{expr}' for variable '{var_name}' evaluated to non-numeric value: {value}. Skipping update.")
                     except Exception as e:
                          print(f"  Error evaluating assignment expression '{value_expr}' for variable '{var_name}' in state '{self.current_state_obj.name}', cycle {self.current_cycle}: {e}. Skipping update.")
                     continue # Processed this expression

                else:
                     # If the expression didn't match known update patterns (like ++ or =)
                     print(f"  Warning: Unhandled variable update expression syntax for state '{self.current_state_name}', cycle {self.current_cycle}: '{expr}'. Skipping update.")

            except Exception as e:
                # Catch unexpected errors during the update processing loop itself
                print(f"  Error during variable update processing for state '{self.current_state_name}', cycle {self.current_cycle}: {e}. Skipping remaining updates for this state.")
                break # Exit loop for updates for this state


# --- Main simulation function ---
def simulate_individual_path(
    markov_chain: MarkovChain,
    start_state_name: str,
    max_steps: int,
    project_dim_info: Optional[DimensionInfo], # Needed for cumulative value keys/names
    project_tables: Dict[str, Table],         # Needed for table lookup in expressions
    initial_variables: Dict[str, float]       # Initial values for simulation variables (evaluated globals)
) -> Optional[Dict[str, float]]: # Return cumulative results (Dict[DimName, Value]) or None on failure
    """
    Simulates a single path through the Markov Chain for one individual.

    Args:
        markov_chain: The MarkovChain object to simulate.
        start_state_name: The name of the starting state.
        max_steps: Maximum number of cycles to simulate.
        project_dim_info: The DimInfo object from the Project.
        project_tables: Dictionary of Table objects from the Project.
        initial_variables: Dictionary of evaluated initial global variable values.

    Returns:
        A dictionary of cumulative results per dimension (e.g., {'Cost': 1000, 'QALY': 5})
        if simulation starts successfully and completes (or reaches max steps).
        Returns None if the start state is not found or a major error occurs early on.
    """
    _lazy_import_utils() # Ensure safe_eval is loaded
    if safe_eval is None:
         print("Error: safe_eval is not loaded. Cannot run simulation.")
         return None # Critical error

    # Map state names to state objects for quick lookup within this MC
    state_map: Dict[str, MarkovState] = {s.name: s for s in markov_chain.markov_states}

    # Find the starting state object
    start_state_obj: Optional[MarkovState] = state_map.get(start_state_name)
    if start_state_obj is None:
        print(f"Error: Start state '{start_state_name}' not found in Markov Chain '{markov_chain.name}'. Cannot start simulation path.")
        return None # Return None on critical error


    # Create an IndividualState object to track this simulation's progress and results
    individual_state = IndividualState(start_state_obj, initial_variables, project_dim_info)

    # Prepare the *base* context for expression evaluation throughout the simulation.
    # This base context contains references to things that don't change per cycle,
    # like global parameters/variables (evaluated once), and helper functions (like table lookup).
    # The MarkovChain's get_context() method evaluates parameters and initial variables.
    # Let's use the *initial* global variables evaluated by the parser, as MC's get_context might be for transition resolution.
    # A better approach is to pass the *evaluated* global parameters and the *initial* global variables from the parser.

    # Refined approach: Create base context directly here.
    # It includes initial global parameters (already evaluated by MC.get_context for transitions),
    # initial global variables (already evaluated by parser), and the table lookup function.
    # Get the initially evaluated global parameters and variables that were linked to the MC by the parser.
    # Note: This assumes mc.parameters and mc.variables are the dictionaries of Parameter/Variable objects.
    # mc.get_context() evaluates their expressions. Let's get the result of that initial evaluation.
    base_context_globals = markov_chain.get_context() # Evaluate initial global params/vars once per simulation path start


    # --- Add Table Lookup Functionality to the Base Evaluation Context ---
    # Define a function that can be called from expressions (e.g., safe_eval context)
    # This function will look up values in the parsed tables dictionary.
    # This needs access to project_tables which is passed to simulate_individual_path.
    def table_lookup_func(table_name: str, lookup_key: float, key_col: Union[int, str] = 0, value_col: Union[int, str] = 1) -> Optional[float]:
         """Allows expressions to call lookup(tableName, key, keyCol, valueCol)."""
         table = project_tables.get(table_name)
         if table:
             # Call the get_value method on the Table object
             return table.get_value(lookup_key, key_col=key_col, value_col=value_col)
         else:
             # Warning handled in Table.get_value now if table is found but columns are bad.
             # Add warning if table_name is not found at all.
             print(f"  Warning: Table '{table_name}' not found during lookup in cycle {individual_state.current_cycle}. Returning None.")
             return None # Return None if table not found

    # Add the lookup function to the base context under a name accessible by safe_eval
    base_evaluation_context: Dict[str, Any] = base_context_globals.copy() # Start with evaluated globals
    base_evaluation_context['lookup'] = table_lookup_func # Allow expressions like lookup('lifetable', t)
    # You might also add math functions if needed in expressions:
    base_evaluation_context['math'] = math # Allow expressions like math.log(...)
    # And potentially random functions if Unif/Norm etc are not pre-handled or need re-sampling:
    # base_evaluation_context['random'] = random # Potentially unsafe depending on safe_eval


    # --- Simulation Loop (Cycles) ---
    # Simulate cycle by cycle until terminated or max_steps reached.
    while individual_state.current_cycle < max_steps and not individual_state.is_terminated:
        # Get the State object for the current state in this cycle
        current_state_obj = individual_state.current_state_obj

        if current_state_obj is None:
             print(f"  Error: Current state object is None in simulation cycle {individual_state.current_cycle}. Terminating path.")
             individual_state.is_terminated = True # Should not happen if start_state_obj was valid
             break # Exit simulation loop


        # --- Step 1: Prepare Evaluation Context for this Cycle's calculations ---
        # This context is specific to the current cycle and individual.
        # It includes the base context + current cycle number ('t') + current individual variable values.
        cycle_evaluation_context = individual_state.get_evaluation_context(base_evaluation_context)


        # --- Step 2: Calculate and accumulate costs/rewards for being IN this state this cycle ---
        # Evaluate cost_expressions and reward_expressions using the cycle context.
        individual_state.add_cycle_values(project_dim_info, cycle_evaluation_context)


        # --- Step 3: Apply Variable Updates for the current state ---
        # Evaluate and apply variable_updates expressions using the cycle context.
        # Note: individual_state.apply_variable_updates updates the individual's variables directly.
        individual_state.apply_variable_updates(cycle_evaluation_context) # Pass the same cycle context


        # --- Step 4: Determine Next State Transition ---
        # Collect all possible StateTransitions from the current state (direct or via ChanceNodes).
        possible_transitions: List[StateTransition] = []
        if current_state_obj.state_transitions:
             possible_transitions = current_state_obj.state_transitions
        elif current_state_obj.chance_nodes:
             if current_state_obj.chance_nodes[0].state_transitions:
                 possible_transitions = current_state_obj.chance_nodes[0].state_transitions

        # Filter for transitions that have been successfully resolved and have a valid target state.
        valid_transitions = [st for st in possible_transitions if st.prob is not None and st.next_state is not None]

        if not valid_transitions:
            # No outgoing transitions defined or valid. State acts as a terminal state for this path.
            # print(f"  No valid transitions from state '{current_state_obj.name}' at cycle {individual_state.current_cycle}. Ending simulation path.")
            individual_state.is_terminated = True # Terminate this path
            continue # Go to the next cycle check (which will now terminate)


        # --- Step 5: Select the next state based on resolved probabilities ---
        probs = [st.prob for st in valid_transitions]
        next_states_objects = [st.next_state for st in valid_transitions]

        # Probabilities should sum to ~1.0. Handle cases where sum is zero or negative.
        valid_weights = [p if p is not None else 0.0 for p in probs]
        weight_sum = sum(valid_weights)

        if weight_sum <= 1e-9: # Sum is zero or very small
             # print(f"  Warning: Probabilities sum to ~0 ({weight_sum:.4f}) from '{current_state_obj.name}' at cycle {individual_state.current_cycle}. No transitions possible. Terminating path.")
             individual_state.is_terminated = True
             continue # Exit the while loop

        # Normalize weights if sum is off (optional but robust)
        if abs(weight_sum - 1.0) > 1e-6:
             # print(f"  Warning: Probabilities from '{current_state_obj.name}' sum to {weight_sum:.4f} at cycle {individual_state.current_cycle}. Normalizing.")
             # Use the sum for normalization to avoid division by zero if sum is slightly above 0
             probs = [p / weight_sum for p in valid_weights]
        else:
             probs = valid_weights


        try:
            # Choose the *StateTransition* object based on weighted probabilities
            # This way we get the resolved next_state directly from the chosen transition.
            chosen_transition_obj = random.choices(valid_transitions, weights=probs, k=1)[0]
            next_state_obj = chosen_transition_obj.next_state

            # Move to the next state
            if next_state_obj:
                 individual_state.current_state_obj = next_state_obj
                 individual_state.current_state_name = next_state_obj.name
                 # state_path.append(individual_state.current_state_name) # Add to path history if tracking

                 # Check for immediate termination states like "Death" name
                 if individual_state.current_state_name == "Death":
                      individual_state.is_terminated = True
                      # print(f"  Simulation path reached Death state at cycle {individual_state.current_cycle}.")

            else: # Should not happen if valid_transitions were filtered correctly
                 print(f"  Error: Chosen transition from '{current_state_obj.name}' in cycle {individual_state.current_cycle} points to a None next state object. Terminating path.")
                 individual_state.is_terminated = True
                 break # Exit simulation loop

        except ValueError as e:
             # This usually happens if weights are not positive and finite,
             # or if weights don't sum to a positive value.
             print(f"  [!] Error choosing next state from '{current_state_obj.name}' at cycle {individual_state.current_cycle} with probabilities {probs}: {e}")
             print(f"  Valid transitions attempted: {[f'{st.name} (prob={st.prob}, target={st.transition})' for st in valid_transitions]}")
             individual_state.is_terminated = True # Cannot continue if next state cannot be chosen
             break # Stop simulation on error
        except Exception as e:
             # Catch any other unexpected errors during selection
             print(f"  Error during next state selection from '{current_state_obj.name}' in cycle {individual_state.current_cycle}: {e}. Terminating path.")
             individual_state.is_terminated = True
             break # Exit simulation loop


        # --- Step 6: Check Markov Chain Termination Condition (for the *next* cycle) ---
        # This condition typically uses the cycle number *just completed* or the *next* cycle number.
        # Amua's 't' in condition seems to refer to the completed cycle number.
        # Evaluate the condition *after* deciding the state for the next cycle, but before incrementing the cycle counter.
        # However, the standard pattern is to check *after* incrementing 't' for the *start* of the next potential cycle.
        # Let's check using the cycle number *about to start* (individual_state.current_cycle + 1).
        # No, Amua condition `t==10` means terminate *after* completing cycle 10.
        # So, check after finishing the current cycle's processing, *before* incrementing for the next.

        # Revisit termination condition logic: The condition should be checked BEFORE starting the cycle if it refers to state AT the start of the cycle, or AFTER the cycle if it refers to state/time AFTER the cycle. Standard Markov models check transition probabilities for cycle t to determine state at cycle t+1. Costs/rewards/variable updates are typically applied upon *entering* or *spending time in* state at t+1. Termination is usually checked at the start of cycle t+1 or end of cycle t. Amua's `t==N` termination condition seems to be 'stop before starting cycle N+1'. So, check AFTER completing cycle 't' and transitioning, using 't'.

        # Let's check termination condition using the cycle number *just completed* (individual_state.current_cycle)
        term_context = individual_state.get_evaluation_context(base_evaluation_context) # Get context including variables *after* updates
        # 't' is already in this context, representing the cycle number just completed.

        termination_expr = markov_chain.termination_condition

        # Check explicitly if already terminated by reaching "Death" state
        if individual_state.is_terminated:
            # Path reached Death or had critical error. No need to evaluate condition.
            pass # Loop will exit

        elif termination_expr and safe_eval:
             try:
                 is_terminated_by_condition = safe_eval(termination_expr, term_context)

                 if is_terminated_by_condition is True: # Check explicitly for True
                      individual_state.is_terminated = True
                      # print(f"  Simulation path terminated by condition '{termination_expr}' at cycle {individual_state.current_cycle}.") # Debugging
                 # Note: The condition might also include state checks like 'currentState=="Death"'
                 # Our context includes 'currentStateName', so this should work if safe_eval handles string comparison.

             except Exception as e:
                  # Log error but continue up to max_steps if condition eval fails
                  print(f"  Error evaluating termination condition '{termination_expr}' in cycle {individual_state.current_cycle}: {e}. Simulation will continue up to max_steps.")
                  # individual_state.is_terminated remains False

        # --- Action 7: Prepare for Next Cycle ---
        # Increment cycle counter *after* completing the current cycle's actions and checks.
        # This prepares for the START of the next potential cycle.
        individual_state.current_cycle += 1


    # --- Simulation path finished ---
    # Return the accumulated results.
    # The IndividualState object contains the cumulative_values dictionary.
    # print(f"Simulation path finished. Total cycles: {individual_state.current_cycle}. Final State: {individual_state.current_state_name}. Cumulative Values: {individual_state.cumulative_values}") # Debugging

    return individual_state.cumulative_values # Return the dictionary of results


# Note: The calling analysis script run_analysis will call simulate_individual_path
# multiple times (for each individual) and aggregate the results.