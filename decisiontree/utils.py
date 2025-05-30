# amua_utils.py
# -*- coding: utf-8 -*-
"""
Provides utility functions used across the AMUA parsing and simulation process.
Includes safe expression evaluation, complement probability distribution, and graph drawing.
"""

import random
import re
import networkx as nx
import matplotlib.pyplot as plt
import math # Import math for functions like exp, log, sqrt, etc.
from typing import Dict, Optional, List, Any, Union # Added Union

# Lazy import models for type hints and isinstance checks to avoid circular dependency
# These should be defined in model.py
TreeNode = None
MarkovState = None
ChanceNode = None
StateTransition = None
MarkovChain = None

def _lazy_import_models():
    """Lazy import model classes from model.py."""
    global TreeNode, MarkovState, ChanceNode, StateTransition, MarkovChain
    if TreeNode is None: # Check if already imported
        try:
            # Use relative import if this is part of a package
            from model import TreeNode, MarkovState, ChanceNode, StateTransition, MarkovChain
        except ImportError:
            # Fallback for standalone execution or testing
            print("Warning: Could not import amua_model in utils. Type hints and model-specific logic may fail.")
            # Define dummy classes if necessary, but simulation will likely fail without real models.
            # Consider adding more specific error handling where model types are checked.

def amua_if(condition_expr_str: str, true_value_expr_str: str, false_value_expr_str: str, context: Dict[str, Any]) -> Any:
    """Evaluates Amua's if(condition, true_value, false_value)."""
    try:
        # Evaluate the condition expression string recursively with the provided context
        # This allows nested expressions in the condition.
        evaluated_condition = safe_eval(str(condition_expr_str), context)

        # Use Python's truthiness check on the result of the condition evaluation.
        # If the condition evaluates to None (e.g., due to an error in the condition itself),
        # treat it as false for the purpose of the if/else.
        if evaluated_condition is not None and bool(evaluated_condition):
             # Evaluate and return the true_value expression string recursively
             return safe_eval(str(true_value_expr_str), context)
        else:
             # Evaluate and return the false_value expression string recursively
             return safe_eval(str(false_value_expr_str), context)
    except Exception as e:
         # This catch is mainly for errors during the recursive calls or the boolean check itself.
         print(f"  [!] Error processing branches inside if() expression: if({condition_expr_str}, {true_value_expr_str}, {false_value_expr_str}). Error: {type(e).__name__}: {e}. Returning None for if().")
         return None

def amua_unif(min_expr_str: str, max_expr_str: str, context: Dict[str, Any]) -> float:
    """Evaluates Amua's Unif(min, max, ~) and returns a uniform random float."""
    # The third argument '~' is ignored in this implementation.
    try:
        # Evaluate min and max expressions recursively with the provided context
        # This allows min/max to be variables or complex expressions.
        evaluated_min = safe_eval(str(min_expr_str), context)
        evaluated_max = safe_eval(str(max_expr_str), context)

        # Check if evaluated results are numbers before passing to random.uniform
        if isinstance(evaluated_min, (int, float)) and isinstance(evaluated_max, (int, float)):
             # Perform uniform sampling
             # Ensure min <= max for random.uniform (Amua might allow max < min, check behavior)
             min_float = float(evaluated_min)
             max_float = float(evaluated_max)
             if min_float > max_float:
                  # Decide Amua behavior: swap? return min? return max? log error?
                  # Let's log a warning and swap for now, assuming it's a common mistake.
                  print(f"  Warning: Unif called with min ({min_float}) > max ({max_float}). Swapping values for Unif({max_float}, {min_float}).")
                  min_float, max_float = max_float, min_float # Swap

             return random.uniform(min_float, max_float)
        else:
             print(f"  [!] Invalid min/max values evaluated for Unif({min_expr_str}, {max_expr_str}, ~): need numbers, got {evaluated_min}, {evaluated_max}. Returning 0.0.")
             return 0.0 # Or raise error
    except Exception as e:
         print(f"  [!] Error evaluating Unif({min_expr_str}, {max_expr_str}, ~) expression: {type(e).__name__}: {e}. Returning 0.0.")
         return 0.0

# --- Safe Expression Evaluation Function ---
# Enhanced to handle Amua-like syntax (if, Unif) and provide basic math/random context.
def safe_eval(expr: str, context: Dict[str, Any] = {}) -> Any:
    """
    Safely evaluates an expression string by pre-processing Amua syntax
    and using a limited set of allowed functions and names.
    Handles 'if(cond, true, false)' and 'Unif(min, max, ~)', and removes '~'.
    Allows access to context variables, basic math functions, and random uniform distribution.

    Args:
        expr: The expression string to evaluate.
        context: A dictionary mapping variable names (str) to their values (float, int, bool, callable, etc.).
                 This context MUST include simulation-specific variables ('t', 'timeSick', etc.)
                 and any necessary lookup functions ('lookup').

    Returns:
        The result of the evaluation (float, int, bool), or None if evaluation fails.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None # Cannot evaluate empty string

    expr_to_eval = expr.strip() # Work with stripped expression

    # Ensure models are loaded if safe_eval is called directly (less common use case)
    _lazy_import_models()


    # --- Pre-process Amua specific syntax via string replacement ---
    # These replacements prepare the string for Python's eval.
    # We replace Amua function names with internal names used in allowed_globals.
    # Also remove '~' which is not valid Python syntax.

    # Replace '~' with nothing. Amua might use it for random number seeds/streams.
    processed_expr = expr_to_eval.replace('~', '')

    # Replace 'Unif(' with '_amua_unif_func('. Ensure it's the function name, not a variable name.
    # Use regex with word boundary \b to avoid replacing 'MyUnif' etc.
    processed_expr = re.sub(r'\bUnif\s*\(', '_amua_unif_func(', processed_expr)

    # Replace 'if(' with '_amua_if_func('. Use word boundary.
    processed_expr = re.sub(r'\bif\s*\(', '_amua_if_func(', processed_expr)

    # Add other Amua function replacements here if needed (e.g., Norm, Poisson, Lookup)
    # Note: The 'lookup' function from tables is expected to be passed directly in the context.
    # If it's called as 'Lookup(...)' in Amua, add a replacement here:
    # processed_expr = re.sub(r'\bLookup\s*\(', '_table_lookup_func(', processed_expr)


    # print(f"  [safe_eval] Original: '{expr}', Processed: '{processed_expr}'") # Debugging pre-processing


    # --- Define the allowed global namespace for eval ---
    # __builtins__ is explicitly set to None for maximum safety.
    allowed_globals: Dict[str, Any] = {
        "__builtins__": None,       # Strictly no built-ins

        # --- Internal functions mapping Amua syntax ---
        # These are the functions called AFTER the string replacement
        "_amua_if_func": lambda cond_str, true_str, false_str: amua_if(cond_str, true_str, false_str, context), # Pass context to amua_if
        "_amua_unif_func": lambda min_str, max_str: amua_unif(min_str, max_str, context), # Pass context to amua_unif
        # Add other internal functions here (e.g., _amua_norm_func, _amua_poisson_func)
        # Note: The 'lookup' function is added via the 'context' argument below.

        # --- Standard Python functions/modules commonly used in expressions ---
        # Include math functions and constants (accessible as math.func or directly if added)
        "math": math,
        "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
        # Basic numerics and safe built-ins
        "int": int, "float": float, "bool": bool, # bool might be needed for 'if' conditions
        "min": min, "max": max, "abs": abs, "round": round,
        # Add other safe functions if needed (e.g., sum, len - but check relevance)
        # "sum": sum, "len": len, # Be cautious with what types these operate on

        # --- Variables and callables provided by the simulation context ---
        # The 'context' argument contains:
        # - Evaluated global Parameters (e.g., pDie=0.05)
        # - Evaluated initial global Variables (e.g., timeSick=0.0, mSDQ=random_value)
        # - Current cycle time 't' (e.g., t=5)
        # - Current state name 'currentState' (string)
        # - The 'lookup' function for tables
        # These are added by the allowed_globals.update(context) call below.
    }

    # Update allowed_globals with the simulation context provided.
    # This makes variables like 'pDie', 'timeSick', 't', 'lookup' accessible for eval.
    # SECURITY NOTE: Ensure the context dictionary passed to safe_eval by the caller (simulator.py)
    # does NOT contain references to dangerous objects/functions (like os, sys, eval, exec, file I/O, etc.).
    # The caller is responsible for building a safe context.
    allowed_globals.update(context)


    # Define allowed names from the combined globals AFTER updating with context
    # This set includes names from allowed_globals AND context keys (variables like 'mSDQ', 'timeChemo', 't', function names like 'lookup').
    allowed_names = set(allowed_globals.keys())


    try:
        # --- Compile the processed expression to check for restricted names ---
        # This step is crucial for security. It checks the names used in the code *before* execution.
        code = compile(processed_expr, '<string>', 'eval')

        # Check all names accessed in the compiled code against our allowed list.
        # co_names contains names that are accessed as globals or built-ins within the expression.
        for name in code.co_names:
             # If a name used in the expression is not in our allowed set, raise a NameError.
             if name not in allowed_names:
                  raise NameError(f"Access to forbidden name '{name}' is denied")

        # --- Evaluate the processed expression using the strictly controlled namespace ---
        # The eval() function executes the compiled code.
        # We pass allowed_globals as both the global and local namespace. This is a common
        # safe practice for 'eval' when you want only explicitly allowed names/functions.
        result = eval(code, allowed_globals, allowed_globals) # Use allowed_globals for both scopes


        # Return the result. Caller should check type if expecting float/int/bool.
        return result

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        # Capture specific evaluation errors
        print(f"  [!] Evaluation failed for expression '{expr}': {type(e).__name__}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during eval
        print(f"  [!] Unexpected error during evaluation of '{expr}': {type(e).__name__}: {e}")
        return None

# --- Complement Probability Distribution Function (unchanged) ---
def complete_direct_state_transition_complement(markov_state: "MarkovState"):
    """
    Distributes remaining probability among complement transitions (prob="C")
    directly attached to a MarkovState.
    """
    _lazy_import_models()
    if not MarkovState or not isinstance(markov_state, MarkovState):
        # print("complete_direct_state_transition_complement called with invalid object.")
        return

    total = 0.0
    complement_nodes = []

    for st in markov_state.state_transitions:
        if st.data.get("prob") == "C":
            complement_nodes.append(st)
        elif st.prob is not None:
            # Ensure the probability is within [0, 1] before adding
            total += max(0.0, min(1.0, st.prob))

    remaining = 1.0 - total

    # Handle floating point errors and negative remainders
    if remaining < -1e-9: # Less than -0.000000001
        print(f"  [!] Warning: total direct transition probabilities ({total:.4f}) exceed 1.0 in MarkovState '{markov_state.name}' (index {markov_state.index}). Remaining set to 0.")
        remaining = 0.0
    elif remaining < 0: # Treat very small negative as 0
         remaining = 0.0

    if complement_nodes:
        if remaining > 1e-9: # Only distribute if there's significant remaining probability
            per_node = remaining / len(complement_nodes)
            for st in complement_nodes:
                # Assign probability, ensuring it's not negative due to floating point issues
                st.prob = max(0.0, per_node)
        else: # No remaining probability, set complement nodes to 0
            for st in complement_nodes:
                st.prob = 0.0

# --- Graph Drawing Helper Functions (unchanged from previous version for now) ---
# These functions are for visualization and don't impact simulation logic.
# They might need updates to use parsed coordinates or handle node identification
# across blocks if indices are not globally unique.
# --- Graph Drawing Helper Functions (unchanged from previous version) ---
# These need networkx and matplotlib
def hierarchy_pos(G: nx.DiGraph, root: Any, width: float = 1.0, vert_gap: float = 0.2, vert_loc: float = 0, xcenter: float = 0.5, max_depth: int = 100) -> Dict[Any, tuple[float, float]]:
    """Calculates a hierarchical layout for a NetworkX graph."""
    pos = {}
    visited = set()
    def _hierarchy_pos(node, left, right, level):
        if node in visited or level > max_depth: return
        visited.add(node)
        pos[node] = ((left + right) / 2.0, -level * vert_gap + vert_loc)
        children = list(G.successors(node))
        if children:
            dx = (right - left) / len(children)
            nextx = left
            for child in children:
                _hierarchy_pos(child, nextx, nextx + dx, level + 1)
                nextx += dx
    if root is not None and root in G: _hierarchy_pos(root, xcenter - width/2, xcenter + width/2, 0)
    elif root is None and G.nodes: print("Warning: No root specified for hierarchy_pos. Using the first node."); first_node = list(G.nodes())[0]; _hierarchy_pos(first_node, xcenter - width/2, xcenter + width/2, 0)
    return pos

# Corrected syntax in draw_markov_chain_tree to avoid semicolon errors
def draw_markov_chain_tree(markov_chain: "MarkovChain", figsize: tuple[int, int] = (14, 10)):
    """Draws a visual tree representation of a MarkovChain using NetworkX and Matplotlib."""
    _lazy_import_models()
    if not MarkovChain or not isinstance(markov_chain, MarkovChain):
        print("Invalid MarkovChain object provided for drawing.")
        return
    # Try importing drawing libraries late if needed
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("NetworkX or Matplotlib not installed. Cannot draw tree.")
        return


    G = nx.DiGraph()
    labels: Dict[int, str] = {}
    node_colors: List[str] = []
    node_indices_order: List[int] = [] # Track order for color mapping

    # Add MarkovChain node itself
    mc_index = markov_chain.index
    if mc_index not in G.nodes:
        G.add_node(mc_index)
        labels[mc_index] = f"MC: {markov_chain.name}"
        node_indices_order.append(mc_index)
        node_colors.append("skyblue") # Color for MC

    # Add MarkovStates and their descendants
    for ms in markov_chain.markov_states:
        ms_index = ms.index
        if ms_index not in G.nodes:
            G.add_node(ms_index)
            labels[ms_index] = f"MS: {ms.name}"
            node_indices_order.append(ms_index)
            node_colors.append("lightgreen") # Color for MS

        # Add edge from MC to MS
        G.add_edge(mc_index, ms_index)

        # MarkovState -> ChanceNode -> StateTransition
        for cn in ms.chance_nodes:
            cn_index = cn.index
            if cn_index not in G.nodes:
                 G.add_node(cn_index)
                 labels[cn_index] = f"CN: {cn.name}"
                 node_indices_order.append(cn_index)
                 node_colors.append("salmon") # Color for CN

            G.add_edge(ms_index, cn_index)

            for st in cn.state_transitions:
                st_index = st.index
                if st_index not in G.nodes:
                     G.add_node(st_index)
                     node_indices_order.append(st_index)
                     node_colors.append("lightgrey") # Color for ST

                label = f"ST: {st.name}\n→{st.transition}\nProb: {st.prob if st.prob is not None else 'C'}"
                labels[st_index] = label
                G.add_edge(cn_index, st_index, label=f"{st.prob if st.prob is not None else 'C':.4f}")

        # MarkovState -> StateTransition (Direct transitions)
        for st in ms.state_transitions:
            st_index = st.index
            if st_index not in G.nodes: # Check if already added via ChanceNode etc.
                 G.add_node(st_index)
                 node_indices_order.append(st_index)
                 node_colors.append("lightgrey") # Color for ST

            label = f"ST: {st.name}\n→{st.transition}\nProb: {st.prob if st.prob is not None else 'C'}"
            labels[st_index] = label
            G.add_edge(ms_index, st_index, label=f"{st.prob if st.prob is not None else 'C':.4f}")


    # Create the color map based on the order nodes were added to node_indices_order
    color_map = [node_colors[node_indices_order.index(node)] for node in G.nodes()]

    # Use hierarchy layout starting from the MC root
    pos = hierarchy_pos(G, root=mc_index, vert_gap=0.6, width=1.5)

    plt.figure(figsize=figsize)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_size=5000, # Increase node size
            node_color=color_map,
            edgecolors="black",
            linewidths=1.0,
            font_size=7,
            arrows=True,
            arrowstyle='->', arrowsize=15,
            node_shape='o'
           )

    # Draw edge labels separately
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6, alpha=0.8)

    plt.title(f"MarkovChain Tree: {markov_chain.name}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()