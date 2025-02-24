"""
====================================================================================================================
Neuronal Connectivity Matrix Analysis Pipeline
====================================================================================================================

Description:
This script processes neuronal connectivity data from traced neurons, categorizing input/output neurons by their 
functional types and hemispheres. It generates a directional connectivity matrix for visualizing synaptic 
interactions and functional neuron classifications.

Features:
- Load and preprocess neuronal data (with hemisphere information).
- Standardize functional classifications for consistency.
- Filter neurons by functional and anatomical criteria (e.g., hemisphere, functional classifier).
- Compute directional connectivity matrices for pre-synaptic (axon) and post-synaptic (dendrite) interactions.
- Visualize connectivity matrices with detailed functional type classifications.
- Enhance connectivity matrices using additional data from LDA-based analysis.

Modules:
1. **Data Preprocessing**:
   - Load raw data (CSV/XLSX).
   - Standardize functional classifier naming conventions.
   - Filter and group neurons by functional types.

2. **Connectivity Matrix Generation**:
   - Generate directional connectivity matrices from traced neuron data.
   - Categorize synaptic connections by hemisphere (same-side vs different-side).
   - Compute percentages of valid synaptic connections for each category.

3. **Visualization**:
   - Plot connectivity matrices with clear labels and categorized functional types.
   - Save visualizations as PDF files named based on their titles.

4. **Enhanced Connectivity Analysis**:
   - Repeat analysis with LDA-enhanced data for improved accuracy.
   - Compare base and enhanced connectivity matrices.

Inputs:
- Traced neuron data files with hemispheric and functional classifications.
- Presynaptic and postsynaptic synapse data for each neuron.

Outputs:
- Directional connectivity matrices saved as PDFs.
- Processed data for further analysis.

Dependencies:
- Python Libraries: pandas, numpy, matplotlib.
- Input Files: CSV/XLSX files containing traced neuron data and synaptic connectivity.

Author: [Jonathan Boulanger-Weill]
Date: [12/06/2024]
Institution: Harvard University
====================================================================================================================
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import pandas as pd
import numpy as np
import os
import fnmatch

# Constants
PATH_ALL_CELLS = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224.xlsx'
ROOT_FOLDER = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/all_cells_111224/'
OUTPUT_CSV = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere.csv'
OUTPUT_PATH = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes/connectivity_matrices'
LDA_CSV_PATH = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere_lda_lof.csv'

# Define the color dictionary
COLOR_CELL_TYPE_DICT = {
    "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purple
    "myelinated": (80/255, 220/255, 100/255, 0.7),                  # Dark gray for axons
    "axon_rostral": (169/255, 169/255, 169/255, 0.7),               # Gray for not functionally imaged
    "axon_caudal": (211/255, 211/255, 211/255, 0.7),                # Gray for not functionally imaged
}

# Functions
def get_inputs_outputs_by_hemisphere_general(root_folder, seed_cell_ids, hemisphere_df):
    """
    Extract and categorize input/output neurons for given seed cell IDs based on hemisphere.
    Results include same-side and different-side synapses and cells for inputs and outputs,
    along with calculated percentages for each category.
    """
    # Load hemisphere data w/out duplicates 
    hemisphere_df['nucleus_id'] = hemisphere_df['nucleus_id'].astype(str)
    # Replace nucleus_id with axon_id where nucleus_id == '0'
    hemisphere_df.loc[hemisphere_df['nucleus_id'] == '0', 'nucleus_id'] = hemisphere_df['axon_id'].astype(str)
    hemisphere_map = hemisphere_df.set_index('nucleus_id')['hemisphere'].to_dict()

    # Initialize results
    results = {
        "outputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0}
        },
        "inputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0}
        },
        "counters": {"output_seed_counter": 0, "input_seed_counter": 0}
    }

    for seed_cell_id in seed_cell_ids:
        # Determine hemisphere of the seed cell
        seed_cell_hemisphere = hemisphere_map.get(str(seed_cell_id), None)
        if seed_cell_hemisphere is None:
            print(f"Seed cell ID {seed_cell_id} has no hemisphere data. Skipping.")
            continue

        #### OUTPUTS ####
        # Find the presynaptic (output) file
        output_file_pattern = f"clem_zfish1_cell_{seed_cell_id}_ng_res_presynapses.csv"
        output_file_path = None

        # Search for the cell presynaptic file
        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, output_file_pattern):
                output_file_path = os.path.join(root, filename)
                break

        # If the cell presynaptic file is not found, fall back to the axon presynaptic file
        if not output_file_path or not os.path.exists(output_file_path):
            output_file_pattern = f"clem_zfish1_axon_{seed_cell_id}_ng_res_presynapses.csv"
            for root, _, files in os.walk(root_folder):
                for filename in fnmatch.filter(files, output_file_pattern):
                    output_file_path = os.path.join(root, filename)
                    break

        if output_file_path:
            # Load and process outputs
            outputs_data = pd.read_csv(output_file_path, comment='#', sep=' ', header=None,
                                       names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                              "prediction_status", "validation_status", "date"])
            valid_outputs = outputs_data[outputs_data['validation_status'].str.contains('valid', na=False)]
            output_ids = valid_outputs['partner_cell_id']

            traced_dendrites = output_ids[output_ids.isin(hemisphere_df['dendrite_id'])]
            matched_outputs = [
                hemisphere_df[hemisphere_df['dendrite_id'] == dendrite].iloc[0]
                for dendrite in traced_dendrites
            ] if not traced_dendrites.empty else []

            output_connected_cells = pd.DataFrame(matched_outputs)
            if not output_connected_cells.empty:
                output_connected_cells_unique = output_connected_cells.drop_duplicates(subset='axon_id')

                # Calculate percentages
                output_percentage_synapses = len(output_connected_cells) / len(valid_outputs) if len(valid_outputs) > 0 else 0
                results["outputs"]["percentages"]["synapses"] += output_percentage_synapses

                results["counters"]["output_seed_counter"] += 1

                # Categorize by hemisphere
                if 'hemisphere' in output_connected_cells_unique.columns:
                    same_outputs_cells = output_connected_cells_unique[output_connected_cells_unique['hemisphere'] == seed_cell_hemisphere]
                    different_outputs_cells = output_connected_cells_unique[output_connected_cells_unique['hemisphere'] != seed_cell_hemisphere]

                    same_outputs_synapses = output_connected_cells[output_connected_cells['hemisphere'] == seed_cell_hemisphere]
                    different_outputs_synapses = output_connected_cells[output_connected_cells['hemisphere'] != seed_cell_hemisphere]
                else:
                    same_outputs_cells = pd.DataFrame()
                    different_outputs_cells = pd.DataFrame()
                    same_outputs_synapses = pd.DataFrame()
                    different_outputs_synapses = pd.DataFrame()

                # Fill NaN with 'not functionally imaged' to handle missing classifier values 
                # Usefull when only looking at functionally imaged neurons 
                dataframes = [same_outputs_cells, different_outputs_cells, same_outputs_synapses, different_outputs_synapses]
                for i, df in enumerate(dataframes):
                    if 'functional classifier' in df.columns:
                        # Ensure we're working with a copy and modifying it in place
                        df = df.copy()
                        
                        # Explicitly cast and fill NaN values
                        df['functional classifier'] = df['functional classifier'].astype('object')
                        df.loc[:, 'functional classifier'] = df['functional classifier'].fillna('not functionally imaged')
                        
                        # Update the original DataFrame
                        dataframes[i] = df

                # Reassign modified DataFrames back to their original variables
                same_outputs_cells, different_outputs_cells, same_outputs_synapses, different_outputs_synapses = dataframes

                # Append to results
                results["outputs"]["cells"]["same_side"] = pd.concat(
                    [results["outputs"]["cells"]["same_side"], same_outputs_cells], ignore_index=True)
                results["outputs"]["cells"]["different_side"] = pd.concat(
                    [results["outputs"]["cells"]["different_side"], different_outputs_cells], ignore_index=True)
                results["outputs"]["synapses"]["same_side"] = pd.concat(
                    [results["outputs"]["synapses"]["same_side"], same_outputs_synapses], ignore_index=True)
                results["outputs"]["synapses"]["different_side"] = pd.concat(
                    [results["outputs"]["synapses"]["different_side"], different_outputs_synapses], ignore_index=True)

        #### INPUTS ####
        # Find the postsynaptic (input) file
        input_file_pattern = f"clem_zfish1_cell_{seed_cell_id}_ng_res_postsynapses.csv"
        input_file_path = None

        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, input_file_pattern):
                input_file_path = os.path.join(root, filename)
                break

        if input_file_path:
            # Load and process inputs
            inputs_data = pd.read_csv(input_file_path, comment='#', sep=' ', header=None,
                                      names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                             "prediction_status", "validation_status", "date"])
            valid_inputs = inputs_data[inputs_data['validation_status'].str.contains('valid', na=False)]
            input_ids = valid_inputs['partner_cell_id']

            traced_axons = input_ids[input_ids.isin(hemisphere_df['axon_id'])]
            matched_inputs = [
                hemisphere_df[hemisphere_df['axon_id'] == axon].iloc[0]
                for axon in traced_axons
            ] if not traced_axons.empty else []

            input_connected_cells = pd.DataFrame(matched_inputs)
            if not input_connected_cells.empty:
                input_connected_cells_unique = input_connected_cells.drop_duplicates(subset='axon_id')

                # Calculate percentages
                input_percentage_synapses = len(input_connected_cells) / len(valid_inputs) if len(valid_inputs) > 0 else 0
                results["inputs"]["percentages"]["synapses"] += input_percentage_synapses

                results["counters"]["input_seed_counter"] += 1

                # Categorize by hemisphere
                if 'hemisphere' in input_connected_cells_unique.columns:
                    same_inputs_cells = input_connected_cells_unique[input_connected_cells_unique['hemisphere'] == seed_cell_hemisphere]
                    different_inputs_cells = input_connected_cells_unique[input_connected_cells_unique['hemisphere'] != seed_cell_hemisphere]

                    same_inputs_synapses = input_connected_cells[input_connected_cells['hemisphere'] == seed_cell_hemisphere]
                    different_inputs_synapses = input_connected_cells[input_connected_cells['hemisphere'] != seed_cell_hemisphere]
                else:
                    same_inputs_cells = pd.DataFrame()
                    different_inputs_cells = pd.DataFrame()
                    same_inputs_synapses = pd.DataFrame()
                    different_inputs_synapses = pd.DataFrame()

                # Fill NaN with 'not functionally imaged' to handle missing classifier values
                # Usefull when only looking at functionally imaged neurons 
                dataframes = [same_inputs_cells, different_inputs_cells, same_inputs_synapses, different_inputs_synapses]
                for i, df in enumerate(dataframes):
                    if 'functional classifier' in df.columns:
                        # Ensure we're working with a copy and modifying it in place
                        df = df.copy()

                        # Explicitly cast and fill NaN values
                        df['functional classifier'] = df['functional classifier'].astype('object')
                        df.loc[:, 'functional classifier'] = df['functional classifier'].fillna('not functionally imaged')

                        # Update the original DataFrame
                        dataframes[i] = df

                # Reassign modified DataFrames back to their original variables
                same_inputs_cells, different_inputs_cells, same_inputs_synapses, different_inputs_synapses = dataframes

                # Append to results
                results["inputs"]["cells"]["same_side"] = pd.concat(
                    [results["inputs"]["cells"]["same_side"], same_inputs_cells], ignore_index=True)
                results["inputs"]["cells"]["different_side"] = pd.concat(
                    [results["inputs"]["cells"]["different_side"], different_inputs_cells], ignore_index=True)
                results["inputs"]["synapses"]["same_side"] = pd.concat(
                    [results["inputs"]["synapses"]["same_side"], same_inputs_synapses], ignore_index=True)
                results["inputs"]["synapses"]["different_side"] = pd.concat(
                    [results["inputs"]["synapses"]["different_side"], different_inputs_synapses], ignore_index=True)

    return results

def generate_directional_connectivity_matrix_general(root_folder, seg_ids, df_w_hemisphere):
    """
    Generate a directional connectivity matrix for functionally imaged neurons.

    Inputs:
        - root_folder: Path to the folder with neuron connectivity data.
        - df_w_hemisphere: DataFrame containing neuron metadata with hemisphere data.
    
    Output:
        - A directional connectivity matrix (inputs vs outputs) with functional IDs as labels.
    """

    # Initialize directional matrix with functional IDs as labels
    seg_ids = [str(id) for id in seg_ids]
    connectivity_matrix = pd.DataFrame(0, index=seg_ids, columns=seg_ids)

    # Initialize a set to store globally counted non-zero synapse IDs
    stored_nonzero_synapse_ids = set()

    for source_id in seg_ids:
        # Fetch the connectivity data for the source neuron
        results = get_inputs_outputs_by_hemisphere_general(root_folder, [source_id], df_w_hemisphere)

        # Try to fetch the output synapse table for the cell
        cell_file_name = f"clem_zfish1_cell_{source_id}_ng_res_presynapses.csv"
        cell_name = f"clem_zfish1_cell_{source_id}"
        cell_output_file_path = os.path.join(root_folder, cell_name, cell_file_name)

        if os.path.exists(cell_output_file_path):
            output_file_path = cell_output_file_path
        else: 
            # If the cell file doesn't exist, fall back to the axon synapse table
            axon_file_name = f"clem_zfish1_axon_{source_id}_ng_res_presynapses.csv"
            axon_name = f"clem_zfish1_axon_{source_id}"
            output_file_path = os.path.join(root_folder, axon_name, axon_file_name)
            
        outputs_data = pd.read_csv(output_file_path, comment='#', sep=' ', header=None,
                                   names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                          "prediction_status", "validation_status", "date"])
        valid_outputs = outputs_data[outputs_data['validation_status'].str.contains('valid', na=False)]

        # Process OUTPUT connections
        for direction in ["same_side", "different_side"]:
            outputs = results["outputs"]["synapses"][direction]
            if not outputs.empty and "nucleus_id" in outputs.columns:
                # Ensure seg_ids is not empty
                if len(seg_ids) > 0:  # Check that seg_ids is not empty
                    try:
                        # Convert both outputs["nucleus_id"] and seg_ids to integers
                        outputs["nucleus_id"] = outputs["nucleus_id"].astype(int)
                        seg_ids = [int(id) for id in seg_ids]

                        # Filter the outputs DataFrame
                        outputs = outputs[outputs["nucleus_id"].isin(seg_ids)]
                    except ValueError as e:
                        print(f"Error converting nucleus_id or seg_ids to integers: {e}")
                else:
                    print("Warning: seg_ids is empty. No filtering applied.")

                for _, output_row in outputs.iterrows():
                    # Isolate corresponding synapses
                    target_dendrite = output_row["dendrite_id"]
                    matching_row = valid_outputs[valid_outputs["partner_cell_id"] == target_dendrite]
                    synapse_ids = matching_row['synapse_id'].tolist()

                    # Count the number of synapse_id == 0
                    zero_synapse_count = synapse_ids.count(0)

                    # Filter non-zero synapse IDs
                    nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]

                    # Identify new non-zero synapses not already counted
                    new_nonzero_synapses = [sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids]

                    # Calculate the total number of new synapses
                    num_new_synapses = zero_synapse_count + len(new_nonzero_synapses)

                    # Update the connectivity matrix if there are new synapses
                    if num_new_synapses > 0:
                        target_func_id = str(output_row["nucleus_id"])
                        connectivity_matrix.loc[source_id, target_func_id] += num_new_synapses  # Update OUTPUTS ONLY

                        # Add new non-zero synapses to the globally stored set
                        stored_nonzero_synapse_ids.update(new_nonzero_synapses)

        # Process INPUT connections
        # Fetch the input synapse table
        file_name = f"clem_zfish1_cell_{source_id}_ng_res_postsynapses.csv"
        input_file_path = os.path.join(root_folder, cell_name, file_name)

        # Check if the file exists before processing (it won't exist for an axon)
        if os.path.exists(input_file_path):
            
            inputs_data = pd.read_csv(input_file_path, comment='#', sep=' ', header=None,
                                    names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                            "prediction_status", "validation_status", "date"])
            valid_inputs = inputs_data[inputs_data['validation_status'].str.contains('valid', na=False)]

            for direction in ["same_side", "different_side"]:
                inputs = results["inputs"]["synapses"][direction]
                if not inputs.empty and "nucleus_id" in inputs.columns:
                    
                    # Ensure seg_ids is not empty
                    if len(seg_ids) > 0:  # Check that seg_ids is not empty
                        try:
                            # Convert both outputs["nucleus_id"] and seg_ids to integers
                            inputs["nucleus_id"] = inputs["nucleus_id"].astype(int)
                            seg_ids = [int(id) for id in seg_ids]

                            # Filter the outputs DataFrame
                            inputs = inputs[inputs["nucleus_id"].isin(seg_ids)]
                        except ValueError as e:
                            print(f"Error converting nucleus_id or seg_ids to integers: {e}")
                    else:
                        print("Warning: seg_ids is empty. No filtering applied.")

                    for _, input_row in inputs.iterrows():
                        # Isolate corresponding synapses
                        target_axon = input_row["axon_id"]
                        matching_row = valid_inputs[valid_inputs["partner_cell_id"] == target_axon]
                        synapse_ids = matching_row['synapse_id'].tolist()

                        # Count the number of synapse_id == 0
                        zero_synapse_count = synapse_ids.count(0)

                        # Filter non-zero synapse IDs
                        nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]

                        # Identify new non-zero synapses not already counted
                        new_nonzero_synapses = [sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids]

                        # Calculate the total number of new synapses
                        num_new_synapses = zero_synapse_count + len(new_nonzero_synapses)

                        # Update the connectivity matrix if there are new synapses
                        if num_new_synapses > 0: 
                            source_input_func_id = str(input_row["nucleus_id"])
                            connectivity_matrix.loc[source_input_func_id, source_id] += num_new_synapses  # Update INPUTS ONLY

                            # Add new non-zero synapses to the globally stored set
                            stored_nonzero_synapse_ids.update(new_nonzero_synapses)

    return connectivity_matrix

def plot_connectivity_matrix(matrix, functional_types, output_path, title="Directional Connectivity Matrix"):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Define functional type order
    CATEGORY_ORDER = ["axon_rostral", "integrator_ipsilateral", "integrator_contralateral", 
                      "dynamic_threshold", "motor_command", "myelinated", "axon_caudal"]

    # Filter functional types to match the matrix
    functional_types = {k: v for k, v in functional_types.items() if k in matrix.index}

    # Sort indices based on functional type order
    def sort_key(func_id):
        category = functional_types.get(func_id, "unknown")
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    sorted_indices = sorted(matrix.index, key=sort_key)
    sorted_matrix = matrix.loc[sorted_indices, sorted_indices]

    # Group boundaries
    group_boundaries = []
    last_type = None
    for i, idx in enumerate(sorted_indices):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(sorted_indices) - 0.5)

    # Define colormap and bounds
    cmap = mcolors.ListedColormap(["white", "blue", "green", "yellow", "pink", "red"])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Replace all values >5 with 5 (to plot as red) 
    matrix_clipped = np.clip(sorted_matrix, 0, 5)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_clipped, cmap=cmap, norm=norm)

    # Add color bar
    cbar = plt.colorbar(
        cax, ax=ax, boundaries=bounds, ticks=[0, 1, 2, 3, 4, 5], spacing="uniform",
        orientation="horizontal", pad=0.1
    )
    cbar.set_label("No. of Synapses")

    # Labels and title
    ax.set_xticks(range(len(sorted_matrix.columns)))
    ax.set_yticks(range(len(sorted_matrix.index)))
    ax.set_xticklabels(sorted_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_matrix.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title, fontsize=12)

    # Functional type bars
    for i, functional_id in enumerate(sorted_matrix.index):
        functional_type = functional_types[functional_id]
        color = COLOR_CELL_TYPE_DICT.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray

        # Left bar
        ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
        # Top bar
        ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

    # Add gridlines to separate groups
    for boundary in group_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.5, zorder=3)
        ax.axvline(boundary, color='black', linewidth=1.5, zorder=3)

    # Adjust axis limits
    ax.set_xlim(-1.5, len(sorted_matrix.columns) - 0.5)
    ax.set_ylim(len(sorted_matrix.index) - 0.5, -1.5)

    # Adjust layout and save
    plt.tight_layout()
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    output_pdf_path = os.path.join(output_path, f"{sanitized_title}.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

def load_and_clean_data(path, drop_duplicates=True):
    """Load data and optionally drop duplicates by 'axon_id'"""
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    if drop_duplicates:
        df = df.drop_duplicates(subset='axon_id')
    return df

def standardize_naming(df):
    """Standardize naming in 'functional classifier' column."""
    replacements = {'dynamic threshold': 'dynamic_threshold', 'motor command': 'motor_command'}
    df['functional classifier'] = df['functional classifier'].replace(replacements)
    return df

def fetch_filtered_ids(df, col_1, condition_1, col_2=None, condition_2=None):
    """Filter DataFrame by conditions and return unique nucleus and functional IDs."""
    filtered = df[df.iloc[:, col_1] == condition_1]
    if col_2 and condition_2:
        filtered = filtered[filtered.iloc[:, col_2] == condition_2]
    return filtered.iloc[:, 5].drop_duplicates(), filtered.iloc[:, 1].drop_duplicates()

def create_nucleus_id_groups(df):
    """Group nucleus IDs by functional types."""
    groups = {
        "axon_rostral": df.loc[(df['type'] == 'axon') & (df['comment'] == 'axon exits the volume rostrally'), 'axon_id'],
        "integrator_ipsilateral": fetch_filtered_ids(df, 9, 'integrator', 11, 'ipsilateral')[0],
        "integrator_contralateral": fetch_filtered_ids(df, 9, 'integrator', 11, 'contralateral')[0],
        "dynamic_threshold": fetch_filtered_ids(df, 9, 'dynamic_threshold')[0],
        "motor_command": fetch_filtered_ids(df, 9, 'motor_command')[0],
        "myelinated": df.loc[(df['type'] == 'cell') & (df['functional classifier'] == 'myelinated'), 'nucleus_id'],
        "axon_caudal": df.loc[(df['type'] == 'axon') & (df['comment'] == 'axon exits the volume caudally'), 'axon_id']
    }
    return {k: [str(id) for id in v] for k, v in groups.items()}

def generate_functional_types(nucleus_id_groups):
    """Create a dictionary mapping nucleus IDs to functional types."""
    return {nucleus_id: functional_type
            for functional_type, ids in nucleus_id_groups.items()
            for nucleus_id in ids}

def filter_connectivity_matrix(matrix, functional_types):
    """Filter the connectivity matrix and functional types by non-zero indices."""
    non_zero_indices = matrix.index[(matrix.sum(axis=1) != 0) | (matrix.sum(axis=0) != 0)]
    filtered_matrix = matrix.loc[non_zero_indices, non_zero_indices]
    filtered_types = {k: v for k, v in functional_types.items() if k in filtered_matrix.index}
    return filtered_matrix, filtered_types

# Main Workflow
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix without enhancement  

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(OUTPUT_CSV)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)
plot_connectivity_matrix(filtered_matrix, filtered_types, OUTPUT_PATH, title="cm_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix with enhancement

# Step 5: Repeat with LDA-enhanced data
lda_data = load_and_clean_data(LDA_CSV_PATH)
lda_data = standardize_naming(lda_data)

nucleus_id_groups_lda = create_nucleus_id_groups(lda_data)
functional_types_lda = generate_functional_types(nucleus_id_groups_lda)

all_ids_nuc_lda = np.concatenate([v for v in nucleus_id_groups_lda.values()])
connectivity_matrix_lda = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc_lda, lda_data)

filtered_matrix_lda, filtered_types_lda = filter_connectivity_matrix(connectivity_matrix_lda, functional_types_lda)
plot_connectivity_matrix(filtered_matrix_lda, filtered_types_lda, OUTPUT_PATH, title="cm_lda_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split 

def create_nucleus_id_groups_hemisphere(df):
    """Group nucleus IDs by functional types and hemispheres."""
    groups = {
        # Axon rostral, split by hemisphere
        "axon_rostral_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_rostral_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ],
        # Integrators split by hemisphere
        "integrator_ipsilateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        "integrator_contralateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_contralateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Dynamic threshold, split by hemisphere
        "dynamic_threshold_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "dynamic_threshold_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Motor command, split by hemisphere
        "motor_command_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "motor_command_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Myelinated, split by hemisphere
        "myelinated_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "myelinated_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Axon caudal, split by hemisphere
        "axon_caudal_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_caudal_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ]
    }
    # Convert all IDs to strings for consistency
    return {k: [str(id) for id in v] for k, v in groups.items()}

COLOR_CELL_TYPE_DICT = {
    # Integrator ipsilateral
    "integrator_ipsilateral_left": (254/255, 179/255, 38/255, 0.7),   # Yellow-orange
    "integrator_ipsilateral_right": (254/255, 179/255, 38/255, 0.7),  # Yellow-orange

    # Integrator contralateral
    "integrator_contralateral_left": (232/255, 77/255, 138/255, 0.7), # Magenta-pink
    "integrator_contralateral_right": (232/255, 77/255, 138/255, 0.7),# Magenta-pink

    # Dynamic threshold
    "dynamic_threshold_left": (100/255, 197/255, 235/255, 0.7),       # Light blue
    "dynamic_threshold_right": (100/255, 197/255, 235/255, 0.7),      # Light blue

    # Motor command
    "motor_command_left": (127/255, 88/255, 175/255, 0.7),            # Purple
    "motor_command_right": (127/255, 88/255, 175/255, 0.7),           # Purple

    # Myelinated
    "myelinated_left": (80/255, 220/255, 100/255, 0.7),               # Green
    "myelinated_right": (80/255, 220/255, 100/255, 0.7),              # Green

    # Axon rostral
    "axon_rostral_left": (169/255, 169/255, 169/255, 0.7),            # Gray
    "axon_rostral_right": (169/255, 169/255, 169/255, 0.7),           # Gray

    # Axon caudal
    "axon_caudal_left": (211/255, 211/255, 211/255, 0.7),             # Light gray
    "axon_caudal_right": (211/255, 211/255, 211/255, 0.7),            # Light gray
}

def plot_connectivity_matrix(matrix, functional_types, output_path, title="cm_lda_lr_split_111224"):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Define functional type order: only the specified categories
    CATEGORY_ORDER = [
        "integrator_ipsilateral_left", "integrator_contralateral_left",
        "dynamic_threshold_left", "motor_command_left", "myelinated_left",
        "integrator_ipsilateral_right", "integrator_contralateral_right",
        "dynamic_threshold_right", "motor_command_right", "myelinated_right",
    ]

    # Filter functional types to match CATEGORY_ORDER
    functional_types = {
        k: v for k, v in functional_types.items() if v in CATEGORY_ORDER and k in matrix.index
    }

    # Filter matrix to include only rows/columns from CATEGORY_ORDER
    filtered_indices = [
        idx for idx in matrix.index if functional_types.get(idx, "unknown") in CATEGORY_ORDER
    ]
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]

    # Sort indices based on CATEGORY_ORDER
    def sort_key(func_id):
        category = functional_types.get(func_id, "unknown")
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    sorted_indices = sorted(filtered_indices, key=sort_key)
    sorted_matrix = filtered_matrix.loc[sorted_indices, sorted_indices]

    # Group boundaries
    group_boundaries = []
    last_type = None
    for i, idx in enumerate(sorted_indices):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(sorted_indices) - 0.5)

    # Define colormap and bounds
    cmap = mcolors.ListedColormap(["white", "blue", "green", "yellow", "pink", "red"])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Replace all values >5 with 5 (to plot as red)
    matrix_clipped = np.clip(sorted_matrix, 0, 5)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_clipped, cmap=cmap, norm=norm)

    # Add color bar
    cbar = plt.colorbar(
        cax, ax=ax, boundaries=bounds, ticks=[0, 1, 2, 3, 4, 5], spacing="uniform",
        orientation="horizontal", pad=0.1
    )
    cbar.set_label("No. of Synapses")

    # Labels and title
    ax.set_xticks(range(len(sorted_matrix.columns)))
    ax.set_yticks(range(len(sorted_matrix.index)))
    ax.set_xticklabels(sorted_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_matrix.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title, fontsize=12)

    # Functional type bars
    for i, functional_id in enumerate(sorted_matrix.index):
        functional_type = functional_types[functional_id]
        color = COLOR_CELL_TYPE_DICT.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray

        # Left bar
        ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
        # Top bar
        ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

    # Add gridlines to separate groups
    for boundary in group_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.5, zorder=3)
        ax.axvline(boundary, color='black', linewidth=1.5, zorder=3)

    # Adjust axis limits
    ax.set_xlim(-1.5, len(sorted_matrix.columns) - 0.5)
    ax.set_ylim(len(sorted_matrix.index) - 0.5, -1.5)

    # Adjust layout and save
    plt.tight_layout()
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    output_pdf_path = os.path.join(output_path, f"{sanitized_title}.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(OUTPUT_CSV)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups_hemisphere(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)
plot_connectivity_matrix(filtered_matrix, filtered_types, OUTPUT_PATH, title="cm_lr_split_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split with enhancement 

# Step 1: Load and preprocess data
lda_data = load_and_clean_data(LDA_CSV_PATH)
lda_data = standardize_naming(lda_data)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups_lda = create_nucleus_id_groups_hemisphere(lda_data)
functional_types_lda = generate_functional_types(nucleus_id_groups_lda)

# Step 3: Generate connectivity matrix
all_ids_nuc_lda = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix_lda = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc_lda, lda_data)

# Step 4: Filter and plot connectivity matrix
filtered_matrix_lda, filtered_types_lda = filter_connectivity_matrix(connectivity_matrix_lda, functional_types_lda)
plot_connectivity_matrix(filtered_matrix_lda, filtered_types_lda, OUTPUT_PATH, title="cm_lda_lr_split_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split + E.I sign

def process_matrix(matrix, df):
    """
    Modify the matrix rows based on the 'neurotransmitter classifier' in df.

    Parameters:
    - matrix (pd.DataFrame): The matrix to be modified.
    - df (pd.DataFrame): DataFrame containing the 'neurotransmitter classifier' and 'nucleus_id' columns.
    
    Returns:
    - pd.DataFrame: Modified matrix.
    """
    # Iterate through each row in the matrix
    for idx in matrix.index:
        # Fetch the corresponding row in df where 'nucleus_id' matches the matrix index
        df_row = df.loc[df['nucleus_id'] == idx]
        
        # Ensure there is a match
        if df_row.empty:
            raise ValueError(f"Index '{idx}' in the matrix does not have a matching 'nucleus_id' in df.")
        
        # Extract the neurotransmitter classifier
        classifier = df_row.iloc[0]['neurotransmitter classifier']
        
        if classifier == 'inhibitory':
            # Multiply the entire row by -1
            matrix.loc[idx] *= -1
        elif classifier == 'unknown':
            # Replace non-zero values with NaN
            matrix.loc[idx] = matrix.loc[idx].apply(lambda x: np.nan if x != 0 else 0)
        # If classifier is 'excitatory', no change is needed

    return matrix

processed_matrix = process_matrix(filtered_matrix, all_cells)
filtered_matrix, filtered_types = filter_connectivity_matrix(processed_matrix, functional_types)

def plot_connectivity_matrix_ei(matrix, functional_types, output_path, title="Directional Connectivity Matrix"):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Define functional type order: only the specified categories
    CATEGORY_ORDER = [
        "integrator_ipsilateral_left", "integrator_contralateral_left",
        "dynamic_threshold_left", "motor_command_left", "myelinated_left",
        "integrator_ipsilateral_right", "integrator_contralateral_right",
        "dynamic_threshold_right", "motor_command_right", "myelinated_right",
    ]

    # Filter functional types to match CATEGORY_ORDER
    functional_types = {
        k: v for k, v in functional_types.items() if v in CATEGORY_ORDER and k in matrix.index
    }

    # Filter matrix to include only rows/columns from CATEGORY_ORDER
    filtered_indices = [
        idx for idx in matrix.index if functional_types.get(idx, "unknown") in CATEGORY_ORDER
    ]
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]

    # Sort indices based on CATEGORY_ORDER
    def sort_key(func_id):
        category = functional_types.get(func_id, "unknown")
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    sorted_indices = sorted(filtered_indices, key=sort_key)
    sorted_matrix = filtered_matrix.loc[sorted_indices, sorted_indices]

    # Clip matrix values between -2 and 2
    clipped_matrix = np.clip(sorted_matrix, -2, 2)

    # Define the color palette for the discrete colormap
    colors = [
        "#D62839",  # Strongly Inhibitory (Ruby Red)
        "#F88F54",  # Weakly Inhibitory (Peach Orange)
        "#FFFFFF",  # Zero (White)
        "#9FD598",  # Weakly Excitatory (Pale Jade Green)
        "#227C71"   # Strongly Excitatory (Teal Green)
    ]
    cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Handle NaNs separately (map them to a neutral gray background)
    matrix_with_nan = clipped_matrix.copy()
    nan_mask = np.isnan(matrix_with_nan)
    matrix_with_nan[nan_mask] = -3  # Assign a value outside bounds for NaNs

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_with_nan, cmap=cmap, norm=norm)

    # Add a white background
    ax.set_facecolor("white")

    # Add color bar
    cbar = plt.colorbar(
        cax, ax=ax, boundaries=bounds, ticks=[-2, -1, 0, 1, 2], spacing="uniform",
        orientation="horizontal", pad=0.1
    )
    cbar.set_label("Synaptic Strength")

    # Labels and title
    ax.set_xticks(range(len(sorted_matrix.columns)))
    ax.set_yticks(range(len(sorted_matrix.index)))
    ax.set_xticklabels(sorted_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_matrix.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title, fontsize=12)

    # Functional type bars
    for i, functional_id in enumerate(sorted_matrix.index):
        functional_type = functional_types[functional_id]
        color = COLOR_CELL_TYPE_DICT.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray

        # Left bar
        ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
        # Top bar
        ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

    # Add gridlines to separate groups
    group_boundaries = []
    last_type = None
    for i, idx in enumerate(sorted_indices):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(sorted_indices) - 0.5)

    for boundary in group_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.5, zorder=3)
        ax.axvline(boundary, color='black', linewidth=1.5, zorder=3)

    # Adjust axis limits
    ax.set_xlim(-1.5, len(sorted_matrix.columns) - 0.5)
    ax.set_ylim(len(sorted_matrix.index) - 0.5, -1.5)

    # Adjust layout and save
    plt.tight_layout()
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    output_pdf_path = os.path.join(output_path, f"{sanitized_title}.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

plot_connectivity_matrix_ei(filtered_matrix, filtered_types, OUTPUT_PATH, title="cm_lr_split_ei_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split + E.I sign with enhancement 

processed_matrix_lda = process_matrix(filtered_matrix_lda, lda_data)
filtered_matrix_lda, filtered_types_lda = filter_connectivity_matrix(processed_matrix_lda, functional_types_lda)

def plot_connectivity_matrix_ei(matrix, functional_types, output_path, title="Directional Connectivity Matrix"):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Define functional type order: only the specified categories
    CATEGORY_ORDER = [
        "integrator_ipsilateral_left", "integrator_contralateral_left",
        "dynamic_threshold_left", "motor_command_left", "myelinated_left",
        "integrator_ipsilateral_right", "integrator_contralateral_right",
        "dynamic_threshold_right", "motor_command_right", "myelinated_right",
    ]

    # Filter functional types to match CATEGORY_ORDER
    functional_types = {
        k: v for k, v in functional_types.items() if v in CATEGORY_ORDER and k in matrix.index
    }

    # Filter matrix to include only rows/columns from CATEGORY_ORDER
    filtered_indices = [
        idx for idx in matrix.index if functional_types.get(idx, "unknown") in CATEGORY_ORDER
    ]
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]

    # Sort indices based on CATEGORY_ORDER
    def sort_key(func_id):
        category = functional_types.get(func_id, "unknown")
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    sorted_indices = sorted(filtered_indices, key=sort_key)
    sorted_matrix = filtered_matrix.loc[sorted_indices, sorted_indices]

    # Clip matrix values between -2 and 2
    clipped_matrix = np.clip(sorted_matrix, -2, 2)

    # Define the color palette for the discrete colormap
    colors = [
        "#D62839",  # Strongly Inhibitory (Ruby Red)
        "#F88F54",  # Weakly Inhibitory (Peach Orange)
        "#FFFFFF",  # Zero (White)
        "#9FD598",  # Weakly Excitatory (Pale Jade Green)
        "#227C71"   # Strongly Excitatory (Teal Green)
    ]
    cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Handle NaNs separately (map them to a neutral gray background)
    matrix_with_nan = clipped_matrix.copy()
    nan_mask = np.isnan(matrix_with_nan)
    matrix_with_nan[nan_mask] = -3  # Assign a value outside bounds for NaNs

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_with_nan, cmap=cmap, norm=norm)

    # Add a white background
    ax.set_facecolor("white")

    # Add color bar
    cbar = plt.colorbar(
        cax, ax=ax, boundaries=bounds, ticks=[-2, -1, 0, 1, 2], spacing="uniform",
        orientation="horizontal", pad=0.1
    )
    cbar.set_label("Synaptic Strength")

    # Labels and title
    ax.set_xticks(range(len(sorted_matrix.columns)))
    ax.set_yticks(range(len(sorted_matrix.index)))
    ax.set_xticklabels(sorted_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_matrix.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title, fontsize=12)

    # Functional type bars
    for i, functional_id in enumerate(sorted_matrix.index):
        functional_type = functional_types[functional_id]
        color = COLOR_CELL_TYPE_DICT.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray

        # Left bar
        ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
        # Top bar
        ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

    # Add gridlines to separate groups
    group_boundaries = []
    last_type = None
    for i, idx in enumerate(sorted_indices):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(sorted_indices) - 0.5)

    for boundary in group_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.5, zorder=3)
        ax.axvline(boundary, color='black', linewidth=1.5, zorder=3)

    # Adjust axis limits
    ax.set_xlim(-1.5, len(sorted_matrix.columns) - 0.5)
    ax.set_ylim(len(sorted_matrix.index) - 0.5, -1.5)

    # Adjust layout and save
    plt.tight_layout()
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    output_pdf_path = os.path.join(output_path, f"{sanitized_title}.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

plot_connectivity_matrix_ei(filtered_matrix_lda, filtered_types_lda, OUTPUT_PATH, title="cm_lda_r_split_ei_111224")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Order E/I, pool L/R
## LDA + Axons + L/R + E/I  

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Drafty draft 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Define the color palette for the discrete color map
colors = [
    "#D62839",  # Strongly Inhibitory (Ruby Red)
    "#F88F54",  # Weakly Inhibitory (Peach Orange)
    "#C0C0C0",  # Neutral (Silver Gray)
    "#9FD598",  # Weakly Excitatory (Pale Jade Green)
    "#227C71"   # Strongly Excitatory (Teal Green)
]

# Create the colormap and normalization
cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
bounds = [-2, -1, 0, 1, 2, 3]  # Discrete bounds for the color categories
norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)

# Example matrix with values ranging from -2 to 2
example_matrix = np.array([
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 1, 2],
    [0, 1, 2, -1, -2],
    [1, 2, -2, -1, 0],
    [2, -2, -1, 0, 1]
])

# Plot the matrix
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(example_matrix, cmap=cmap, norm=norm)

# Add colorbar
cbar = plt.colorbar(
    cax, ticks=[-2, -1, 0, 1, 2], boundaries=bounds, spacing='uniform', orientation='vertical', pad=0.1
)
cbar.ax.set_yticklabels(['Strongly Inhibitory', 'Weakly Inhibitory', 'Neutral', 'Weakly Excitatory', 'Strongly Excitatory'])
cbar.set_label("Connection Type", rotation=270, labelpad=15)

# Add labels
ax.set_title("Example Connectivity Matrix")
plt.show()
