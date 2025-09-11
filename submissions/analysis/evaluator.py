# submissions/analysis/evaluator.py
import pandas as pd
import networkx as nx
import numpy as np
import re
import time
import io
from django.conf import settings
from storages.backends.gcloud import GoogleCloudStorage

# Assume evaluate_trilemma is defined elsewhere
# from .trilemma_calculator import evaluate_trilemma

# ==============================================================================
# --- Main Evaluator Class ---
# ==============================================================================

class SubmissionEvaluator:
    """
    Handles the entire process of evaluating a submission by:
    1. Loading the static base data (topology, reinforcement costs) from GCS.
    2. Loading the submission-specific profile data from GCS.
    3. Running the trilemma analysis across all stations.
    4. Aggregating the results into the three final scores.
    """
    def __init__(self):
        print("Initializing SubmissionEvaluator...")
        self.df_full = None
        self.df_reinforcement = None
        self.storage = GoogleCloudStorage()
        self._load_static_data()
        self.NOMINAL_VOLTAGE = 400.0

    def _load_static_data(self):
        """Loads the non-changing data files (topology, costs) from GCS, enforcing correct data types."""
        try:
            print("Loading static evaluation data from Google Cloud Storage...")
            
            # This dtype map helps, but we need a more robust cleaning step below.
            topology_dtype_map = {
                'station': str, 'from': str, 'to': str, 'id_equ': str,
                'name': str, 'clazz': str, 'line_id': str, 'cable_id': str,
                'transformer_id': str, 'switch_id': str, 'node_id': str,
                'cable_type': str 
            }

            with self.storage.open(settings.GRID_TOPOLOGY_FILE_PATH, 'rb') as f:
                self.df_full = pd.read_csv(f, sep=";", dtype=topology_dtype_map, na_filter=False)
            
            # --- FIX: Add robust data cleaning for critical columns ---
            # 1. Drop any rows where 'station' is missing, as they cannot be processed.
            #    Using .replace with an empty string first handles potential whitespace-only cells.
            self.df_full['station'] = self.df_full['station'].str.strip()
            self.df_full.replace('', np.nan, inplace=True)
            initial_rows = len(self.df_full)
            self.df_full.dropna(subset=['station'], inplace=True)
            final_rows = len(self.df_full)
            if initial_rows > final_rows:
                print(f"INFO: Dropped {initial_rows - final_rows} rows with missing station names from topology data.")

            # 2. Force conversion to string to eliminate any lingering non-string types (like float NaNs).
            self.df_full['station'] = self.df_full['station'].astype(str)
            self.df_full['from'] = self.df_full['from'].astype(str)
            self.df_full['to'] = self.df_full['to'].astype(str)
            # --- END OF FIX ---

            with self.storage.open(settings.REINFORCEMENT_COSTS_FILE_PATH, 'rb') as f:
                self.df_reinforcement = pd.read_csv(f)
            
            print("Successfully loaded and cleaned static data.")
        except Exception as e:
            print(f"FATAL ERROR: Could not load static evaluation data. Error: {e}")
            raise
    
        
    def _load_submission_profiles(self, submission):
        """
        Downloads and loads all profile Parquet files for a given submission.
        Returns a dictionary of DataFrames.
        """
        profiles = {}
        file_mapping = {
            'base_consumption': submission.file1, 'pv_profiles': submission.file2,
            'ev_profiles': submission.file3, 'hp_profiles': submission.file4,
            'battery_in': submission.file5, 'battery_out': submission.file6,
            'battery_soc': submission.file7, 'curtailed_energy': submission.file8,
        }

        print(f"Loading profiles for submission {submission.id}...")
        for name, file_field_object in file_mapping.items():
            if file_field_object and file_field_object.name:
                try:
                    with self.storage.open(file_field_object.name, 'rb') as f:
                        # --- FIX 1: Read the parquet file into a temporary df ---
                        temp_df = pd.read_parquet(io.BytesIO(f.read()))

                        # --- FIX 2: Set 'timestamp' as the index if it exists ---
                        if 'timestamp' in temp_df.columns:
                            temp_df = temp_df.set_index('timestamp')

                        profiles[name] = temp_df
                except Exception as e:
                    print(f"Warning: Could not read Parquet file '{file_field_object.name}' for {name}. Error: {e}. Skipping.")
                    profiles[name] = pd.DataFrame()
            else:
                profiles[name] = pd.DataFrame()

        base_df = profiles.get('base_consumption')
        if base_df is None or base_df.empty:
            raise ValueError("Evaluation failed: base_consumption.parquet is missing or empty.")

        # --- FIX 3: Use the index from the base dataframe as the master index ---
        # This is more robust than creating a new date_range.
        master_index = base_df.index
        
        for key, df in profiles.items():
            if not df.empty:
                # Reindex all other dataframes to match the base_consumption index.
                # This handles missing timestamps and ensures alignment.
                df = df.reindex(master_index).fillna(0)

                # Enforce that all column names are strings
                df.columns = df.columns.astype(str)

                profiles[key] = df

        return profiles
        
    def run_evaluation(self, submission):
        """
        The main public method to evaluate a submission and return the three scores.
        """
        try: # Add a try...except block to catch the error and inspect data
            print("DEBUG: 1. Starting run_evaluation...")
            profiles = self._load_submission_profiles(submission)

            print("DEBUG: 2. Profiles loaded. Combining into df_net_load...")
            df_net_load = (profiles.get('base_consumption', pd.DataFrame())
                        .add(profiles.get('ev_profiles', pd.DataFrame()), fill_value=0)
                        .add(profiles.get('hp_profiles', pd.DataFrame()), fill_value=0)
                        .add(profiles.get('battery_in', pd.DataFrame()), fill_value=0)
                        .subtract(profiles.get('pv_profiles', pd.DataFrame()), fill_value=0)
                        .subtract(profiles.get('battery_out', pd.DataFrame()), fill_value=0)
                        .add(profiles.get('curtailed_energy', pd.DataFrame()), fill_value=0))
            
            print("DEBUG: 3. df_net_load created. Reindexing...")
            if 'base_consumption' in profiles and not profiles['base_consumption'].empty:
                df_net_load = df_net_load.reindex(profiles['base_consumption'].index).fillna(0)
            else:
                raise ValueError("Cannot proceed without a valid base_consumption profile.")

            print("DEBUG: 4. Reindexing complete. Getting unique station names...")
            # This line should now be safe because of the cleaning in _load_static_data
            all_station_names = sorted(self.df_full['station'].unique())
            print(f"DEBUG: 5. Found {len(all_station_names)} stations. Starting loop...")
            
            
            aggregated_results = {
                'grid_reinforcement_cost_chf': 0.0,
                'installed_pv_kwp': 0.0,
                'autarchy_percentages': []
            }

            for station_name in all_station_names:
                print(f"---> Analyzing station: {station_name}")
                
                station_data_for_analysis = self.df_full[self.df_full['station'] == station_name]
                
                all_nodes_in_station = pd.concat([station_data_for_analysis['from'], station_data_for_analysis['to']]).unique()
                customer_ids = sorted([
                    node for node in all_nodes_in_station 
                    if node.startswith('HAS')
                ])
                
                if not customer_ids:
                    print(f"     -> No customers found for station {station_name}. Skipping.")
                    continue
                
                station_customer_cols = [col for col in customer_ids if col in df_net_load.columns]
                if not station_customer_cols:
                    print(f"     -> No profile data found for any customers in station {station_name}. Skipping.")
                    continue
                
                df_net_load_station_only = df_net_load[station_customer_cols]

                try:
                    station_results = evaluate_trilemma(
                        station_name=station_name,
                        leg_customer_ids=customer_ids,
                        df_full_topology=station_data_for_analysis, 
                        all_profiles=profiles,
                        df_net_load_full_station=df_net_load_station_only,
                        df_reinforcement_costs=self.df_reinforcement,
                        nominal_voltage=self.NOMINAL_VOLTAGE
                    )
                except TypeError as e:
                    # This block remains as a good safeguard for future issues
                    print("\n" + "="*80)
                    print(f"FATAL: A TypeError occurred inside evaluate_trilemma for station '{station_name}'.")
                    print(f"Error message: {e}")
                    print("="*80 + "\n")
                    raise
                
                aggregated_results['grid_reinforcement_cost_chf'] += station_results['grid_reinforcement_cost_chf']
                aggregated_results['installed_pv_kwp'] += station_results['installed_pv_kwp']
                aggregated_results['autarchy_percentages'].append(station_results['autarchy_percentage'])

            final_grid_cost = aggregated_results['grid_reinforcement_cost_chf']
            final_pv_installed = aggregated_results['installed_pv_kwp']
            
            final_autarchy = np.mean(aggregated_results['autarchy_percentages']) if aggregated_results['autarchy_percentages'] else 0.0

            print(f"--- Final Aggregated Scores for Submission {submission.id} ---")
            print(f"Total Grid Cost: {final_grid_cost:,.2f} CHF")
            print(f"Total Installed PV: {final_pv_installed:,.2f} kWp")
            print(f"Average Autarchy: {final_autarchy:.2f} %")
            
            return {
                'grid_costs': final_grid_cost,
                'renewables_installed': final_pv_installed,
                'autarchy_rate': final_autarchy / 100.0
            }

        except TypeError as e:
            print("\n" + "="*80)
            print("DEBUG: CAUGHT THE TYPEERROR. INSPECTING DATA STATE...")
            print(f"Error Message: {e}")

            print("\n--- Inspecting df_full (Topology) ---")
            print(self.df_full.info())
            print("Sample of 'from' and 'to' columns:")
            print(self.df_full[['station', 'from', 'to']].head())

            print("\n--- Inspecting Loaded Profiles ---")
            for name, df in profiles.items():
                if not df.empty:
                    print(f"Profile '{name}' | Columns type: {df.columns.dtype} | First 5 columns: {df.columns[:5].tolist()}")
                else:
                    print(f"Profile '{name}' is empty.")
            
            print("="*80 + "\n")
            raise e # Re-raise the exception after printing debug info

# ==============================================================================
# --- PASTE ALL YOUR ANALYSIS HELPER FUNCTIONS BELOW ---
# (build_and_simplify_network, find_failures_with_yearly_profile, 
#  suggest_grid_reinforcement, evaluate_trilemma, etc.)
# ==============================================================================
# EDH-Hackathon/src/functions.py

"""
This module provides functions for analyzing a low-voltage electrical grid.
It includes capabilities to:
1.  Build and simplify a network graph from raw CSV data.
2.  Simulate grid load over a yearly time-series profile to identify overloads.
"""

import pandas as pd
import networkx as nx
import numpy as np
import re
import operator
import time

# ==============================================================================
# --- NETWORK BUILDING AND SIMPLIFICATION (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def build_and_simplify_network(df_station: pd.DataFrame) -> tuple:
    """
    Processes raw network data for a SINGLE STATION, builds a simplified graph,
    and prunes dangling edges.

    MODIFICATION: Added coordinate columns to the numeric conversion step to
                  prevent TypeError during visualization.

    Args:
        df_station (pd.DataFrame): DataFrame for a single station.

    Returns:
        tuple: (G_simplified, consumer_properties, root_node_ids)
    """
    station_name = df_station['station'].iloc[0] if not df_station.empty else "Unknown"
    print(f"\n--- Building and simplifying network for: {station_name} ---")

    df_raw = df_station.copy()

    # Step 1 is unchanged...
    # ==============================================================================
    # === Step 1: Normalize Consumer Connection Direction ===
    # ==============================================================================
    print("Step 1: Normalizing consumer connection directions...")
    reversed_mask = df_raw['to'].str.startswith('HAS', na=False) & \
                    ~df_raw['from'].str.startswith('HAS', na=False)
    df_raw.loc[reversed_mask, ['from', 'to']] = \
        df_raw.loc[reversed_mask, ['to', 'from']].values
    
    # ==============================================================================
    # --- Step 2: Convert columns to appropriate data types ---
    # <<< FIX APPLIED HERE >>>
    # ==============================================================================
    print("Step 2: Converting columns to appropriate data types...")
    # Add the coordinate columns to the list of columns to be converted to numeric
    numeric_cols = [
        'length', 'ratedCurrent', 'Irmax_hoch', 'X', 'R', 'X0', 'R0', 
        'C', 'G', 'C0', 'x1', 'y1', 'x2', 'y2'
    ]
    string_cols = ['from', 'to', 'id_equ', 'name', 'station']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    for col in string_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype('string')
    if 'normalOpen' in df_raw.columns:
        df_raw['normalOpen'] = df_raw['normalOpen'].astype(bool)

    # ==============================================================================
    # --- Step 2.5: Extract Node Coordinates ---
    # ==============================================================================
    print("Step 2.5: Extracting node coordinates...")
    node_coordinates = {}
    coord_cols = ['x1', 'y1', 'x2', 'y2']
    
    if all(col in df_raw.columns for col in coord_cols):
        from_nodes = df_raw[['from', 'x1', 'y1']].rename(
            columns={'from': 'node_id', 'x1': 'x', 'y1': 'y'}
        )
        to_nodes = df_raw[['to', 'x2', 'y2']].rename(
            columns={'to': 'node_id', 'x2': 'x', 'y2': 'y'}
        )
        all_nodes_df = pd.concat([from_nodes, to_nodes]).dropna().drop_duplicates(subset=['node_id'])
        node_coordinates = {row.node_id: (row.x, row.y) for row in all_nodes_df.itertuples()}
        print(f"  -> Extracted coordinates for {len(node_coordinates)} unique nodes.")
    else:
        print("  -> Coordinate columns ('x1', 'y1', 'x2', 'y2') not found. Skipping coordinate extraction.")

    # The rest of the function remains unchanged...
    # ==============================================================================
    # --- Step 3: Segregating network data ---
    # ==============================================================================
    print("Step 3: Segregating network data...")
    is_consumer_row = df_raw['from'].str.startswith('HAS', na=False)
    is_transformer_row = df_raw['clazz'] == 'PowerTransformer'
    df_consumers = df_raw[is_consumer_row].copy()
    df_transformers = df_raw[is_transformer_row].copy()
    df_edges = df_raw[~is_transformer_row].copy()
    transformer_pins = set()
    if not df_transformers.empty:
        transformer_pins = set(df_transformers['from']).union(set(df_transformers['to']))
        
    # ==============================================================================
    # --- Step 4: Infer and Store Consumer Properties ---
    # ==============================================================================
    print("Step 4: Parsing and storing consumer properties...")
    STANDARD_FUSE_SIZES = sorted([35, 50, 63, 80, 100, 125, 160, 200, 250], reverse=True)
    def _infer_fuse_rating(name_str: str, cable_imax: float) -> float:
        if pd.notna(name_str):
            match = re.search(r'(\d+)\s*A', str(name_str))
            if match: return float(match.group(1))
        if cable_imax > 0:
            for fuse_size in STANDARD_FUSE_SIZES:
                if fuse_size <= cable_imax: return float(fuse_size)
        return 63.0

    consumer_properties = {}
    for consumer_id, group in df_consumers.groupby('from'):
        consumer_row = group.iloc[0]
        cable_imax = consumer_row.get('Irmax_hoch', 0)
        consumer_properties[consumer_id] = {
            'is_consumer': True, 'consumer_Imax': cable_imax,
            'consumer_fuse_A': _infer_fuse_rating(consumer_row['name'], cable_imax),
            'consumer_R': consumer_row.get('R', 0), 'consumer_X': consumer_row.get('X', 0),
        }

    # ==============================================================================
    # --- Step 5: Simplifying Network Topology ---
    # ==============================================================================
    print("Step 5: Simplifying network topology (merging nodes and parallel edges)...")
    G_multi = nx.from_pandas_edgelist(df_edges, source='from', target='to', edge_attr=True, create_using=nx.MultiGraph)
    df_zero_length_edges = df_edges[df_edges['length'] == 0]
    G_contracted = G_multi.copy()
    node_to_representative = {}

    if not df_zero_length_edges.empty:
        G_zero_length = nx.from_pandas_edgelist(df_zero_length_edges, 'from', 'to')
        components_to_merge = list(nx.connected_components(G_zero_length))
        for component in components_to_merge:
            representative_node = sorted(list(component), key=lambda x: (x not in transformer_pins, x.startswith('HAS'), x))[0]
            for node in component:
                node_to_representative[node] = representative_node
                if node != representative_node and G_contracted.has_node(node):
                    G_contracted = nx.contracted_nodes(G_contracted, representative_node, node, self_loops=False)
    
    G_simplified = nx.Graph()
    for u, v in G_contracted.edges():
        if G_simplified.has_edge(u, v): continue
        parallel_edges = list(G_contracted.get_edge_data(u, v).values())
        agg_imax = sum(edge.get('Irmax_hoch', 0) for edge in parallel_edges)
        agg_length = np.mean([edge.get('length', 0) for edge in parallel_edges])
        total_conductance_G, total_susceptance_B = 0, 0
        for edge in parallel_edges:
            R, X = edge.get('R', 0), edge.get('X', 0)
            z_squared = R**2 + X**2
            if z_squared > 0:
                total_conductance_G += R / z_squared
                total_susceptance_B += -X / z_squared
        y_squared = total_conductance_G**2 + total_susceptance_B**2
        agg_r = total_conductance_G / y_squared if y_squared > 0 else 0
        agg_x = -total_susceptance_B / y_squared if y_squared > 0 else 0
        G_simplified.add_edge(u, v, Irmax_hoch=agg_imax, length=agg_length, R=agg_r, X=agg_x, parallel_count=len(parallel_edges))

    for original_node, props in consumer_properties.items():
        final_node_name = node_to_representative.get(original_node, original_node)
        if not G_simplified.has_node(final_node_name): G_simplified.add_node(final_node_name)
        if 'contained_consumers' not in G_simplified.nodes[final_node_name]:
            G_simplified.nodes[final_node_name].update({'contained_consumers': [], 'is_consumer_connection': True})
        G_simplified.nodes[final_node_name]['contained_consumers'].append(original_node)
        
    if node_coordinates:
        print("  -> Attaching coordinates to simplified graph nodes...")
        nodes_with_coords = 0
        for node in G_simplified.nodes():
            coords = node_coordinates.get(node)
            if coords:
                G_simplified.nodes[node]['pos'] = coords
                nodes_with_coords += 1
        print(f"  -> Successfully attached coordinates to {nodes_with_coords}/{G_simplified.number_of_nodes()} nodes.")

    # ==============================================================================
    # --- Step 6: Detecting all root nodes (transformers) ---
    # ==============================================================================
    print("Step 6: Detecting root nodes (transformer LV-sides)...")
    root_node_ids = set()
    if not df_transformers.empty:
        for _, row in df_transformers.iterrows():
            from_node, to_node = row['from'], row['to']
            rep_from = node_to_representative.get(from_node, from_node)
            rep_to = node_to_representative.get(to_node, to_node)
            degree_from = G_simplified.degree(rep_from) if G_simplified.has_node(rep_from) else 0
            degree_to = G_simplified.degree(rep_to) if G_simplified.has_node(rep_to) else 0
            if degree_to > degree_from: root_node_ids.add(rep_to)
            elif degree_from > degree_to: root_node_ids.add(rep_from)
            elif degree_to > 0: root_node_ids.add(rep_to)
            else: print(f"  -> Warning: Transformer between {from_node} and {to_node} seems disconnected.")
    root_node_ids = list(root_node_ids)

    if not root_node_ids and G_simplified.number_of_nodes() > 0:
        print("  -> Warning: No valid 'PowerTransformer' elements found. Falling back to centrality.")
        centrality = nx.betweenness_centrality(G_simplified)
        single_root = max(centrality, key=centrality.get)
        root_node_ids.append(single_root)
        print(f"  -> Selected '{single_root}' as the single root based on fallback method.")
    for root_id in root_node_ids:
        if G_simplified.has_node(root_id):
            G_simplified.nodes[root_id]['is_transformer'] = True

    # ==============================================================================
    # --- Step 7: Prune Dangling Edges ---
    # ==============================================================================
    print("Step 7: Pruning dangling edges without consumers...")
    nodes_pruned_total = 0
    while True:
        nodes_to_remove = []
        for node in G_simplified.nodes():
            is_leaf = G_simplified.degree(node) == 1
            is_consumer = G_simplified.nodes[node].get('is_consumer_connection', False)
            is_root = node in root_node_ids

            if is_leaf and not is_consumer and not is_root:
                nodes_to_remove.append(node)
        
        if not nodes_to_remove:
            break
        else:
            G_simplified.remove_nodes_from(nodes_to_remove)
            nodes_pruned_total += len(nodes_to_remove)

    if nodes_pruned_total > 0:
        print(f"  -> Finished pruning. A total of {nodes_pruned_total} nodes were removed.")
    else:
        print("  -> No dangling non-consumer edges found to prune.")
    
    print(f"✅ Network build complete for {station_name}. Detected {len(root_node_ids)} root node(s): {root_node_ids}")

    return G_simplified, consumer_properties, root_node_ids

# ==============================================================================
# --- NETWORK FAILURE ANALYSIS (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def find_failures_with_yearly_profile(
    graph: nx.Graph,
    net_profile_df: pd.DataFrame,
    consumer_props: dict,
    root_node_ids: list[str], # MODIFIED: Now accepts a list of root nodes
    nominal_voltage: float = 230.0,
    power_factor: float = 1.0
) -> dict:
    """
    Finds network failures by simulating power flow for a yearly profile.
    
    This version is updated to handle networks with multiple transformers (root nodes)
    by using a "super source" simulation method.

    This analysis is performed in two parts:
    1.  Fuse Failures: A worst-case check on each consumer's max injection. (Unchanged)
    2.  Link Failures: A time-series simulation. It iterates through every
        timestep, calculates the current flowing through each cable towards all
        sources, and records the maximum current to check against thermal limits.

    Args:
        graph (nx.Graph): The simplified network graph.
        net_profile_df (pd.DataFrame): DataFrame with consumer IDs as columns and
                                       net power (kW) at each timestep as rows.
        consumer_props (dict): Dictionary of consumer properties.
        root_node_ids (list[str]): A list of IDs for all transformer/grid connection points.
        nominal_voltage (float): The nominal phase-to-neutral voltage (V).
        power_factor (float): The power factor for converting power to current.

    Returns:
        dict: A dictionary containing 'fuse_failures', 'link_failures', and the
              graph with analysis results attached ('graph_analysis').
    """
    t_start_total = time.time()
    print("\n--- Starting Yearly Profile-Based Network Analysis (Multi-Transformer Mode) ---")

    # --- Input Validation for Multiple Roots ---
    if not isinstance(root_node_ids, list) or not root_node_ids:
        raise ValueError("`root_node_ids` must be a non-empty list.")
    for root_id in root_node_ids:
        if not graph.has_node(root_id):
            raise ValueError(f"Root node '{root_id}' from the list was not found in the graph.")

    g_analysis = graph.copy()

    # --- Part A: Check Consumer Fuse Limits (Worst-Case Injection) ---
    # This part is independent of network topology and remains unchanged.
    print("Step A: Checking for fuse failures based on max yearly injection...")
    t_start_fuse = time.time()
    fuse_failures = []
    max_injection_kw = net_profile_df.min()

    for consumer_id, injection_kw in max_injection_kw.items():
        if injection_kw >= 0:
            continue
        if consumer_id not in consumer_props:
            continue
        
        # P = V * I * pf  =>  I = P / (V * pf)
        # Using phase-to-neutral voltage, so no sqrt(3) is needed.
        power_watts = abs(injection_kw) * 1000
        generated_current = power_watts / (nominal_voltage * power_factor)
        fuse_limit = consumer_props[consumer_id].get('consumer_fuse_A', np.inf)

        if generated_current > fuse_limit:
            overload = ((generated_current - fuse_limit) / fuse_limit) * 100
            fuse_failures.append({
                'consumer_id': consumer_id,
                'fuse_limit_A': fuse_limit,
                'generated_current_A': round(generated_current, 2),
                'overload_percentage': round(overload, 1)
            })
    t_end_fuse = time.time()
    print(f"  -> Fuse check completed in {t_end_fuse - t_start_fuse:.2f} seconds.")


    # --- Part B: Check Network Link Overloads (Time-Series Simulation) ---
    print("Step B: Simulating power flow using the 'Super Source' method...")
    t_start_simulation = time.time()

    # --- Setup the Super Source for simulation ---
    SUPER_SOURCE_ID = 'SUPER_SOURCE'
    g_analysis.add_node(SUPER_SOURCE_ID)
    for root_id in root_node_ids:
        # Add a zero-impedance, infinite-capacity link from each real root to the super source
        g_analysis.add_edge(root_id, SUPER_SOURCE_ID, Irmax_hoch=np.inf, R=0, X=0, length=0)
    
    # Initialize a "high-water mark" for current on each edge.
    nx.set_edge_attributes(g_analysis, 0.0, 'max_observed_current_A')

    # The recursive function is unchanged, but it will now trace paths to the SUPER_SOURCE.
    def _calculate_upstream_flow(node_id, visited_nodes):
        visited_nodes.add(node_id)
        
        power_watts = g_analysis.nodes[node_id].get('current_timestep_kw', 0) * -1000
        current_from_this_node = power_watts / (nominal_voltage * power_factor)
        
        total_downstream_current = 0
        for neighbor_id in g_analysis.neighbors(node_id):
            if neighbor_id not in visited_nodes:
                child_current = _calculate_upstream_flow(neighbor_id, visited_nodes)
                g_analysis.edges[node_id, neighbor_id]['calculated_current'] = child_current
                total_downstream_current += child_current
        
        return current_from_this_node + total_downstream_current

    # Main simulation loop
    num_timesteps = len(net_profile_df)
    for i, (timestamp, series) in enumerate(net_profile_df.iterrows()):
        if (i + 1) % 5000 == 0:
            print(f"  ...processed {i+1} of {num_timesteps} timesteps...")

        # 1. Update node power for the current timestep
        nx.set_node_attributes(g_analysis, 0, 'current_timestep_kw')
        for node, data in g_analysis.nodes(data=True):
            if 'contained_consumers' in data:
                total_kw = sum(series.get(cons_id, 0) for cons_id in data['contained_consumers'])
                g_analysis.nodes[node]['current_timestep_kw'] = total_kw
    
        # 2. Run the power flow calculation starting from the SUPER_SOURCE.
        _calculate_upstream_flow(SUPER_SOURCE_ID, visited_nodes=set())

        # 3. Update the max_observed_current for each edge.
        for u, v, data in g_analysis.edges(data=True):
            current_now = abs(data.get('calculated_current', 0))
            if current_now > data['max_observed_current_A']:
                g_analysis.edges[u, v]['max_observed_current_A'] = current_now

    t_end_simulation = time.time()
    print("  ...simulation complete.")
    print(f"  -> Time-series simulation completed in {t_end_simulation - t_start_simulation:.2f} seconds.")


    # 4. Final Assessment: Find failures, ignoring the artificial super source links.
    link_failures = []
    # Iterate over original graph edges to avoid including the artificial ones.
    for u, v, data in graph.edges(data=True):
        # Get the max observed current from the analysis graph
        max_observed = g_analysis.edges[u, v].get('max_observed_current_A', 0)
        max_allowed = data.get('Irmax_hoch', 0)

        if max_allowed > 0 and max_observed > max_allowed:
            overload = ((max_observed - max_allowed) / max_allowed) * 100
            link_failures.append({
                'link': tuple(sorted((u, v))),
                'max_allowed_current_A': round(max_allowed, 2),
                'calculated_current_A': round(max_observed, 2),
                'overload_percentage': round(overload, 1)
            })

    # Clean up the analysis graph by removing the super source for a cleaner output
    g_analysis.remove_node(SUPER_SOURCE_ID)
    
    t_end_total = time.time()
    print(f"\n--- Analysis Finished. Total elapsed time: {t_end_total - t_start_total:.2f} seconds. ---")
            
    return {'fuse_failures': fuse_failures, 'link_failures': link_failures, "graph_analysis": g_analysis}



# ==============================================================================
# --- NETWORK FAILURE ANALYSIS (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def suggest_grid_reinforcement(
    initial_graph: nx.Graph,
    initial_results: dict,
    reinforcement_costs_df: pd.DataFrame,
    # Parameters needed for re-analysis
    net_profile_df: pd.DataFrame,
    consumer_props: dict,
    root_node_ids: list,
    nominal_voltage: float,
    max_iterations: int = 50 # Kept for signature compatibility, but not used in logic
) -> dict:
    """
    Analyzes grid failures and suggests a cost-effective reinforcement plan by
    applying all necessary changes in a single, efficient run.

    This method works by:
    1. Identifying ALL overloaded links from the initial failure analysis.
    2. For each failed link, determining the cheapest cable upgrade that meets its peak demand.
    3. Aggregating all upgrades into a single reinforcement plan.
    4. Applying all changes to the graph model.
    5. Running a final verification analysis to confirm the solution is effective.

    Args:
        initial_graph: The original simplified graph.
        initial_results: The results from the first failure analysis run.
        reinforcement_costs_df: DataFrame with costs and specs for upgrades.
        net_profile_df, consumer_props, root_node_ids, nominal_voltage:
            Parameters required to re-run the failure analysis for verification.
        max_iterations: Safeguard parameter, not used in this single-run approach.

    Returns:
        A dictionary containing the final status, total cost, and reinforcement plan.
    """
    print("\n" + "="*20 + "\n--- Starting Grid Reinforcement (Single Run Mode) ---\n" + "="*20)
    
    # --- 1. Prepare Inputs ---
    g_reinforced = initial_graph.copy()
    initial_failures = initial_results['link_failures']
    
    if not initial_failures:
        print("✅ No link failures were found in the initial analysis. No reinforcement needed.")
        return {
            'status': 'Success',
            'total_cost_CHF': 0.0,
            'reinforcement_plan': pd.DataFrame(),
            'reinforced_graph': g_reinforced
        }

    # Pre-process the reinforcement options for quick lookups
    df_lines = reinforcement_costs_df[
        reinforcement_costs_df['material'] == 'line'
    ].sort_values('cost').copy()
    df_lines['Irmax'] = pd.to_numeric(df_lines['Irmax'], errors='coerce')
    
    non_repairable_cost_row = reinforcement_costs_df[reinforcement_costs_df['type'] == 'large']
    NON_REPAIRABLE_COST = non_repairable_cost_row['cost'].iloc[0] if not non_repairable_cost_row.empty else 244728.1650

    total_cost = 0.0
    reinforcement_plan = []
    
    print(f"Found {len(initial_failures)} overloaded links to fix. Planning all upgrades now.")

    # --- 2. Plan All Reinforcements Based on Initial Analysis ---
    for i, failure in enumerate(initial_failures):
        link_to_fix = failure['link']
        required_current = failure['calculated_current_A']
        u, v = link_to_fix
        
        # Find the cheapest valid cable upgrade for this specific link
        possible_upgrades = df_lines[df_lines['Irmax'] > required_current]
        
        if possible_upgrades.empty:
            print(f"\n🚨 CRITICAL FAILURE: Link {link_to_fix} requires > {required_current:.2f} A.")
            print(f"   The maximum available cable capacity is {df_lines['Irmax'].max()} A.")
            print("   The grid is considered non-repairable with the available materials.")
            return {
                'status': 'Non-Repairable',
                'total_cost_CHF': NON_REPAIRABLE_COST,
                'reason': f"Link {link_to_fix} requires > {required_current:.2f} A, but max available is {df_lines['Irmax'].max()} A.",
                'reinforcement_plan': pd.DataFrame(reinforcement_plan)
            }
        
        cheapest_upgrade = possible_upgrades.iloc[0]
        
        # Calculate cost and prepare the action log
        link_data = g_reinforced.edges[u, v]
        link_length = link_data.get('length', 0)
        
        fix_cost = 0
        if link_length > 0:
            fix_cost = cheapest_upgrade['cost'] * link_length
        
        total_cost += fix_cost
        
        action = {
            'fix_order': i + 1,
            'fixed_link': link_to_fix,
            'overload_percentage': failure['overload_percentage'],
            'original_Imax_A': link_data.get('Irmax_hoch'),
            'required_Imax_A': required_current,
            'new_cable_type': cheapest_upgrade['type'],
            'new_cable_Imax_A': cheapest_upgrade['Irmax'],
            'cost_CHF': round(fix_cost, 2)
        }
        reinforcement_plan.append(action)
        
        # Apply the fix to the graph model in memory
        g_reinforced.edges[u, v]['Irmax_hoch'] = cheapest_upgrade['Irmax']
        g_reinforced.edges[u, v]['reinforcement_type'] = cheapest_upgrade['type']

    print("\n--- Summary of Planned Reinforcements ---")
    print(pd.DataFrame(reinforcement_plan).to_string(index=False))
    print(f"\nTotal estimated cost: {total_cost:.2f} CHF")
    
    # --- 3. Verification Step ---
    print("\n--- Verifying the Complete Reinforcement Plan ---")
    print("-> Re-running failure analysis on the fully modified grid...")
    
    final_results = find_failures_with_yearly_profile(
        graph=g_reinforced,
        net_profile_df=net_profile_df,
        consumer_props=consumer_props,
        root_node_ids=root_node_ids,
        nominal_voltage=nominal_voltage
    )
    
    # --- 4. Finalize and Return Results ---
    if final_results['link_failures']:
        print("\n" + "="*20 + "\n--- Reinforcement Verification FAILED ---\n" + "="*20)
        print("🚨 After applying all upgrades, new failures were detected. This indicates a complex network effect.")
        print("   This can happen if changing cable impedances (not modeled here) reroutes current in unexpected ways.")
        print("   The proposed plan is not a complete solution.")
        
        return {
            'status': 'Failed (Verification Check)',
            'total_cost_CHF': round(total_cost, 2),
            'reinforcement_plan': pd.DataFrame(reinforcement_plan),
            'remaining_failures_after_fix': final_results['link_failures']
        }

    print("\n" + "="*20 + "\n--- Reinforcement Complete and Verified! ---\n" + "="*20)
    print("✅ All link failures have been successfully resolved with the proposed plan.")
    
    return {
        'status': 'Success',
        'total_cost_CHF': round(total_cost, 2),
        'reinforcement_plan': pd.DataFrame(reinforcement_plan),
        'reinforced_graph': g_reinforced
    }


# --- Helper function for printing results consistently ---
def print_analysis_results(title, results):
    print(f"\n{'='*20}\n--- {title} ---\n{'='*20}")
    fuse_failures = results['fuse_failures']
    link_failures = results['link_failures']

    if not fuse_failures and not link_failures:
        print("\n✅ SUCCESS: The network is robust under these conditions. No overloads detected.")
        return

    if fuse_failures:
        print(f"\n🚨 FUSE FAILURES: Found {len(fuse_failures)} overloaded consumer fuses.")
        display(pd.DataFrame(fuse_failures))
    else:
        print("\n✅ No fuse failures were detected.")

    if link_failures:
        print(f"\n🚨 LINK FAILURES: Found {len(link_failures)} overloaded cables.")
        display(pd.DataFrame(link_failures))
    else:
        print("\n✅ No link/cable failures were detected.")
        
        
        

import os

def update_and_save_parquet(new_data_df, file_path, customers_to_update):
    """
    Saves or updates a Parquet file with new profile data for a specific set of customers.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing the new profile data. 
                                    It can be a large frame, but only data from 
                                    'customers_to_update' will be used.
        file_path (str): The full path to the Parquet file to be saved.
        customers_to_update (list): A list of customer IDs whose data should be
                                    updated or added to the file.
    """
    # Filter the new data to only include columns for the customers we just analyzed
    relevant_new_data = new_data_df[customers_to_update]

    if os.path.exists(file_path):
        print(f"File '{os.path.basename(file_path)}' exists. Loading and updating...")
        try:
            existing_df = pd.read_parquet(file_path)
            
            # Update existing columns and add new ones from the relevant new data
            for col in relevant_new_data.columns:
                existing_df[col] = relevant_new_data[col]
            
            final_df = existing_df
            print(f"Updated data for {len(customers_to_update)} customers.")

        except Exception as e:
            print(f"Error reading existing file {file_path}: {e}. Overwriting with new data.")
            final_df = relevant_new_data
    else:
        print(f"File '{os.path.basename(file_path)}' does not exist. Creating new file...")
        final_df = relevant_new_data

    try:
        final_df.to_parquet(file_path, index=True)
        print(f"Successfully saved data to '{file_path}'")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")


def evaluate_trilemma(
    station_name: str,
    leg_customer_ids: list,
    df_full_topology: pd.DataFrame,
    all_profiles: dict,
    df_net_load_full_station: pd.DataFrame,
    df_reinforcement_costs: pd.DataFrame,
    nominal_voltage: float =400.0
):
    """
    Evaluates the energy trilemma for a given Local Energy Community (LEG).

    The trilemma is assessed based on three key metrics:
    1. Grid Reinforcement Cost: The cost to upgrade the shared grid infrastructure
       of the station to handle the total load, including the LEG's contribution.
    2. Installed PV Capacity: The total peak power (kWp) of all PV systems
       within the specified LEG.
    3. Autarchy: The self-sufficiency rate of the LEG, calculated as the ratio of
       self-consumed energy to the total energy consumed by the LEG.

    Args:
        station_name (str): The name of the station the LEG belongs to.
        leg_customer_ids (list): A list of customer IDs (e.g., 'HAS-xxxx') forming the LEG.
        df_full_topology (pd.DataFrame): The complete network topology data (`df_full`).
        all_profiles (dict): A dictionary containing all profile DataFrames
                             (e.g., {"base_consumption": df_consumption, "pv_profiles": df_pv, ...}).
        df_net_load_full_station (pd.DataFrame): The pre-calculated net load for the entire station.
        df_reinforcement_costs (pd.DataFrame): DataFrame with reinforcement costs.

    Returns:
        dict: A dictionary containing the trilemma results:
              {'grid_reinforcement_cost_chf': float,
               'installed_pv_kwp': float,
               'autarchy_percentage': float}
    """
    print("\n" + "="*80)
    print(f"--- Starting Trilemma Evaluation for LEG in Station: '{station_name}' ---")
    print(f"LEG consists of {len(leg_customer_ids)} members.")
    print("="*80 + "\n")
    
    # ==========================================================================
    # --- 1. Calculate Grid Reinforcement Cost for the Entire Station ---
    # The cost depends on the total load of the station, as infrastructure is shared.
    # ==========================================================================
    print(f"--- [1/3] Evaluating Grid Reinforcement Cost for Station: '{station_name}' ---")
    df_one_station = df_full_topology[df_full_topology['station'] == station_name].copy()
    
    # Build the network graph for the station
    G, consumer_props, roots = build_and_simplify_network(df_one_station)
    
    print("\nRunning dynamic profile analysis to find failures...")
    dynamic_results = find_failures_with_yearly_profile(
        graph=G,
        net_profile_df=df_net_load_full_station,
        consumer_props=consumer_props,
        root_node_ids=roots,
        nominal_voltage=nominal_voltage
    )
    
    print("\nCalculating suggested grid reinforcements...")
    reinforcement_results = suggest_grid_reinforcement(
        initial_graph=G,
        initial_results=dynamic_results,
        reinforcement_costs_df=df_reinforcement_costs,
        net_profile_df=df_net_load_full_station,
        consumer_props=consumer_props,
        root_node_ids=roots,
        nominal_voltage=nominal_voltage
    )
    
    grid_cost = reinforcement_results.get('total_cost_CHF', 0.0)
    print(f"✅ Finished Grid Cost Calculation.")
    print(f"--> Estimated Grid Reinforcement Cost: {grid_cost:.2f} CHF\n")

    # ==========================================================================
    # --- 2. Calculate Total Installed PV Capacity for the LEG ---
    # ==========================================================================
    print(f"--- [2/3] Calculating Installed PV Capacity for the LEG ---")
    
    # --- FIX: Use the correct key 'pv_profiles' instead of 'PV' ---
    df_pv_profiles = all_profiles.get("pv_profiles")
    
    # This check is crucial. If there's no PV file, df_pv_profiles will be None.
    if df_pv_profiles is None or df_pv_profiles.empty:
        leg_pv_customers = []
    else:
        # Filter for LEG customers that actually have a PV profile
        leg_pv_customers = [cid for cid in leg_customer_ids if cid in df_pv_profiles.columns]
    
    if not leg_pv_customers:
        installed_pv = 0.0
    else:
        # Get the max value (peak power in kW) for each customer and sum them up
        installed_pv = df_pv_profiles[leg_pv_customers].max().sum()
        
    print(f"✅ Finished PV Capacity Calculation.")
    print(f"--> Total Installed PV in LEG: {installed_pv:.2f} kWp\n")

    # ==========================================================================
    # --- 3. Calculate Autarchy (Self-Sufficiency) for the LEG ---
    # ==========================================================================
    print(f"--- [3/3] Calculating Autarchy for the LEG ---")

    def get_leg_profile(profile_df, default_val=0):
        """Helper to safely get and sum profiles for LEG members."""
        if profile_df is None or profile_df.empty:
            # FIX: Use the guaranteed base_consumption index as a reference
            base_index = all_profiles["base_consumption"].index
            return pd.Series(default_val, index=base_index)
        
        # Find which of the LEG customers are in this dataframe's columns
        leg_cols = [cid for cid in leg_customer_ids if cid in profile_df.columns]
        
        if not leg_cols:
            return pd.Series(default_val, index=profile_df.index)
            
        return profile_df[leg_cols].sum(axis=1)

    # --- FIX: Use the correct, consistent keys for all profiles ---
    total_load_profile_kw = (get_leg_profile(all_profiles.get("base_consumption")) +
                             get_leg_profile(all_profiles.get("ev_profiles")) +
                             get_leg_profile(all_profiles.get("hp_profiles")) +
                             get_leg_profile(all_profiles.get("battery_in")))

    total_generation_profile_kw = (get_leg_profile(all_profiles.get("pv_profiles")) +
                                   get_leg_profile(all_profiles.get("battery_out")))

    # Convert power (kW) to energy (kWh) by multiplying by the time interval in hours (15 min = 0.25 h)
    time_interval_h = 0.25
    total_energy_consumed_kwh = total_load_profile_kw.sum() * time_interval_h
    
    if total_energy_consumed_kwh == 0:
        autarchy_percentage = 0.0
        print("-> Total energy consumption for the LEG is zero. Autarchy is 0%.")
    else:
        # Self-consumed power at each timestep is the minimum of what was generated and what was loaded
        self_consumption_profile_kw = np.minimum(total_load_profile_kw, total_generation_profile_kw)
        
        # Calculate total self-consumed energy
        total_self_consumed_energy_kwh = self_consumption_profile_kw.sum() * time_interval_h
        
        # Autarchy = (Energy self-consumed / Total energy consumed)
        autarchy = total_self_consumed_energy_kwh / total_energy_consumed_kwh
        autarchy_percentage = autarchy * 100
        print(f"-> LEG Total Yearly Consumption: {total_energy_consumed_kwh:,.2f} kWh")
        print(f"-> LEG Self-Consumed Energy:   {total_self_consumed_energy_kwh:,.2f} kWh")
        
    print(f"✅ Finished Autarchy Calculation.")
    print(f"--> Autarchy of the LEG: {autarchy_percentage:.2f} %\n")
    
    # --- Return the final results ---
    final_results = {
        'grid_reinforcement_cost_chf': grid_cost,
        'installed_pv_kwp': installed_pv,
        'autarchy_percentage': autarchy_percentage
    }
    
    print("="*80)
    print("--- Trilemma Evaluation Complete ---")
    print(f"Grid Reinforcement Cost: {final_results['grid_reinforcement_cost_chf']:,.2f} CHF")
    print(f"LEG Installed PV:        {final_results['installed_pv_kwp']:.2f} kWp")
    print(f"LEG Autarchy:            {final_results['autarchy_percentage']:.2f} %")
    print("="*80)

    return final_results



# def evaluate_trilemma(
#     station_name: str,
#     leg_customer_ids: list,
#     df_full_topology: pd.DataFrame,
#     all_profiles: dict,
#     df_net_load_full_station: pd.DataFrame,
#     df_reinforcement_costs: pd.DataFrame,
#     nominal_voltage: float =400.0
# ):
#     """
#     Evaluates the energy trilemma for a given Local Energy Community (LEG).

#     The trilemma is assessed based on three key metrics:
#     1. Grid Reinforcement Cost: The cost to upgrade the shared grid infrastructure
#        of the station to handle the total load, including the LEG's contribution.
#     2. Installed PV Capacity: The total peak power (kWp) of all PV systems
#        within the specified LEG.
#     3. Autarchy: The self-sufficiency rate of the LEG, calculated as the ratio of
#        self-consumed energy to the total energy consumed by the LEG.

#     Args:
#         station_name (str): The name of the station the LEG belongs to.
#         leg_customer_ids (list): A list of customer IDs (e.g., 'HAS-xxxx') forming the LEG.
#         df_full_topology (pd.DataFrame): The complete network topology data (`df_full`).
#         all_profiles (dict): A dictionary containing all profile DataFrames
#                              (e.g., {"Consumption": df_consumption, "PV": df_pv, ...}).
#         df_net_load_full_station (pd.DataFrame): The pre-calculated net load for the entire station.
#         df_reinforcement_costs (pd.DataFrame): DataFrame with reinforcement costs.

#     Returns:
#         dict: A dictionary containing the trilemma results:
#               {'grid_reinforcement_cost_chf': float,
#                'installed_pv_kwp': float,
#                'autarchy_percentage': float}
#     """
#     print("\n" + "="*80)
#     print(f"--- Starting Trilemma Evaluation for LEG in Station: '{station_name}' ---")
#     print(f"LEG consists of {len(leg_customer_ids)} members.")
#     print("="*80 + "\n")
    
#     # ==========================================================================
#     # --- 1. Calculate Grid Reinforcement Cost for the Entire Station ---
#     # The cost depends on the total load of the station, as infrastructure is shared.
#     # ==========================================================================
#     print(f"--- [1/3] Evaluating Grid Reinforcement Cost for Station: '{station_name}' ---")
#     df_one_station = df_full_topology[df_full_topology['station'] == station_name].copy()
#     NOMINAL_VOLTAGE = nominal_voltage


#     # Build the network graph for the station
#     G, consumer_props, roots = build_and_simplify_network(df_one_station)
    
#     print("\nRunning dynamic profile analysis to find failures...")
#     dynamic_results = find_failures_with_yearly_profile(
#         graph=G,
#         net_profile_df=df_net_load_full_station,
#         consumer_props=consumer_props,
#         root_node_ids=roots,
#         nominal_voltage=NOMINAL_VOLTAGE
#     )
    
#     print("\nCalculating suggested grid reinforcements...")
#     reinforcement_results = suggest_grid_reinforcement(
#         initial_graph=G,
#         initial_results=dynamic_results,
#         reinforcement_costs_df=df_reinforcement_costs,
#         net_profile_df=df_net_load_full_station,
#         consumer_props=consumer_props,
#         root_node_ids=roots,
#         nominal_voltage=NOMINAL_VOLTAGE
#     )
    
#     grid_cost = reinforcement_results.get('total_cost_CHF', 0.0)
#     print(f"✅ Finished Grid Cost Calculation.")
#     print(f"--> Estimated Grid Reinforcement Cost: {grid_cost:.2f} CHF\n")

#     # ==========================================================================
#     # --- 2. Calculate Total Installed PV Capacity for the LEG ---
#     # ==========================================================================
#     print(f"--- [2/3] Calculating Installed PV Capacity for the LEG ---")
#     df_pv_profiles = all_profiles.get("PV")
    
#     # Filter for LEG customers that actually have a PV profile
#     leg_pv_customers = [cid for cid in leg_customer_ids if cid in df_pv_profiles.columns]
    
#     if not leg_pv_customers:
#         installed_pv = 0.0
#     else:
#         # Get the max value (peak power in kW) for each customer and sum them up
#         installed_pv = df_pv_profiles[leg_pv_customers].max().sum()
        
#     print(f"✅ Finished PV Capacity Calculation.")
#     print(f"--> Total Installed PV in LEG: {installed_pv:.2f} kWp\n")

#     # ==========================================================================
#     # --- 3. Calculate Autarchy (Self-Sufficiency) for the LEG ---
#     # ==========================================================================
#     print(f"--- [3/3] Calculating Autarchy for the LEG ---")

#     def get_leg_profile(profile_df, default_val=0):
#         """Helper to safely get and sum profiles for LEG members."""
#         if profile_df is None:
#             # This should not happen if all_profiles is complete, but it's a safe fallback.
#             return pd.Series(default_val, index=all_profiles["Consumption"].index)
        
#         # Find which of the LEG customers are in this dataframe's columns
#         leg_cols = [cid for cid in leg_customer_ids if cid in profile_df.columns]
        
#         if not leg_cols:
#             return pd.Series(default_val, index=profile_df.index)
            
#         return profile_df[leg_cols].sum(axis=1)

#     # Calculate total load and generation profiles specifically for the LEG
#     total_load_profile_kw = (get_leg_profile(all_profiles.get("Consumption")) +
#                              get_leg_profile(all_profiles.get("EV")) +
#                              get_leg_profile(all_profiles.get("HP")) +
#                              get_leg_profile(all_profiles.get("battery_in")))

#     total_generation_profile_kw = (get_leg_profile(all_profiles.get("PV")) +
#                                    get_leg_profile(all_profiles.get("battery_out")))

#     # Convert power (kW) to energy (kWh) by multiplying by the time interval in hours (15 min = 0.25 h)
#     time_interval_h = 0.25
#     total_energy_consumed_kwh = total_load_profile_kw.sum() * time_interval_h
    
#     if total_energy_consumed_kwh == 0:
#         autarchy_percentage = 0.0
#         print("-> Total energy consumption for the LEG is zero. Autarchy is 0%.")
#     else:
#         # Self-consumed power at each timestep is the minimum of what was generated and what was loaded
#         self_consumption_profile_kw = np.minimum(total_load_profile_kw, total_generation_profile_kw)
        
#         # Calculate total self-consumed energy
#         total_self_consumed_energy_kwh = self_consumption_profile_kw.sum() * time_interval_h
        
#         # Autarchy = (Energy self-consumed / Total energy consumed)
#         autarchy = total_self_consumed_energy_kwh / total_energy_consumed_kwh
#         autarchy_percentage = autarchy * 100
#         print(f"-> LEG Total Yearly Consumption: {total_energy_consumed_kwh:,.2f} kWh")
#         print(f"-> LEG Self-Consumed Energy:   {total_self_consumed_energy_kwh:,.2f} kWh")
        
#     print(f"✅ Finished Autarchy Calculation.")
#     print(f"--> Autarchy of the LEG: {autarchy_percentage:.2f} %\n")
    
#     # --- Return the final results ---
#     final_results = {
#         'grid_reinforcement_cost_chf': grid_cost,
#         'installed_pv_kwp': installed_pv,
#         'autarchy_percentage': autarchy_percentage
#     }
    
#     print("="*80)
#     print("--- Trilemma Evaluation Complete ---")
#     print(f"Grid Reinforcement Cost: {final_results['grid_reinforcement_cost_chf']:,.2f} CHF")
#     print(f"LEG Installed PV:        {final_results['installed_pv_kwp']:.2f} kWp")
#     print(f"LEG Autarchy:            {final_results['autarchy_percentage']:.2f} %")
#     print("="*80)

#     return final_results