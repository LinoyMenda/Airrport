import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import celltypist
import pandas as pd
from typing import TextIO
# Import AnnData for cell type annotation
from anndata import AnnData
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import singlecellexperiment as sce
import singler
import celldex

def setup_logging(logfile: str) -> TextIO:
    """
    Redirects stdout and stderr to a log file.
    
    Args:
        logfile: The path to the log file.
        
    Returns:
        The file handle for the opened log file.
    """
    print(f"Redirecting output to log file: {logfile}")
    log_file_handle = open(logfile, "w")
    sys.stdout = log_file_handle
    sys.stderr = log_file_handle
    print("Starting Scanpy pipeline...")
    return log_file_handle

def close_logging(log_file_handle: TextIO):
    """
    Closes the log file and restores stdout/stderr.
    
    Args:
        log_file_handle: The file handle returned by setup_logging.
    """
    print("Analysis pipeline completed successfully.")
    sys.stdout.close()
    # Restore standard output
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Log file created. Check 'scanpy_analysis.log' for details.")

def load_data(data_dir: str, airrport_path: str, vdj_path: str, igblast_path: str) -> tuple[AnnData, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the 10x Genomics data and renames barcodes.
    Loads AIRRPORT dat (parquet file)
    Loads Igblast data and merge with AIRRPORT data by seq_id 
    Load vdj-seq data (refernce)
    """
    print(f"\n--- Loading single cells Data ---")
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    # replace -1 from barcodes with empthy string
    adata.obs_names = adata.obs_names.str.replace('-1', '', regex=False)
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes.")

    
    ### AIRRPORT results ###
    print(f"\n--- Loading AIRRPORT data from {airrport_path} ---")
    df_airrport = pd.read_parquet(airrport_path)
    # Clean Cell Barcode (CB)
    df_airrport['CB'] = df_airrport['CB'].astype(str).str.replace('Z:', '', regex=False)
    df_airrport['CB'] = df_airrport['CB'].str.replace('*', '-1', regex=False)
    df_airrport['CB'] = df_airrport['CB'].str.replace('-1', '', regex=False)

    ### VDJ (CDR3 sequencing for reference) ###
    print(f"\n--- Loading VDJ data from {vdj_path} ---")
    df_vdj = pd.read_csv(vdj_path)
    # Add “vdj” before each column name
    df_vdj = df_vdj.add_prefix('vdj_')
    df_vdj['vdj_barcode'] = df_vdj['vdj_barcode'].str.replace("-1$", "", regex=True)

    ### IgBlast for AIRRPORT results ###
    print(f"\n--- Loading IgBlast data from {igblast_path} ---")
    df_airrport_igblast = pd.read_csv(igblast_path, sep='\t')
    # Add “IgBlast” before each column name
    df_airrport_igblast = df_airrport_igblast.add_prefix('IgBlast_')
    
    # Select summary columns
    df_airrport_igblast_summary = df_airrport_igblast[[
        "IgBlast_sequence_id", "IgBlast_sequence_aa", "IgBlast_sequence_alignment_aa",
        "IgBlast_cdr3", "IgBlast_cdr3_aa", "IgBlast_v_support", "IgBlast_j_support",
        "IgBlast_v_identity", "IgBlast_j_identity"
    ]]

    ### Left join to IgBlast with AIRRPORT ###
    print(f"\n--- Joining IgBlast and AIRRPORT data... ---")
    df_airrport_igblast_unified = pd.merge(
        df_airrport_igblast_summary,
        df_airrport,
        left_on="IgBlast_sequence_id",
        right_on="seq_id",
        how="left"
    )

    ### Filter by conditions in order to remove noise ###
    print(f"\n--- Filtering results based on strict conditions... ---")
    cdr3_strict_conditions = df_airrport_igblast_unified[
        (df_airrport_igblast_unified['IgBlast_v_support'] < 1e-05) &
        (df_airrport_igblast_unified['IgBlast_v_identity'] >= 90) &
        (df_airrport_igblast_unified['IgBlast_j_support'] < 1e-05) &
        (df_airrport_igblast_unified['IgBlast_j_identity'] >= 90)
    ]
    
    print(f"Found {len(cdr3_strict_conditions)} high-confidence (strict conditions of IgBlast) AIRR-seq entries.")


    return adata, df_airrport, df_vdj, df_airrport_igblast, cdr3_strict_conditions, df_airrport_igblast_unified

def analysis_logs(adata, df_airrport, df_vdj, df_airrport_igblast, cdr3_strict_conditions, classifier_table):
    print(f"\n--- AIRRPORT Data Analysis ---")
    if "CDR3_match" in df_airrport.columns:
        unique_airrport_cdr3_count = df_airrport['CDR3_match'].nunique()
        print(f"Unique values in df_airrport['CDR3_match']: {unique_airrport_cdr3_count}")
        airrport_cdr3_set = set(df_airrport['CDR3_match'].dropna())
    else:
        print("Column 'CDR3_match' not found in df_airrport. Skipping count.")
        airrport_cdr3_set = set()

    print(f"\n--- VDJ Data Analysis ---")
    if "vdj_cdr3" in df_vdj.columns:
        unique_vdj_cdr3_count = df_vdj['vdj_cdr3'].nunique()
        print(f"Unique values in df_vdj['vdj_cdr3']: {unique_vdj_cdr3_count}")
        vdj_cdr3_set = set(df_vdj['vdj_cdr3'].dropna())
    else:
        print("Column 'vdj_cdr3' not found in df_vdj. Skipping count.")
        vdj_cdr3_set = set()

    print(f"\n--- Common Sequence Analysis ---")
    if "CDR3_match" in df_airrport.columns and "vdj_cdr3" in df_vdj.columns:
        common_sequences = airrport_cdr3_set.intersection(vdj_cdr3_set)
        print(f"Unique CDR3 sequences from AIRRPORT that are also present in the VDJ reference: {len(common_sequences)}")
    else:
        print("Skipping common sequence count due to missing column(s).")
        
    # --- NEW SECTION: Classifier Table Analysis ---
    print(f"\n--- Classifier Table Analysis ---")
    if classifier_table is not None and not classifier_table.empty:
        # 1. Unique CDR3 count in classifier_table
        if 'CDR3_match' in classifier_table.columns:
            unique_cdr3_classifier = classifier_table['CDR3_match'].nunique()
            print(f"Unique CDR3 sequences in classifier_table: {unique_cdr3_classifier}")
        else:
             print(f"Warning: 'CDR3_match' not found in classifier_table.")

        # 2. Unique CB count for those sequences
        # This is just the total unique CBs in the table, as every row has a CB.
        if 'CB' in classifier_table.columns:
            unique_cb_classifier = classifier_table['CB'].nunique()
            print(f"Unique Cell Barcodes (CB) in classifier_table: {unique_cb_classifier}")
        else:
             print(f"Warning: 'CB' not found in classifier_table.")

        # 3. Count of sequences that are BOTH 'in_vdj' AND 'T cells'
        # We need to check if the required columns exist first.
        required_cols = ['label', 'CDR3_match'] # We can use the 'label' column we created earlier
        if all(col in classifier_table.columns for col in required_cols):
             # The 'label' column already holds (is_t_cell & is_in_vdj)
             # We want unique CDR3s that have at least one True label.
             true_label_cdr3s = classifier_table.loc[classifier_table['label'] == True, 'CDR3_match'].nunique()
             print(f"Unique CDR3s that are both 'in_vdj' and 'T cell': {true_label_cdr3s}")

        # 4. How many cdr3 there is for each cell type?
        if 'cell_type' in classifier_table.columns and 'CDR3_match' in classifier_table.columns:
            print("Unique CDR3 count per cell type:")
            # Count unique CDR3s per cell type
            cdr3_per_cell_type = classifier_table.groupby('cell_type')['CDR3_match'].nunique()
            cdr3_per_cell_type = cdr3_per_cell_type.sort_values(ascending=False)

            # Create horizontal bar plot
            plt.figure(figsize=(8, 6))
            ax = cdr3_per_cell_type.plot(kind='barh', color='skyblue')

                # Add labels on each bar
            for bar in ax.patches:
                width = bar.get_width()
                plt.text(width + 0.5,              # x position slightly past the bar
                 bar.get_y() + bar.get_height()/2, # y position at the center of the bar
                 str(int(width)),                  # label text
                 va='center',                      # vertical alignment
                 fontsize=10)

            plt.xlabel("Unique CDR3s")
            plt.ylabel("Cell type")
            plt.title("Unique CDR3 count per cell type")
            plt.gca().invert_yaxis()  # Highest count on top

            # Save figure to file
            plt.tight_layout()  # adjust layout
            # Create folder if it doesn't exist
            os.makedirs("figures", exist_ok=True)
            plt.savefig("figures/cdr3_per_cell_type.png", dpi=300)  # can also use .pdf or .svg
            plt.close()  # close the figure to free memory
        else:
            print("\nError: Columns 'CDR3_match' or 'cell_type' not found, skipping CDR3 count per cell type.")

    else:
        print("classifier_table is empty or None.")


def preprocess_sc_data(adata: AnnData) -> AnnData:
    """
    Runs standard preprocessing (filtering, normalization, HVGs).
    
    Args:
        adata: The AnnData object to preprocess.
        
    Returns:
        The preprocessed AnnData object.
    """
    print("--- Standard Preprocessing for Single cells ---")
    
    # Basic filtering (adjust thresholds as needed for your specific tissue)
    # This removes any cell that detected fewer than 200 genes.
    # Why? Cells with very few detected genes are often dead cells, empty droplets
    # (identifying background noise rather than a real cell), or failed library preparation.
    sc.pp.filter_cells(adata, min_genes=200)
    # This removes any gene that was detected in fewer than 3 cells across the entire dataset.
    # Why? deeply rarely expressed genes (e.g., found in only 1 or 2 cells out of thousands) provide little statistical power for clustering or differential expression and increase computational noise.
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"Data shape after filtering: {adata.n_obs} cells x {adata.n_vars} genes")

    # Mitochondrial gene filtering
    # High mitochondrial DNA (mtDNA) percentage is a classic sign of a stressed or dying cell, and these are usually removed from analysis.
    adata.var['mito_genes'] = adata.var_names.str.startswith('MT-')  # MT- for human
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito_genes'], percent_top=None, log1p=False, inplace=True)

    # Filter cells based on mitochondrial content (e.g., < 10% or < 20%)
    # Adjust this threshold based on your data's distribution (plot with sc.pl.violin)
    print(f"Cells before mitochondrial gene filtering: {adata.n_obs}")
    adata = adata[adata.obs.pct_counts_mito_genes < 20, :]
    print(f"Cells after mitochondrial gene filtering: {adata.n_obs}")

    # Normalization and Scaling
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Identify highly variable genes for dimensionality reduction
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Make a copy of raw data for later use (e.g., CellTypist prefers it)
    # This MUST be done AFTER normalization/log1p but BEFORE subsetting and scaling.
    adata.raw = adata 
    
    # 2. Subset to highly variable genes
    # This subsets the main adata object for clustering.
    adata = adata[:, adata.var.highly_variable]
    print(f"Subsetting to {adata.n_vars} highly variable genes.")

    # 3. Scale data to unit variance and 0 mean.
    # This is done LAST, only on the highly variable genes.
    sc.pp.scale(adata, max_value=10) 
    
    return adata

def cluster_and_embed(adata: AnnData) -> AnnData:
    """
    Runs PCA, neighborhood graph, UMAP, and Leiden clustering.
    
    Args:
        adata: The preprocessed AnnData object.
        
    Returns:
        The clustered and embedded AnnData object.
    """
    print("\n--- Clustering and Embedding ---")
    # PCA and Neighborhood graph
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    #Clustering and Embedding
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5) # Adjust resolution for more/fewer clusters
    print("Saving clustering plots...")
    sc.pl.umap(adata, color=['leiden'], save="_leiden_clusters.png", show=False)
    sc.pl.pca(adata, color='leiden', save="_pca_clusters.png", show=False)
    return adata

def annotate_cell_types(adata: AnnData, model_name: str) -> AnnData:
    """
    Annotates cell types using CellTypist.
    
    Args:
        adata: The clustered AnnData object.
        model_name: The name of the CellTypist model to use.
        
    Returns:
        The AnnData object with 'predicted_cell_type' in .obs.
    """
    print("--- Cell Type Annotation (CellTypist) ---")
       
    # Download model (e.g., for immune cells)
    print(f"Downloading/loading model: {model_name}")
    model = celltypist.models.download_models(model=model_name)
    
    # Predict cell types
    # Note: CellTypist automatically uses adata.raw if available, which is good practice.
    print("Running cell type prediction...")
    predictions = celltypist.annotate(adata, model=model, majority_voting=True)
    
    # Add predictions to AnnData object
    # We use the 'majority_voting' result for cleaner clusters
    adata.obs['predicted_cell_type'] = predictions.predicted_labels['majority_voting']

    print("Saving CellTypist annotation plot...")
    sc.pl.umap(adata, color='predicted_cell_type', save="_celltypist_annotation.png", show=False)


    # Optional: Save a CSV of just the barcodes and their new cell types for easy viewing later
    print("Saving annotation CSV...")
    adata.obs[['leiden', 'predicted_cell_type']].to_csv("cell_type_annotations.csv")

    ### singleR
    sce_adata = sce.SingleCellExperiment.from_anndata(adata) 
    ref_data = celldex.fetch_reference("blueprint_encode", "2024-02-26", realize_assays=True)

    return adata


def classifier_table(
    adata: AnnData,
    df_airrport_igblast_unified: pd.DataFrame,
    df_vdj: pd.DataFrame,
    publicness_file_path: str,
    cell_type_col: str = 'predicted_labels'
) -> pd.DataFrame:
    """
    Builds the classifier table by joining AIRR, IgBlast, publicness,
    gene expression, and cell type data.
    
    Args:
        df_airrport: DataFrame from AIRRPORT (e.g., df_airrport_SRX10124718)
        df_igblast: DataFrame from IgBlast (e.g., df_airrport_igblast_..._unified)
        adata: Your preprocessed AnnData object
        publicness_file_path: String path to the _ppub_counts.csv file
        cell_type_col: Name of the column in adata.obs with cell types
        
    Returns:
        A new DataFrame 'classifier_table'
    """
    print("\n--- Building Classifier Table ---")

    # --- Read and join publicness ---
    print("Read publicness data")
    publicness_df = pd.read_csv(publicness_file_path)
    print(publicness_df.head())

    publicness_subset = (
        publicness_df[['CDR3_match', 'publicness_score']]
        .drop_duplicates(subset=['CDR3_match'], keep='first')
    )
    
    classifier_table = pd.merge(
        df_airrport_igblast_unified,
        publicness_subset,
        how='left',
        left_on='CDR3_match',
        right_on='CDR3_match'
    )

    print("Add gene expression matrix to classifier table")
    expr_df = adata.to_df()
    expr_df = expr_df.reset_index().rename(columns={'index': 'CB'})
    classifier_table = pd.merge(
        classifier_table,
        expr_df,
        how='left',
        on='CB'
    )

    #  Add Cell Type from annotation
    print("Adding cell types")
    if cell_type_col in adata.obs.columns:
        # Create a map with cleaned barcodes
        cell_type_series = adata.obs[cell_type_col]
        cell_type_map = cell_type_series.to_dict()
        
        # Map cell types using the cleaned CBs
        classifier_table['cell_type'] = classifier_table['CB'].map(cell_type_map)
    else:
        print(f"Warning: Cell type column '{cell_type_col}' not found in adata.obs. Skipping.")
        classifier_table['cell_type'] = None

    print("Concatenate CDR3_match and CB, then check whether the resulting sequence exists in the VDJ reference --> new columns: is_in_vdj")
    # concatenated of CDR3_match and CB in vdj reference
    vdj_keys = set(
        df_vdj['vdj_cdr3'].astype(str) + "_" + df_vdj['vdj_barcode'].astype(str)
    )

    # Create the new concatenated key column in classifier_table.
    classifier_table['temp_key'] = (
        classifier_table['CDR3_match'].astype(str) + "_" + classifier_table['CB'].astype(str)
    )

    # Use .isin() to check if each key is in the vdj_keys set.
    classifier_table['is_in_vdj'] = classifier_table['temp_key'].isin(vdj_keys)

    # Remove the temporary key column
    classifier_table = classifier_table.drop(columns=['temp_key'])

    print("Create new columns: label (return TRUE if is_t_cell & is_in_vdj)")
    classifier_table['cell_type'] = classifier_table['cell_type'].fillna('')
    is_t_cell = classifier_table['cell_type'].str.contains("T cells", case=False)
    is_in_vdj = classifier_table['is_in_vdj'] == True
    classifier_table['label'] = is_t_cell & is_in_vdj

    print("\nDone! Final table shape:", classifier_table.shape)
    print("Write classifier table head to csv file")
    classifier_table.head(5).to_csv("classifier_head.csv", index=False)


    return classifier_table

def build_model(classifier_table):
    print("\n--- Building Models for ML ---")

    print("Label distribution:")
    print(classifier_table['label'].value_counts())

    # --- STEP 1: DEFINE FEATURES & LABEL ---
    print("Automatically selecting numeric features:")
    exclude_cols = classifier_table.select_dtypes(include='object').columns.tolist()
    exclude_cols += ['label', 'is_in_vdj']
    X = classifier_table.drop(columns=exclude_cols, errors='ignore').copy()
    y = classifier_table['label']

    print(f"Number of samples: {X.shape[0]}")
    print(f"Original numeric features: {X.shape[1]}")

    # --- STEP 2: HANDLE MISSING DATA (Imputation) ---
    print("Imputing missing data (NaNs)...")
    # FIX 1: Ensure we capture only the columns kept by the imputer
    imputer = SimpleImputer(strategy='median')
    # Optional: if you have sklearn >= 1.2, you can use: imputer.set_output(transform="pandas")
    X_imputed = imputer.fit_transform(X)
    
    # Get the feature names that survived imputation
    if hasattr(imputer, "get_feature_names_out"):
        feature_names_imputed = imputer.get_feature_names_out(input_features=X.columns)
    else:
        # Fallback for older sklearn versions if needed, though less likely in your env
        feature_names_imputed = X.columns 

    X = pd.DataFrame(X_imputed, columns=feature_names_imputed)
    print(f"Features after imputation: {X.shape[1]} (dropped {len(exclude_cols) - X.shape[1]} all-NaN columns if any)")

    # --- STEP 3: FEATURE SELECTION ---
    print("\n--- Feature Selection ---")
    
    # a) Remove zero-variance features
    var_selector = VarianceThreshold(threshold=0.0)
    X_var = var_selector.fit_transform(X)
    kept_features_var = X.columns[var_selector.get_support()]
    print(f"Kept {len(kept_features_var)} features after variance filtering")

    # b) Select important features using RandomForest
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_var, y)

    selector_model = SelectFromModel(rf_selector, threshold="mean", prefit=True)
    X_selected = selector_model.transform(X_var)
    kept_features_model = kept_features_var[selector_model.get_support()]
    print(f"Selected {len(kept_features_model)} features with importance above mean")

    X = pd.DataFrame(X_selected, columns=kept_features_model)
    print(f"Final feature matrix shape: {X.shape}")

    # --- STEP 4: TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- STEP 5: SCALE FEATURES ---
    scaler = StandardScaler()
    # X_train_scaled and X_test_scaled are now numpy arrays
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- STEP 6: DEFINE MODELS ---
    scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()
    print(f"Using scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }

    # --- STEP 7: TRAIN, EVALUATE, OVERSAMPLE & DOWNSAMPLE ---
    for name, model in models.items():
        print(f"\n--- Training {name} (Base) ---")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        print(f"Classification Report ({name} - Base):")
        print(classification_report(y_test, y_pred, target_names=['Not T-Cell', 'Is T-Cell']))

        # --- Feature importances ---
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Since X_train_scaled is numpy, we use the column names from X_train (the DF)
            feature_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            print("\nTop 10 important features:")
            print(feature_df.head(10))

        # --- Oversampling ---
        print(f"\n--- Training {name} (Oversampled) ---")
        ros = RandomOverSampler(random_state=42)
        # FIX 2: Use X_train_scaled instead of X_train so data remains scaled
        X_over, y_over = ros.fit_resample(X_train_scaled, y_train) 
        print("Class distribution after oversampling:")
        print(y_over.value_counts())
        
        model.fit(X_over, y_over)
        y_pred_over = model.predict(X_test_scaled)
        print(f"F1 score ({name} - Oversampled): {f1_score(y_test, y_pred_over):.3f}")

        # --- Oversampling with SMOTE ---
        print(f"\n--- Training {name} (SMOTE Oversampled) ---")
        # Initialize SMOTE. random_state ensures reproducibility.
        smote = SMOTE(random_state=42)
        
        # Apply SMOTE to the scaled training data
        X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
        
        print("Class distribution after SMOTE:")
        print(y_smote.value_counts())
        
        # Train model on SMOTE-augmented data
        model.fit(X_smote, y_smote)
        y_pred_smote = model.predict(X_test_scaled)
        
        print(f"F1 score ({name} - SMOTE): {f1_score(y_test, y_pred_smote):.3f}")
        print(f"Classification Report ({name} - SMOTE):")
        print(classification_report(y_test, y_pred_smote, target_names=['Not T-Cell', 'Is T-Cell']))

        # --- Downsampling ---
        print(f"\n--- Training {name} (Downsampled) ---")
        rus = RandomUnderSampler(random_state=42)
        # FIX 2: Use X_train_scaled here too
        X_under, y_under = rus.fit_resample(X_train_scaled, y_train)
        print("Class distribution after downsampling:")
        print(y_under.value_counts())
        
        model.fit(X_under, y_under)
        y_pred_under = model.predict(X_test_scaled)
        print(f"F1 score ({name} - Downsampled): {f1_score(y_test, y_pred_under):.3f}")

    print("\n--- Pipeline completed successfully ---")
             
def main():
    """
    Main function to run the entire pipeline.
    """
    # --- Configuration ---
    '''
    SC_DATA_DIR = '/dsi/efroni-lab/AIRRPORT/test_env/sc/GEX/runs/SRX10124718_gex/outs/filtered_feature_bc_matrix/'
    AIRRPORT_PATH = '/home/ls/linoym/r_files/airrport/SRX10124718_sample/matched_SRX10124718_unaligned_reads_plusCBUB_trimmed_R2_collapsed.parquet'
    VDJ_PATH = '/home/ls/linoym/r_files/airrport/SRX10124718_sample/P4_LNM_vdj_filtered_contig_annotations.csv'
    IGBLAST_PATH = '/home/ls/linoym/r_files/airrport/SRX10124718_sample/igblast_SRX10124718.tsv'
    '''
    SC_DATA_DIR = '/dsi/efroni-lab/AIRRPORT/test_env/sc/GEX/SRX10124709/SRX10124709_gex/outs/filtered_feature_bc_matrix'
    AIRRPORT_PATH = '/dsi/efroni-lab/AIRRPORT/singlecell_breast/results/Tumor/airrport/matched_SRX10124711_unaligned_reads_plusCBUB_trimmed_R2.parquet'
    VDJ_PATH = '/dsi/efroni-lab/AIRRPORT/singlecell_breast/results/Tumor/vdj/P5_tumor_vdj_filtered_contig_annotations.csv'
    IGBLAST_PATH = '/dsi/efroni-lab/AIRRPORT/singlecell_breast/results/Tumor/igblast/igblast_SRX10124711.tsv'
    
    LOG_FILE = "scanpy_analysis.log"
    CELLTYPIST_MODEL = 'Immune_All_Low.pkl'
    FINAL_ADATA_FILE = "sc_rna_seq_processed.h5ad"
    PUBLICNESS_PATH = "/home/ls/linoym/r_files/airrport/SRX10124718_ppub_counts.csv"

    log_handle = setup_logging(LOG_FILE)
    
    try:
        # Run Pipeline
        sc_rna_seq, df_airrport, df_vdj, df_airrport_igblast, cdr3_strict_conditions, df_airrport_igblast_unified = load_data(
            SC_DATA_DIR,
            AIRRPORT_PATH,
            VDJ_PATH,
            IGBLAST_PATH
        )
        # Preprocessing
        sc_rna_seq = preprocess_sc_data(sc_rna_seq)
        # Dimensionality Reduction & Clustering
        sc_rna_seq = cluster_and_embed(sc_rna_seq)
        # Annotate
        sc_rna_seq = annotate_cell_types(sc_rna_seq, CELLTYPIST_MODEL)
        # Final Save of the entire object
        sc_rna_seq.write(FINAL_ADATA_FILE)
        classifier_data = classifier_table(sc_rna_seq,df_airrport_igblast_unified,df_vdj ,PUBLICNESS_PATH,cell_type_col = 'predicted_cell_type')
        analysis_logs(sc_rna_seq, df_airrport, df_vdj, df_airrport_igblast, cdr3_strict_conditions, classifier_data)
        build_model(classifier_data)

    except Exception as e:
        print(f"--- ERROR: Pipeline failed ---", file=sys.stderr)
        print(str(e), file=sys.stderr)
        # Re-raise the exception after logging it
        raise
    finally:
        # This block will run NO MATTER WHAT (success or error),
        # ensuring your log file is always closed properly.
        close_logging(log_handle)

if __name__ == "__main__":
    main()