# Clean And Shape Data for Analysis
# Output:Every row is a trial, saves to the analysis folder

# ============================================================================
# Imports
# ============================================================================
import pandas as pd

from stop_wm.config import ProjectConfig

# Initialize configuration
config = ProjectConfig()
PREPROCESSED_DATA_DIR = config.preprocessed_data_dir
ANALYSIS_DIR = config.results_dir

# ============================================================================
# Functions
# ============================================================================

def load_and_clean_df(df):
    """Clean and shape the dataframe by filtering and dropping columns."""
    df = df.query('exp_stage == "test"')

    # only include rows where trial_id is one of the following:
    # test_memory_trial, test_stop_trial, test_memory_recognition
    # add test_trial to the list of trial_ids, that is for the stop signal
    # task that DOESNT have a memory component
    df = df[df['trial_id'].isin([
        'test_memory_trial', 'test_stop_trial', 'test_memory_recognition',
        'test_trial'
    ])]

    # Drop columns that exist in the dataframe
    columns_to_drop = [
        'internal_node_id', 'view_history', 'attention_check_question',
        'success', 'trial_type', 'trial_index', 'time_elapsed',
        'stimulus', 'exp_stage', 'response_ends_trial', 'SS_stimulus',
        'choices', 'timing_post_trial', 'question', 'exp_id'
    ]
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def reshape_trial_data(df):
    """
    Alternative approach using pandas pivot functionality.
    This creates separate columns for each trial_id and metric combination.
    """
    # Apply the fix for probe trial indexing
    df_fixed = df.copy()
    df_fixed.loc[
        df_fixed['trial_id'] == 'test_memory_recognition', 'current_trial'
    ] -= 1

    # Filter to the three trial types
    trial_data = df_fixed[
        df_fixed['trial_id'].isin([
            'test_memory_trial',
            'test_stop_trial',
            'test_memory_recognition',
            'test_trial'
        ])
    ].copy()

    # Get columns to exclude from pivoting (identifiers that should be the same)
    id_cols = ['current_trial', 'block_num']

    # Melt the data to long format first, then pivot
    # Select only numeric/categorical columns that vary by trial type
    value_cols = [col for col in trial_data.columns
                  if col not in id_cols + ['trial_id'] and
                  trial_data[col].dtype in ['object', 'int64', 'float64', 'bool']]

    # Melt the data
    melted = trial_data.melt(
        id_vars=['current_trial', 'block_num', 'trial_id'],
        value_vars=value_cols,
        var_name='metric',
        value_name='value'
    )

    # Create new column names combining trial_id and metric
    melted['trial_metric'] = (
        melted['trial_id'].str.replace('test_', '') + '_' + melted['metric']
    )

    # Pivot to one row per trial format
    pivoted = melted.pivot_table(
        index=['block_num', 'current_trial'],
        columns='trial_metric',
        values='value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()

    # Flatten column names
    pivoted.columns.name = None

    return pivoted


# ============================================================================
# Main
# ============================================================================
def main():
    """Main function to clean and shape data for analysis."""
    # Dictionary to collect reshaped dataframes for each task
    task_dataframes = {}

    # Process each subject's folder individually
    for subject_dir in PREPROCESSED_DATA_DIR.iterdir():
        if subject_dir.is_dir():
            print(f"Processing subject: {subject_dir.name}")

            # Process each task directory for this subject
            for task_dir in subject_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name in ['race_ethnicity_RMR_survey_rdoc']:
                    continue  # Skip non-task directories
                
                task_name = task_dir.name
                print(f"  Processing task: {task_name}")

                # Find CSV files for this task
                csv_files = list(task_dir.glob('*.csv'))
                
                if not csv_files:
                    print(f"    No CSV files found for {task_name}")
                    continue

                # Use the first CSV file found for this task
                csv_file = csv_files[0]
                print(f"    Using file: {csv_file.name}")

                # Load and process the data
                participant_df = pd.read_csv(csv_file)
                cleaned_df = load_and_clean_df(participant_df)
                reshaped_df = reshape_trial_data(cleaned_df)

                # Add participant ID column
                reshaped_df['participant_id'] = subject_dir.name

                # Save the individual dataframe
                output_file = subject_dir / f"reshaped_{task_name}_data_{subject_dir.name}.csv"
                reshaped_df.to_csv(output_file, index=False)
                print(f"    Saved reshaped data to: {output_file}")

                # Add to task-specific collection
                if task_name not in task_dataframes:
                    task_dataframes[task_name] = []
                task_dataframes[task_name].append(reshaped_df)

    # Save combined dataframes for each task
    for task_name, task_dfs in task_dataframes.items():
        if task_dfs:
            combined_df = pd.concat(task_dfs, ignore_index=True)
            combined_output_file = ANALYSIS_DIR / f'all_participants_reshaped_data_{task_name}.csv'
            combined_df.to_csv(combined_output_file, index=False)
            print(f"Saved combined {task_name} data to: {combined_output_file}")
            print(f"Total participants processed for {task_name}: {len(task_dfs)}")
        else:
            print(f"No participants found for task: {task_name}")


if __name__ == "__main__":
    main()

