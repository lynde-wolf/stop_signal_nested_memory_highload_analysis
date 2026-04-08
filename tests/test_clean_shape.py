import pandas as pd

from src.stop_wm.clean_shape import is_flagged, load_and_clean_df, reshape_trial_data


def test_load_and_clean_df():
    """Test the load_and_clean_df function with sample data."""
    # Create sample test data
    test_data = {
        'exp_stage': ['test', 'test', 'practice', 'test', 'test'] * 20,
        'trial_id': [
            'test_memory_trial', 'test_stop_trial', 'test_memory_recognition',
            'test_trial', 'test_memory_trial'
        ] * 20,
        'current_trial': list(range(100)),
        'rt': [500, 600, 700, 800, 900] * 20,
        'correct': [True, False, True, True, False] * 20,
        'condition': ['go', 'stop', 'go', 'stop', 'go'] * 20,
        'stimLength': [2, 4, 2, 4, 2] * 20,
        'wm_load': [0, 2, 4, 0, 2] * 20,
        'wm0': [True, False, True, False, True] * 20,
        'wm2': [False, True, False, True, False] * 20,
        'wm4': [False, False, True, False, False] * 20,
        'internal_node_id': ['1', '2', '3', '4', '5'] * 20,
        'view_history': ['a', 'b', 'c', 'd', 'e'] * 20,
        'attention_check_question': ['q1', 'q2', 'q3', 'q4', 'q5'] * 20,
        'success': [True, False, True, True, False] * 20,
        'trial_type': ['memory', 'stop', 'recognition', 'trial', 'memory'] * 20,
        'trial_index': list(range(100)),
        'time_elapsed': [1000, 2000, 3000, 4000, 5000] * 20,
        'stimulus': ['stim1', 'stim2', 'stim3', 'stim4', 'stim5'] * 20,
        'response_ends_trial': [True, False, True, True, False] * 20,
        'SS_stimulus': ['ss1', 'ss2', 'ss3', 'ss4', 'ss5'] * 20,
        'choices': ['choice1', 'choice2', 'choice3', 'choice4', 'choice5'] * 20,
        'timing_post_trial': [100, 200, 300, 400, 500] * 20,
        'question': ['q1', 'q2', 'q3', 'q4', 'q5'] * 20,
        'exp_id': ['exp1', 'exp2', 'exp3', 'exp4', 'exp5'] * 20
    }
    
    df = pd.DataFrame(test_data)
    cleaned_df = load_and_clean_df(df)

    # Check that only test stage data is kept
    assert cleaned_df.shape[0] == 80  # 4 out of 5 rows per group are 'test' stage
    assert 'exp_stage' not in cleaned_df.columns
    assert 'internal_node_id' not in cleaned_df.columns
    assert 'view_history' not in cleaned_df.columns
    assert 'attention_check_question' not in cleaned_df.columns
    assert 'success' not in cleaned_df.columns
    assert 'trial_type' not in cleaned_df.columns
    assert 'trial_index' not in cleaned_df.columns
    assert 'time_elapsed' not in cleaned_df.columns
    assert 'stimulus' not in cleaned_df.columns
    assert 'response_ends_trial' not in cleaned_df.columns
    assert 'SS_stimulus' not in cleaned_df.columns
    assert 'choices' not in cleaned_df.columns
    assert 'timing_post_trial' not in cleaned_df.columns
    assert 'question' not in cleaned_df.columns
    assert 'exp_id' not in cleaned_df.columns


def test_reshape_trial_data():
    """Test the reshape_trial_data function with sample data."""
    # Create sample test data
    test_data = {
        'current_trial': [0, 0, 1, 1, 2, 2],
        'trial_id': [
            'test_memory_trial', 'test_stop_trial', 'test_memory_trial',
            'test_stop_trial', 'test_memory_trial', 'test_stop_trial'
        ],
        'rt': [500, 600, 700, 800, 900, 1000],
        'correct': [True, False, True, True, False, True],
        'condition': ['go', 'stop', 'go', 'stop', 'go', 'stop'],
        'stimLength': [2, 4, 2, 4, 2, 4],
        'wm_load': [0, 2, 0, 2, 0, 2],
        'wm0': [True, False, True, False, True, False],
        'wm2': [False, True, False, True, False, True],
        'wm4': [False, False, False, False, False, False]
    }
    
    df = pd.DataFrame(test_data)
    reshaped_df = reshape_trial_data(df)

    # Check that the data is properly reshaped
    assert 'current_trial' in reshaped_df.columns
    assert reshaped_df.shape[0] == 3  # 3 unique current_trial values
    assert reshaped_df.shape[1] > 10  # Should have many columns after reshaping


def test_is_flagged():
    """Test the is_flagged function with sample data."""
    # Test case where participant should be flagged (low accuracy)
    flagged_metrics = {
        'prolific_id': 'test123',
        'completion_date': '1730250295511',
        'go_accuracy': 0.2,  # Below minimum threshold of 0.55
        'stop_accuracy': 0.3,  # Within range
        'go_rt': 600,  # Within range
        'go_omission_rate': 0.3  # Within range
    }
    
    assert is_flagged(flagged_metrics, 'stop_signal')

    # Test case where participant should not be flagged
    good_metrics = {
        'prolific_id': 'test456',
        'completion_date': '1730250295511',
        'go_accuracy': 0.8,  # Above minimum threshold
        'stop_accuracy': 0.5,  # Within range
        'go_rt': 600,  # Within range
        'go_omission_rate': 0.2  # Within range
    }
    
    assert not is_flagged(good_metrics, 'stop_signal')

    # Test case with missing task (should not be flagged)
    assert not is_flagged(good_metrics, 'nonexistent_task')
