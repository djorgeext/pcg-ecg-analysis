# This code was developed by David Springer for comparison purposes in the
# paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
from sklearn.linear_model import LogisticRegression

def trainBandPiMatricesSpringer(state_observation_values):
    """
    Train the B matrix and pi vector for the Springer HMM.
    The pi vector is the initial state probability, while the B matrix are
    the observation probabilities. In the case of Springer's algorith, the
    observation probabilities are based on a logistic regression-based
    probabilities. 

    Inputs:
    state_observation_values: list of lists (or arrays) of size (N_recordings, 4)
    Each element is a numpy array of shape (K, J) where K is samples and J is features.

    Outputs:
    B_matrix: List of arrays containing [Intercept, Coeffs...] for each state, 
              mimicking MATLAB's mnrfit output (log odds of Other vs Target).
    pi_vector: Initial state probabilities.
    total_obs_distribution: List [mean, covariance] of the total observation sequence.
    """

    number_of_states = 4

    # Set pi_vector
    # The true value of the pi vector, which are the initial state
    # probabilities, are dependant on the heart rate of each PCG, and the
    # individual sound duration for each patient. Therefore, instead of setting
    # a patient-dependant pi_vector, simplify by setting all states as equally
    # probable:
    pi_vector = [0.25, 0.25, 0.25, 0.25]

    # Train the logistic regression-based B_matrix:

    B_matrix = [None] * number_of_states
    
    # Organize data by state
    # state_observation_values is expected to be iterable of length N_recordings
    # each item is iterable of length 4 (states)
    
    # Initialize lists to hold arrays
    statei_values = [[] for _ in range(number_of_states)]
    
    for recording in state_observation_values:
        for state_idx in range(number_of_states):
            # Check if empty
            if recording[state_idx] is not None and len(recording[state_idx]) > 0:
                statei_values[state_idx].append(recording[state_idx])
    
    # Concatenate to form single array per state
    for state_idx in range(number_of_states):
        if len(statei_values[state_idx]) > 0:
            statei_values[state_idx] = np.vstack(statei_values[state_idx])
        else:
            statei_values[state_idx] = np.empty((0, 0)) # Should handle gracefully later?

    # In order to use Bayes' formula with the logistic regression derived
    # probabilities, we need to get the probability of seeing a specific
    # observation in the total training data set. This is the
    # 'total_observation_sequence', and the mean and covariance for each state
    # is found:
    
    valid_states = [s for s in statei_values if len(s) > 0]
    if not valid_states:
         total_observation_sequence = np.array([])
         # Should probably error out
    else:
        total_observation_sequence = np.vstack(valid_states)
    
    mean_vec = np.mean(total_observation_sequence, axis=0)
    cov_mat = np.cov(total_observation_sequence, rowvar=False)
    
    total_obs_distribution = [mean_vec, cov_mat]

    for state in range(number_of_states):
        # Python 0-based index 'state' corresponds to MATLAB 1-based 'state'
        
        # Randomly select indices of samples from the other states not being 
        # learnt, in order to balance the two data sets.
        
        length_of_state_samples = len(statei_values[state])
        
        # Number of samples required from each of the other states:
        length_per_other_state = int(np.floor(length_of_state_samples / (number_of_states - 1)))
        
        # If the length of the main class / (num states - 1) >
        # length(shortest other class), then only select
        # length(shortect other class) from the other states,
        # and (3* length) for main class
        
        min_length_other_class = np.inf
        
        for other_state in range(number_of_states):
            samples_in_other_state = len(statei_values[other_state])
            
            if other_state != state:
                min_length_other_class = min(min_length_other_class, samples_in_other_state)
        
        # This means there aren't enough samples in one of the
        # states to match the length of the main class being
        # trained:
        if length_per_other_state > min_length_other_class:
            length_per_other_state = int(min_length_other_class)
            
        training_data_target = None
        training_data_others = []
        
        for other_state in range(number_of_states):
            samples_in_other_state = len(statei_values[other_state])
            
            if other_state == state:
                # Make sure you only choose (n-1)*3 *
                # length_per_other_state samples for the main
                # state, to ensure that the sets are balanced:
                
                # Note: The MATLAB comment says "(n-1)*3" but the code says:
                # indices = randperm(samples_in_other_state,length_per_other_state*(number_of_states-1));
                # num_states - 1 = 3. So it selects 3 * length_per_other_state.
                # Yes, that matches "evenly split across all other classes" logic from loop above.
                
                num_samples_target = length_per_other_state * (number_of_states - 1)
                
                indices = np.random.choice(samples_in_other_state, num_samples_target, replace=False)
                training_data_target = statei_values[other_state][indices, :]
                
            else:
                indices = np.random.choice(samples_in_other_state, length_per_other_state, replace=False)
                state_data = statei_values[other_state][indices, :]
                training_data_others.append(state_data)
        
        training_data_others_stacked = np.vstack(training_data_others)
        
        # Label all the data:
        # labels = ones(length(training_data{1}) + length(training_data{2}),1);
        # labels(1:length(training_data{1})) = 2;
        
        # MATLAB: training_data{1} is Target (label 2). training_data{2} is Others (label 1).
        
        labels_target = np.full(len(training_data_target), 2)
        labels_others = np.full(len(training_data_others_stacked), 1)
        
        all_data = np.vstack([training_data_target, training_data_others_stacked])
        all_labels = np.concatenate([labels_target, labels_others])
        
        # Train the logisitic regression model for this state:
        # [B,~,~] = mnrfit(all_data,labels);
        # MATLAB mnrfit with labels [1, 2] uses 2 as reference. Estimates log(P(1)/P(2)).
        # P(1) = Other, P(2) = Target.
        # So it estimates log(P(Other)/P(Target)).
        
        # Sklearn LogisticRegression
        # fit(X, y). Classes will be [1, 2].
        # It predicts log(P(2)/P(1)) if we treat 1 as class 0 and 2 as class 1?
        # Sklearn stores coef for the "positive" class wrt the first class in 'classes_'.
        # classes_ = [1, 2]. "Positive" is 2.
        # Sklearn calculates log(P(Y=2)/P(Y=1)).
        # This is log(P(Target)/P(Other)).
        # This is - log(P(Other)/P(Target)).
        # So B_matlab = - B_sklearn.
        
        # Use simple logistic regression with minimal regularization to match MATLAB's mnrfit (MLE)
        # MATLAB mnrfit does not use regularization. Sklearn defaults to L2 regularization (C=1.0).
        # We set C to a very large value to emulate no regularization.
        clf = LogisticRegression(solver='lbfgs', fit_intercept=True, C=1e10, max_iter=1000)
        clf.fit(all_data, all_labels)
        
        # Reconstruct B
        # MATLAB B includes intercept as first element.
        intercept = clf.intercept_[0]
        coefs = clf.coef_[0]
        
        # Negate to match mnrfit's log(Other/Target) vs sklearn's log(Target/Other)
        b_sklearn = np.concatenate([[intercept], coefs])
        b_matlab = - b_sklearn
        
        B_matrix[state] = b_matlab

    return B_matrix, pi_vector, total_obs_distribution
