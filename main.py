import numpy as np
import pickle


def main ():
	STRUCTURES_DICT = "structures_dictionary.pickle"
	AMINO_ACID_DICT = "amino_acids_dictionary.pickle"
	TRAINING_SEQUENCES = "training_sequences.pickle"
	TRAINING_STRUCTURES = "training_structures.pickle"
	structures_dict = open_file(STRUCTURES_DICT)
	amino_acid_dict = open_file(AMINO_ACID_DICT)
	sequences = open_file(TRAINING_SEQUENCES)
	labels = open_file(TRAINING_STRUCTURES)
	num_sequences = len(sequences)
	state_vector = [2,2,2,2,2,2,2,2]
	tpm = generate_transition_probability_matrix(state_vector)
	epm = generate_emission_probability_matrix(state_vector, amino_acid_dict)
	lpm = generate_label_emission_matrix(state_vector, structures_dict)
	ip = generate_initial_probabilities(state_vector)
	tpm, epm = em(sequences, labels, tpm, epm, lpm, ip, amino_acid_dict, structures_dict, num_iterations=1)


# Opens pickle file and returns stored data object
def open_file(file_name):
	with open(file_name, 'rb') as handle:
		return pickle.load(handle)


# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)     


# Generates a random transition probability  matrix
# state_vector contains the number of states for each type of secondary structure
def generate_transition_probability_matrix(state_vector):
	num_states = sum(state_vector)
	trans_prob_matrix = np.random.rand(num_states, num_states)
	trans_prob_matrix = normalize_tpm(trans_prob_matrix)
	return trans_prob_matrix


def generate_initial_probabilities(state_vector):
	num_states = sum(state_vector)
	ip = np.ones(num_states)
	ip = ip / num_states
	ip = np.log(ip)
	return ip


def generate_emission_probability_matrix(state_vector, amino_acid_dict):
	num_states = sum(state_vector)
	num_emissions = len(amino_acid_dict)
	emiss_prob_matrix = np.random.rand(num_states,num_emissions)
	return emiss_prob_matrix


def generate_label_emission_matrix(state_vector, sec_str_dict):
	num_states = sum(state_vector)
	num_labels = len(sec_str_dict)
	label_emiss_prob_matrix = np.zeros((num_states, num_labels))
	current_label = 0
	current_state = 0
	for i in range(0,num_labels):
		num_label_states = state_vector[i]
		for j in range(0, num_label_states):
			label_emiss_prob_matrix[current_state][current_label] = 1
			current_state += 1
		current_label += 1
	return label_emiss_prob_matrix


''' Helper function: Computes the sum of two event probabilities in log space
Arguments:
	a: log probability of first event
	b: log probability of second event
Returns:
	sum: sum of events a and b in log space
'''
def log_space_sum(a,b):
	sum_values = 0
	if a < b:
		sum_values = b + np.log(1+np.exp(a-b))
	else:
		sum_values = a + np.log(1+np.exp(b-a))
	return sum_values


''' Helper function: Computes a log space sum over a vector of probabilities
Arguments:
	a: vector of log probabilities
Returns:
	sum: log space sum over the vector of probabilities
'''
def log_space_summation(a):
	num_elements = len(a)
	assert num_elements > 1
	sum_vector = log_space_sum(a[0], a[1])
	if num_elements > 2:
		for i in range(2,num_elements):
			sum_vector = log_space_sum(sum_vector, a[i])
	return sum_vector


''' Helper function: Computes an elementwise log space sum for corresponding elements in matrices A and B
'''
def log_space_matrix_sum(A, B):
	assert A.shape == B.shape
	(nrows, ncols) = A.shape
	for i in range(0,nrows):
		for j in range(0,ncols):
			a = A[i][j]
			b = B[i][j]
			a = log_space_sum(a,b)
			A[i][j] = a
	return A


''' Helper function: Outputs forward probabilities of a given observation
Arguments:
	obs: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities, formatted in the reverse order
					 of the posterior probability (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	F: matrix of forward probabilities
	likelihood_f: P(obs) calculated using the forward algorithm
'''
def forward(obs, trans_probs, emiss_probs, init_probs, amino_acid_dict):
	num_emissions = len(obs) + 1
	num_states = len(init_probs)
	F = np.zeros((num_states,num_emissions))
	# Populate F with initial probabilities
	for i in range(0, num_states):
		F[i][0] = init_probs[i]
	for i in range(1,num_emissions):
		emission = obs[i-1]
		emission_idx = amino_acid_dict[emission]
		for j in range(0,num_states):
			emiss_prob = emiss_probs[j][emission_idx]
			prob = 0
			trans_prob_vector = trans_probs[:,j]
			F_vector = F[:,i-1]
			prob = log_space_summation(np.multiply(trans_prob_vector,F_vector))
			F[j][i] = prob + emiss_prob
	likelihood_f = log_space_summation(F[:,num_emissions-1])
	return F, likelihood_f


''' Helper function: Outputs backward probabilities of a given observation
Arguments:
	obs: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities, formatted in the reverse order
					 of the posterior probability (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	B: matrix of backward probabilities
	likelihood_b: P(obs) calculated using the backward algorithm
'''
def backward(obs, trans_probs, emiss_probs, init_probs, amino_acid_dict):
	num_emissions = len(obs) + 1
	num_states = len(init_probs)
	B = np.zeros((num_states,num_emissions))
	# initialize B with final probabilities
	for i in range(0, num_states):
		B[i][num_emissions-1] = 1
	for i in range(num_emissions-3,-1,-1):
		emission = obs[i+1]
		emission_idx = amino_acid_dict[emission]
		for j in range(0,num_states):
			emiss_prob_vector = emiss_probs[:,emission_idx]
			prob = 0
			B_vector = B[:,i+1]
			trans_prob_vector = trans_probs[j,:]
			summation_vector = np.multiply(trans_prob_vector, emiss_prob_vector)
			summation_vector = np.multiply(summation_vector, B_vector)
			prob = log_space_summation(summation_vector)
			B[j][i] = prob
	emission = obs[0]
	emission_idx = amino_acid_dict[emission]
	emiss_prob_vector = emiss_probs[:,emission_idx]
	summation_vector = np.multiply(emiss_prob_vector, init_probs)
	summation_vector = np.multiply(summation_vector, B[:,0])
	likelihood_b = log_space_summation(summation_vector)
	return B, likelihood_b


''' Helper function: Compute normalization constant
Arguments:
	R: matrix of posterior probabilities that have not yet been normalized
Returns:
	R: normalized matrix of posterior probabilities
'''
def normalize(R):
	Z = 0
	(_,n) = R.shape
	for i in range(0,n):
		Z = log_space_sum(R[0][i], R[1][i])
		R[0][i] = np.exp(R[0][i] - Z)
		R[1][i] = np.exp(R[1][i] - Z)
	return R


def normalize_log_vector(vec):
	denominator = log_space_summation(vec)
	vec = vec - denominator
	return vec

''' Helper function: Normalizes a transition probability matrix
'''
def normalize_tpm(tpm):
	tpm = tpm + 1 # smooth the data to remove 0 probabilities
	num_states = len(tpm)
	for i in range(0, num_states):
		denominator = sum(tpm[i,:])
		tpm[i,:] = tpm[i,:] / denominator
	tpm = np.log(tpm)
	return tpm


def normalize_log_prob_tpm(tpm):
	num_states = len(tpm)
	for i in range(0,num_states):
		row = tpm[i,:]
		denominator = log_space_summation(row)
		tpm[i,:] = row - denominator
	return tpm

''' Helper function: Normalizes an emission probability matrix of probablities
'''
def normalize_epm(epm):
	epm = epm + 1 # smooth the data to remove 0 probabilities
	(num_states, _) = epm.shape
	for i in range(0, num_states):
		denominator = sum(epm[i,:])
		epm[i,:] = epm[i,:] / denominator
	epm = np.log(epm)
	return epm


''' Outputs the forward and backward probabilities of a given observation.
Arguments:
	obs: observed sequence of emitted states (list of emissions)
	trans_probs: transition log probabilities (dictionary of dictionaries)
	emiss_probs: emission log probabilities, formatted in the reverse order
					 of the posterior probability (dictionary of dictionaries)
	init_probs: initial log probabilities for each hidden state (dictionary)
Returns:
	F: matrix of forward probabilities
		likelihood_f: P(obs) calculated using the forward algorithm
	B: matrix of backward probabilities
		likelihood_b: P(obs) calculated using the backward algorithm
	R: matrix of posterior probabilities
'''
def forward_backward(obs, trans_probs, emiss_probs, init_probs, amino_acid_dict):
	F, likelihood_f = forward(obs, trans_probs, emiss_probs, init_probs, amino_acid_dict)
	B, likelihood_b = backward(obs, trans_probs, emiss_probs, init_probs, amino_acid_dict)
	(_,n) = F.shape
	R = F[:,1:n] + B[:,0:n-1]
	R = normalize(R)
	return F[:,1:n], likelihood_f, B[:,0:n-1], likelihood_b, R


''' Helper function: Computes new transition probability matrix
'''
def update_trans_and_emiss_probs(sequences, labels, tp, ep, lp, ip, amino_acid_dict, structures_dict):
	num_samples = len(sequences)
	new_tp = np.zeros(tp.shape)
	new_ep = np.zeros(ep.shape)
	for n in range(0,num_samples):
	# for n in range(0,1):
		print(n)
		sequence = sequences[n]
		label = labels[n]
		seq_length = len(sequence)
		F, likelihood, B, _, _ = forward_backward(sequence, tp, ep, ip, amino_acid_dict)
		for i in range(0,seq_length):
			if i < seq_length-1:
				#update transition probabilities
				emission = sequence[i+1]
				label_emission = label[i+1]
				tp_residual = np.zeros(tp.shape)
				aa_index = amino_acid_dict[emission]
				ss_index = structures_dict[label_emission]
				F_vector = F[:,i]
				B_vector = B[:,i+1]
				e_vector = ep[:,aa_index]
				l_vector = lp[:,ss_index]
				# log-normalize first before exponentiating
				tp_residual = normalize_log_prob_tpm(tp_residual + F_vector[:, np.newaxis] + B_vector[np.newaxis, :] + e_vector[:, np.newaxis] - likelihood)
				tp_residual = np.exp(tp_residual)
				tp_residual = tp_residual*l_vector[:, np.newaxis]
				if i == 0:
					new_tp = tp_residual
				else:
					new_tp = log_space_matrix_sum(new_tp, tp_residual)
			#update emission probabilities 
			emission2 = sequence[i]
			label_emission2 = label[i]
			#log-normalize before exponentiating
			emiss_residual = normalize_log_vector(F[:,i] + B[:,i] - likelihood)
			emiss_residual = np.exp(emiss_residual)
			aa_index = amino_acid_dict[emission2]
			ss_index = structures_dict[label_emission2]
			l_vector = lp[:,ss_index]
			new_ep[:, aa_index] = (new_ep[:, aa_index] + emiss_residual) * l_vector
	new_tp = normalize_tpm(new_tp) # convert back to log space here
	new_ep = normalize_epm(new_ep)  # convert back to log space here
	return new_tp, new_ep


''' Performs EM on CHMM
Arguments:
	obs: the set of sequences under analysis
	tp: transition probabilities in log space
	ep: emission probabilities in log space
	ip: initalization probabilities in log space
Returns:
	tp: updated transition probabilities, in log space
	ep: updated emission probabilities, in log space
'''
def em(sequences, labels, tp, ep, lp, ip, amino_acid_dict, structures_dict, num_iterations = 100):
	for i in range(0,num_iterations):
		tp, ep = update_trans_and_emiss_probs(sequences, labels, tp, ep, lp, ip, amino_acid_dict, structures_dict)
	return tp, ep


if __name__ == "__main__":
	main()