# for preprocessing protein secondary structure data
import pickle
import numpy as np

def main ():
	text_file = "ss.txt"
	num_samples = 10256
	seqs, secstrs = read_text_file(text_file, num_samples)
	full_sequence_indices, _ = select_full_sequences(seqs)
	train_seqs = [seqs[i] for i in full_sequence_indices]
	train_secstrs = [secstrs[i] for i in full_sequence_indices]
	num_seqs = len(train_seqs)
	# amino_acids_dict = create_dictionary(train_seqs)
	# sec_strs_dict = create_dictionary(train_secstrs)
	# print(amino_acids_dict)
	# print(sec_strs_dict)
	
	# TRAINING_SEQUENCES = "training_sequences.pickle"
	# TRAINING_STRUCTURES = "training_structures.pickle"
	# AMINO_ACIDS_DICTIONARY = "amino_acids_dictionary.pickle"
	# STRUCTURES_DICTIONARY = "structures_dictionary.pickle"
	
	# save_file(TRAINING_SEQUENCES, train_seqs)
	# save_file(TRAINING_STRUCTURES, train_secstrs)
	# save_file(AMINO_ACIDS_DICTIONARY, amino_acids_dict)
	# save_file(STRUCTURES_DICTIONARY, sec_strs_dict)

# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)     


# Opens pickle file and returns stored data object
def open_file(file_name):
	with open(file_name, 'rb') as handle:
		return pickle.load(handle)


def read_text_file(file_name, data_size):
	sequences = [None]*data_size
	secondary_structure = [None]*data_size
	with open(file_name, 'r') as myfile:
		list_of_data = myfile.read().splitlines()
	i = 0
	n = 0
	while i < len(list_of_data) and n < data_size:
		if ":sequence" in list_of_data[i]:
			i += 1
			while ":secstr" not in list_of_data[i]:
				if sequences[n] == None:
					sequences[n] = list_of_data[i]
				else:
					sequences[n] = sequences[n] + list_of_data[i]
				i += 1
		elif ":secstr" in list_of_data[i]:
			i += 1
			while ":sequence" not in list_of_data[i]:
				if secondary_structure[n] == None:
					secondary_structure[n] = list_of_data[i]
				else:
					secondary_structure[n] = secondary_structure[n] + list_of_data[i]
				i += 1
			n += 1
	return sequences, secondary_structure


def select_full_sequences(sequences):
	num_sequences = len(sequences)
	num_full_sequences = 0
	full_sequence_indices = []
	for i in range(0,num_sequences):
		if "X" not in sequences[i] and "U" not in sequences[i]:
			num_full_sequences += 1
			full_sequence_indices.append(i)
	return full_sequence_indices, num_full_sequences


def create_dictionary(sequences):
	num_sequences = len(sequences)
	char_dict = {}
	n = 0
	for i in range(0,num_sequences):
		sequence = sequences[i]
		seq_length = len(sequence)
		for j in range(0,seq_length):
			char = sequence[j]
			if char not in char_dict.keys():
				char_dict[char] = n
				n += 1
	return char_dict

if __name__ == "__main__":
	main()