# function to load sequences and enhancer activity
def prepare_input(set):
    # Convert sequences to one-hot encoding matrix
    file_seq = str("Sequences_" + set + ".fa")
    input_fasta_data_A = IOHelper.get_fastas_from_file(file_seq, uppercase=True)

    # get length of first sequence
    sequence_length = len(input_fasta_data_A.sequence.iloc[0])

    # Convert sequence to one hot encoding matrix
    seq_matrix_A = SequenceHelper.do_one_hot_encoding(input_fasta_data_A.sequence, sequence_length,
                                                      SequenceHelper.parse_alpha_to_seq)
    print(seq_matrix_A.shape)
    
    X = np.nan_to_num(seq_matrix_A) # Replace NaN with zero and infinity with large finite numbers
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    Activity = pd.read_table("Sequences_activity_" + set + ".txt")
    Y_dev = Activity.Dev_log2_enrichment
    Y_hk = Activity.Hk_log2_enrichment
    Y = [Y_dev, Y_hk]
    
    print(set)

    return input_fasta_data_A.sequence, seq_matrix_A, X_reshaped, Y