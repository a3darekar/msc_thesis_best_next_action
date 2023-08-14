from collections import defaultdict

deal_status = {'lost': 0, 'won': 1}
deal_status_types = {v: k for k, v in deal_status.items()}

def get_encode_activity_types(df):
    return {v: k for k, v in get_activity_types(df).items()}

def get_activity_types(df):
    return dict(enumerate(df.activity_type.unique(), 1))

def add_outcome_to_df(group):
    entry = group.tail(1)
    entry.timestamp = entry.timestamp
    entry.activity_type = entry.deal_status_string
    entry.activity_seq += 1
    entry.activity_active_days = 0
    entry.activity_count += 1
    return pd.concat([group, entry], ignore_index=True)


def get_class_report_top_n(n,probs):    
    top_n_res = []
    for i, x in enumerate(probs):
        idx = (-x).argsort()[:n]
        if (y_true[i] in idx):
            top_n_res.append([y_true[i], y_true[i]])
        else:
            top_n_res.append([y_true[i], idx[0]])

    top_n_eval = pd.DataFrame(top_n_res, columns=['true','pred'])
    return classification_report(top_n_eval.true, top_n_eval.pred)


def prepare_kpi_lookup_table(df):
    seq, outcomes = df['activity_type_list'], df['deal_status']

    # Count the success ratio for each sequence in the dataset
    success_ratios = defaultdict(lambda: [0, 0])
    for sequence, success in zip(seq, outcomes):
        success_ratios[tuple(sequence)][0] += success
        success_ratios[tuple(sequence)][1] += 1
    kpi_probability_table = {sequence: success[0] / success[1] for sequence, success in success_ratios.items()}

    return kpi_probability_table

# Define the beam search function
def beam_search(model, activity_seqs, kpi_probability_table, beam_width=3, predict_next_n=10, use_kpi=False, unseen_score= 0):
    min_length = len(activity_seqs) + predict_next_n - 3
    alpha = 0.5
    # Initialize the search beam starting point
    beam = [{'sequence': [], 'score': 0.3}]
    if not use_kpi:
        beam = [{'sequence': activity_seqs, 'score': 0.3}]
    else:
        beam = [{'sequence': activity_seqs, 'score': get_score(kpi_probability_table, activity_seqs, unseen_score)}]
    topk = []
    topk.append(beam[0])
    # Loop over the sequence length
    for t in range(predict_next_n):
        # Create a list to store the new candidate sequences
        candidates = []
        
        # Loop over the beam candidates
        for candidate in beam:
            seq, score = candidate['sequence'], candidate['score']
            # Use the model to score the next activity
            prediction = predict_sequence(seq, model)[0]
            # Loop over the top-k predicted activities
            for i in np.argsort(prediction)[-beam_width:]:
                new_seq = np.append(seq, i)
                if use_kpi:
                    success_prob = get_score(kpi_probability_table, new_seq, unseen_score)
                    new_score = (prediction[i]**alpha) * (success_prob**(1-alpha))
                else:
                    new_score = prediction[i]
                new_candidate = {
                    'sequence': new_seq,
                    'score': new_score
                }
                candidates.append(new_candidate)

        # Sort the candidate list by score and select the top-k candidates
        beam = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
        topk += beam
    topk = sorted(topk, key=lambda x: x['score'], reverse=True)
    topk = list(filter(lambda x: len(x["sequence"]) >= min_length, topk))
    
    # Return the top-scoring sequence
    return topk[0]['sequence'], topk[0]['score']
