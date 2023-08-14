import sys
import model, utils, preprocess
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def main(input_path='BPI Challenge 2017.xes'):
    print('reading dataset from xes')
    df = preprocess.xes_to_dataframe(input_path)
    df, num_activity_types, max_seq_length = preprocess.collect_sequence_traces(df)
    
    encode_activity_types = utils.get_encode_activity_types(df)

    df['case_outcome_string'] = df.case_outcome.apply(lambda x: utils.case_outcome_types[x])

    df = df.groupby('case_id').apply(lambda x: utils.add_outcome_to_df(x)).reset_index(drop=True)
    df['le_activity_type'] = df.activity_type.apply(lambda x: encode_activity_types[x])

    num_activity_types += 1

    # prepare train and test sets

    days_train, days_test, acts_train, acts_test, y_train, y_test, _ = preprocess.prepare_sequences(df)
    X_train, y_train, X_test, y_test = preprocess.split_dataset(days_train, days_test, acts_train, acts_test, y_train, y_test, num_activity_types, max_seq_length)

    # train activity recommender model 

    recommender_model = model.build(max_seq_length, num_activity_types)

    model.train(recommender_model, X_train, y_train)

    loss, score = recommender_model.evaluate(X_test, y_test)
    print("Loss: %.2f%% Accuracy: %.2f%%" % (loss, score*100))

    pred_probs = recommender_model.predict(X_test)
    pred_class = np.argmax(pred_probs, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    activity_types = utils.get_activity_types(df)

    print(classification_report(y_true, pred_class))

    print("\x1b[31m\"Report on Top-2 Next Activity\"\x1b[0m")
    print(utils.get_class_report_top_n(2,pred_probs))


    predictions = model.predict(model=recommender_model, X=X_test)

    # recommender_model.save("bpi_baseline_model")

    activity_types = utils.get_activity_types(df)
    activity_ids = {v: k for k, v in activity_types.items()}

    # Collect tail sequences
    test_df = df.groupby('case_id').last()[['activity_type_list', 'case_outcome', 'activity_count']]
    test_df['activity_type_list'] = test_df['activity_type_list'].apply(lambda x: x[:-1])

    # test_df.to_csv('test_bpi_df.csv')

    print('evaluating model without KPI lookup support')
    for beam_width in [1, 2, 3, 5, 7]:
        new_sequences, original_sequences, new_preds, improvements, true_outcomes = [], [], [], [], []
        for idx, row in tqdm(test_df.iterrows()):
            old_seq = row['activity_type_list']
            original_sequences.append(old_seq)
            prediction_len = len(old_seq) - len(old_seq[:starter_length])
            recommended_sequence, predicted_score = beam_search(recommender_model, old_seq[:starter_length], kpi_probability_table, beam_width=beam_width, predict_next_n=prediction_len, use_kpi=False)

            improvements.append(abs(predicted_score - kpi_probability_table.get(tuple(old_seq), 0)))
            new_sequences.append(recommended_sequence)
            new_preds.append(kpi_probability_table.get(tuple(recommended_sequence), 0))
            true_outcomes.append(row['case_outcome'])

        new_preds = np.array(new_preds)
        true_outcomes = np.array(true_outcomes)
        predictions = np.where(new_preds>0.75, 1, 0)
        print(f"For beam width - {beam_width} (without kpi probability optimized setting):")
        print(f"Average improvement predicted: ", np.mean(improvements))

        print("accuracy_score: ", accuracy_score(true_outcomes, predictions))
        print("precision_score: ", precision_score(true_outcomes, predictions))
            
        print("recall_score: ", recall_score(true_outcomes, predictions))
        print("f1_score: ", f1_score(true_outcomes, predictions))
        fpr, tpr, _ = roc_curve(true_outcomes, predictions)

        print("AUC-ROC: {:.4f}\n\n\n".format(auc(fpr, tpr)))
  
    print('evaluating model with KPI lookup support')
    for beam_width in [1, 2, 3, 5, 7]:
        new_sequences, original_sequences, new_preds, improvements, true_outcomes = [], [], [], [], []
        for idx, row in tqdm(test_df.iterrows()):
            old_seq = row['activity_type_list']
            original_sequences.append(old_seq)
            prediction_len = len(old_seq) - len(old_seq[:starter_length])
            recommended_sequence, predicted_score = beam_search(recommender_model, old_seq[:starter_length], kpi_probability_table, beam_width=beam_width, predict_next_n=prediction_len, use_kpi=True)

            improvements.append(abs(predicted_score - kpi_probability_table.get(tuple(old_seq), 0)))
            new_sequences.append(recommended_sequence)
            new_preds.append(kpi_probability_table.get(tuple(recommended_sequence), 0))
            true_outcomes.append(row['case_outcome'])

        new_preds = np.array(new_preds)
        true_outcomes = np.array(true_outcomes)
        predictions = np.where(new_preds>0.75, 1, 0)
        print(f"For beam width - {beam_width} (kpi probability optimized setting):")
        print(f"Average improvement predicted: ", np.mean(improvements))

        print("accuracy_score: ", accuracy_score(true_outcomes, predictions))
        print("precision_score: ", precision_score(true_outcomes, predictions))
            
        print("recall_score: ", recall_score(true_outcomes, predictions))
        print("f1_score: ", f1_score(true_outcomes, predictions))
        fpr, tpr, _ = roc_curve(true_outcomes, predictions)

        print("AUC-ROC: {:.4f}\n\n\n".format(auc(fpr, tpr)))


if __name__ == "__main__":
    print(sys.argv)
    input_path = None
    if len(sys.argv) > 1:
        log_name = sys.argv[1]
        input_path = os.path.join(os.getcwd(), 'data', log_name)

        print('input_path:', input_path)
    main(input_path)
