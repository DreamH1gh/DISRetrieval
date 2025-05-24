from collections import Counter
import argparse
import string
import re
import json


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return (1.0, 1.0)
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(' '.join(ground_truth)).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (0, 0)
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, recall)
    # prediction = prediction.split('\n\n')

def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    if text_evidence_only:
                        evidence = [text for text in answer_info["highlighted_evidence"] if "FLOAT SELECTED" not in text]
                    else:
                        evidence = answer_info["highlighted_evidence"]
                    references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue
        answer_f1s_and_types = [
            (token_f1_score(predicted[question_id]["answer"], reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        evidence_f1s = [
            paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
            for reference in gold[question_id]
        ]
        # max_evidence_f1s.append(max(evidence_f1s))
        for idx, x in enumerate(evidence_f1s):
            if not isinstance(x, (tuple, list)):
                print(f"Bad element at index {idx}: {x} (type: {type(x)})")
        sorted_f1s = sorted(evidence_f1s, key=lambda x: x[0], reverse=True)
        max_evidence_f1s.append(sorted_f1s[0])

    mean = lambda x: sum(x) / len(x) if x else 0.0
    f1s = [x[0] for x in max_evidence_f1s]
    recalls = [x[1] for x in max_evidence_f1s]
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Evidence F1": mean(f1s),
        "Evidence_Recall": mean(recalls),
        "Missing predictions": num_missing_predictions
    }

def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate the text to a specified number of words while preserving the original whitespace.

    Args:
        text: The input text
        max_words: The target maximum number of words

    Returns:
        The truncated text
    """
    matches = list(re.finditer(r'\S+', text))
    if len(matches) <= max_words:
        return text
    
    cutoff_index = matches[max_words - 1].end()
    return text[:cutoff_index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        help="""JSON file of results""",
        default=""
    )
    parser.add_argument(
        "--gold",
        type=str,
        help="Test or dev set from the released dataset",
        default="evaluate/qasper_test.json"
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        default=True,
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1"
    )
    parser.add_argument(
        "--reform",
        action="store_true",
        default=False
    )
    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, args.text_evidence_only)
    predicted_answers_and_evidence = {}

    with open(args.predictions, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for prediction_data in data:
        answer = prediction_data["predicted_answer"] if prediction_data["predicted_answer"] != 'no answer>' else 'unanswerable'
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": answer,
            "evidence": prediction_data["predicted_evidence"]
        }
    evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence)
    print(json.dumps(evaluation_output, indent=2))

