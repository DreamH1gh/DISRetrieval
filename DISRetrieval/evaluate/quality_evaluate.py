import json
import fire
import Levenshtein
import re

def check_by_distance(pred, options):
    best = -1
    min_distance = 10000000
    for i, op in enumerate(options):
        distance = Levenshtein.distance(pred.lower(), op.lower())
        if distance < min_distance:
            best = i + 1
            min_distance = distance
    assert best > 0
    return best

def check_quality(path, model):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)

    total = 0
    acc = 0 
    for item in data:
        total += 1
        options = item['options']
        pred = item['pred'] if 'pred' in item.keys() else item['predicted_answer']
        pred = pred.lower()

        pred_id = 0

        ### evaluate unifiedQA results
        if model == 'unifiedQA':
            pred_id = check_by_distance(pred, options)
        elif model == 'llm':
            ## evaluate llm results
            op = ['(a)', '(b)', "(c)", '(d)']
            pattern = r"\(([abcd])\)"
            match = re.search(pattern, pred)
            if match:
                letter = match.group(1)
                option_str = f"({letter})" 
                if option_str in op:
                    pred_id = op.index(option_str) + 1
        else:
            print(f'{model} error')


        label = item['label'] if 'label' in item.keys() else item['answer']
        if pred_id == label:
            acc += 1
    print(acc * 100 / total)

if __name__ == "__main__":
    fire.Fire(check_quality)