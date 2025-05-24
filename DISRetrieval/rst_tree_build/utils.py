import json
from pathlib import Path

def ensure_path_exits(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def read_sent(inf):
    sentence = []
    for line in inf:
        line = line.strip()
        if line == '':
            yield sentence
            sentence = []
        else:
            sentence.append(line)
    if len(sentence) > 0:
        yield sentence

def parse_bracket_tree(bracket_string):
    stack = []
    current_node = None
    index = 0
    left_child = True
    while index < len(bracket_string):
        char = bracket_string[index]
        if char == '(':
            new_node = {"label": "", "children": []}
            if current_node:
                stack.append((current_node, left_child))
                current_node["children"].append(new_node)
            current_node = new_node
            left_child = True
            index += 1
        elif char == ')':
            if stack:
                current_node, left_child = stack.pop()
            index += 1
            left_child = False
        elif char.isspace():
            index += 1
        else:
            start = index
            while index < len(bracket_string) and not bracket_string[index].isspace() and bracket_string[index] not in '()':
                index += 1
            current_node["label"] += bracket_string[start:index] if current_node['label'] == "" else " " + bracket_string[start:index]
    return current_node