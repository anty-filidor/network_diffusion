#!/usr/bin/env python3
"""Convert docstrings and docs text to idiomatic British English.

Usage:
  scripts/britishise_docstrings.py --dry-run
  scripts/britishise_docstrings.py --apply

It processes Python files in `network_diffusion/` and ReStructuredText files in `docs/source/`.
"""

import argparse
import ast
import pathlib
import re
import sys

# US -> UK map
REPLACEMENTS = {
    "color": "colour",
    "colors": "colours",
    "colouring": "colouring",
    "coloring": "colouring",
    "center": "centre",
    "centers": "centres",
    "defense": "defence",
    "defenses": "defences",
    "meter": "metre",
    "meters": "metres",
    "pseudorandom": "pseudo-random",
    "randomize": "randomise",
    "randomized": "randomised",
    "randomizing": "randomising",
    "maximize": "maximise",
    "maximized": "maximised",
    "maximizing": "maximising",
    "minimize": "minimise",
    "minimized": "minimised",
    "minimizing": "minimising",
    "analyze": "analyse",
    "analyzed": "analysed",
    "analyzing": "analysing",
    "analyzer": "analyser",
    "prioritize": "prioritise",
    "prioritized": "prioritised",
    "prioritizing": "prioritising",
    "optimize": "optimise",
    "optimized": "optimised",
    "optimizing": "optimising",
    "optimizer": "optimiser",
    "initialization": "initialisation",
    "initialize": "initialise",
    "initialized": "initialised",
    "initializing": "initialising",
    "modeling": "modelling",
    "modelled": "modelled",
    "traveler": "traveller",
    "travelers": "travellers",
    "liters": "litres",
    "favorite": "favourite",
    "favorites": "favourites",
    "utilize": "utilise",
    "utilized": "utilised",
    "utilizing": "utilising",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "neighborhood": "neighbourhood",
    "neighbor": "neighbour",
    "neighbors": "neighbours",
    "license": "licence",
    "licenses": "licences",
    # common typos and slight grammatical errors found in text
    "colledded": "collected",
    "proprely": "properly",
    "actitor": "actor",
    "fist": "first",
    "choosen": "chosen",
    "alghorithm": "algorithm",
}

WORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(REPLACEMENTS.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)

PHRASE_REPLACEMENTS = [
    (re.compile(r"\bCheck if\b", re.IGNORECASE), "Check whether"),
    (re.compile(r"\bVerify if\b", re.IGNORECASE), "Verify whether"),
    (re.compile(r"\bGet actors randomly\b", re.IGNORECASE), "Retrieve actors at random"),
    (re.compile(r"\bGet the community with the lowest/highest value \(whatever it is\)\b", re.IGNORECASE), "Get the community with the lowest or highest value"),
    (re.compile(r"\bGet.*randomly\b", re.IGNORECASE), lambda m: m.group(0).replace("Get", "Retrieve", 1).replace("randomly", "at random")),
    (re.compile(r"\bGet network layers where actor exists\b", re.IGNORECASE), "Retrieve network layers in which an actor exists"),
    (re.compile(r"\bGet actor's states for where actitor exists\b", re.IGNORECASE), "Retrieve an actor's states where the actor exists"),
    (re.compile(r"\bReturn length of the network, i\.e\. num of actors\b", re.IGNORECASE), "Return the length of the network, i.e. the number of actors"),
    (re.compile(r"\bGet number of actors that live in the network\b", re.IGNORECASE), "Get the number of actors in the network"),
    (re.compile(r"\bGet number of nodes that live in each layer of the network\b", re.IGNORECASE), "Get the number of nodes in each layer of the network"),
    (re.compile(r"\bCreate nodewise ranking\b", re.IGNORECASE), "Create a per-node ranking"),
    (re.compile(r"\bReturn value of mean centrality for actors\b", re.IGNORECASE), "Return the mean centrality value for actors"),
    (re.compile(r"\bA script where (.*) is defined\b", re.IGNORECASE), r"A script defining \1"),
    (re.compile(r"\bContainer for a (.*)\b", re.IGNORECASE), r"A container for \1"),
    (re.compile(r"\bGet .* for the given .*\b", re.IGNORECASE), lambda m: m.group(0).replace("Get", "Retrieve",1)),
]


def britishise_word(match):
    word = match.group(0)
    key_lower = word.lower()
    replacement = REPLACEMENTS.get(key_lower, word)
    if word.isupper():
        return replacement.upper()
    if word[0].isupper():
        return replacement.capitalize()
    return replacement


def britishise_text(text):
    if not text:
        return text

    def apply_phrase_replacements(piece):
        updated = piece
        for pat, repl in PHRASE_REPLACEMENTS:
            if callable(repl):
                updated = pat.sub(repl, updated)
            else:
                updated = pat.sub(repl, updated)
        return updated

    # split by inline code markers: `...` and ``...`` for Sphinx
    parts = re.split(r"(``.*?``|`.*?`)", text, flags=re.DOTALL)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        else:
            piece = WORD_RE.sub(britishise_word, part)
            piece = apply_phrase_replacements(piece)
            out.append(piece)
    return "".join(out)


def extract_literal_text(src):
    m = re.match(r"^(?P<prefix>[rubfRUBF]*)(?P<quote>['\"]{3})(?P<content>.*)(?P=quote)$", src, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Unsupported literal form {src!r}")
    return m.group('prefix'), m.group('quote'), m.group('content')


def replace_docstring_literal(snippet, new_text):
    prefix, quote, old_text = extract_literal_text(snippet)
    if quote not in ('"""', "'''"):
        raise ValueError("Only triple-quoted literals are supported for patching")
    safe = new_text.replace(quote, '\\' + quote)
    return f"{prefix}{quote}{safe}{quote}"


def process_python_file(path, dry_run=True):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    docstring_nodes = []

    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        docstring_nodes.append(tree.body[0])

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.body and isinstance(child.body[0], ast.Expr) and isinstance(child.body[0].value, ast.Constant) and isinstance(child.body[0].value.value, str):
                    docstring_nodes.append(child.body[0])
            visit(child)

    visit(tree)

    if not docstring_nodes:
        return 0, []

    lines = src.splitlines(keepends=True)
    total_changed = 0
    report = []
    edits = []

    for node in docstring_nodes:
        start = (node.lineno - 1, node.col_offset)
        end = (node.end_lineno - 1, node.end_col_offset)
        if start[0] == end[0]:
            snippet = lines[start[0]][start[1]:end[1]]
        else:
            snippet = lines[start[0]][start[1]:] + ''.join(lines[start[0] + 1:end[0]]) + lines[end[0]][:end[1]]

        try:
            original_value = ast.literal_eval(snippet)
        except SyntaxError:
            continue

        updated_value = britishise_text(original_value)
        if updated_value != original_value:
            try:
                replaced = replace_docstring_literal(snippet, updated_value)
            except ValueError:
                continue
            edits.append((start, end, replaced))
            total_changed += 1
            report.append((path, original_value, updated_value))

    if total_changed and not dry_run:
        for start, end, replacement in reversed(edits):
            before = ''.join(lines[:start[0]]) + lines[start[0]][:start[1]]
            after = ''.join(lines[end[0]][end[1]:] + ''.join(lines[end[0] + 1:]) if end[0] + 1 < len(lines) else lines[end[0]][end[1]:])
            src = before + replacement + after
            lines = src.splitlines(keepends=True)
        path.write_text(''.join(lines), encoding='utf-8')

    return total_changed, report


def process_rst_file(path, dry_run=True):
    text = path.read_text(encoding='utf-8')
    in_code_block = False
    changed = 0
    report = []
    out_lines = []

    for line in text.splitlines(keepends=True):
        if re.match(r"^\s*\.\.\s+code-block::", line):
            in_code_block = True
            out_lines.append(line)
            continue
        if in_code_block:
            if re.match(r"^\s*$", line):
                in_code_block = False
            out_lines.append(line)
            continue

        new_line = line
        # preserve inline literal segments ``...``
        parts = re.split(r"(``.*?``)", new_line, flags=re.DOTALL)
        rewritten = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                rewritten.append(part)
            else:
                rewritten.append(WORD_RE.sub(britishise_word, part))
        processed = ''.join(rewritten)
        if processed != line:
            changed += 1
            report.append((path, line.strip(), processed.strip()))
        out_lines.append(processed)

    if changed and not dry_run:
        path.write_text(''.join(out_lines), encoding='utf-8')

    return changed, report


def find_files():
    py_files = sorted(pathlib.Path('network_diffusion').rglob('*.py'))
    rst_files = sorted(pathlib.Path('docs/source').rglob('*.rst'))
    return py_files, rst_files


def main():
    parser = argparse.ArgumentParser(description='Britishise docstrings and docs')
    parser.add_argument('--dry-run', action='store_true', help='Only report what would change')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print('Either --dry-run or --apply is required', file=sys.stderr)
        return 1

    py_files, rst_files = find_files()
    overall = 0

    for path in py_files:
        changed, report = process_python_file(path, dry_run=args.dry_run)
        if changed:
            print(f"[PY] {path}: {changed} docstring(s) changed")
            for file_path, before, after in report:
                print('  -', repr(before[:80]), '->', repr(after[:80]))
            overall += changed

    for path in rst_files:
        changed, report = process_rst_file(path, dry_run=args.dry_run)
        if changed:
            print(f"[RST] {path}: {changed} line(s) changed")
            overall += changed

    print(f"Total edits proposed: {overall}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
