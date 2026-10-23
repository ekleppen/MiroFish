"""Temporary helper for the literal translation pass. Delete when done.

Usage: python3 .translate_keys_check.py [files...]   (defaults to backend/app/**/*.py)
Checks every t('key', **kwargs) call: the key must exist as a string in both
locales/en.json and locales/zh.json, and the {placeholders} in each translation
must be supplied as keyword arguments. Also lists remaining Chinese string
literals per file (excluding docstrings) so leftovers are easy to find.
"""
import ast
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
HAN = re.compile(r'[\u4e00-\u9fff]')


def lookup(data, key):
    for part in key.split('.'):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data if isinstance(data, str) else None


def docstring_ids(tree):
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def main():
    locales = {lang: json.load(open(os.path.join(ROOT, 'locales', f'{lang}.json'), encoding='utf-8')) for lang in ('en', 'zh')}
    files = sys.argv[1:] or glob.glob(os.path.join(ROOT, 'backend/app/**/*.py'), recursive=True)
    problems = 0
    for f in sorted(files):
        tree = ast.parse(open(f, encoding='utf-8').read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 't'):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            key = node.args[0].value
            kwargs = {k.arg for k in node.keywords if k.arg}
            has_star = any(k.arg is None for k in node.keywords)
            for lang, data in locales.items():
                value = lookup(data, key)
                if value is None:
                    print(f'MISSING   {f}:{node.lineno} {key} ({lang})')
                    problems += 1
                    continue
                needed = set(re.findall(r'\{(\w+)\}', value))
                if not has_star and needed - kwargs:
                    print(f'NO ARG    {f}:{node.lineno} {key} ({lang}) needs {sorted(needed - kwargs)}')
                    problems += 1
        docs = docstring_ids(tree)
        left = sorted({n.lineno for n in ast.walk(tree)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and HAN.search(n.value) and id(n) not in docs})
        if left:
            print(f'CHINESE   {os.path.relpath(f, ROOT)}: {len(left)} literal(s) at lines {left[:40]}{" ..." if len(left) > 40 else ""}')
    print('key problems:', problems)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
