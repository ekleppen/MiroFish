"""Temporary helper for the comment translation pass. Delete when done.

Usage: python3 .translate_check.py [files...]   (defaults to all tracked .py files)
For each file: reports whether code (AST with docstrings blanked) matches HEAD,
and how many comment/docstring lines still contain Chinese.
"""
import ast
import io
import re
import subprocess
import sys
import tokenize

HAN = re.compile(r'[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]')


def docstring_nodes(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                yield first.value


def code_fingerprint(src):
    tree = ast.parse(src)
    for node in docstring_nodes(tree):
        node.value = ''
    return ast.dump(tree)


def remaining(src):
    count = 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and HAN.search(tok.string):
            count += 1
    for node in docstring_nodes(ast.parse(src)):
        count += sum(1 for line in node.value.splitlines() if HAN.search(line))
    return count


def main():
    root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()
    files = sys.argv[1:] or subprocess.check_output(['git', 'ls-files', '*.py'], cwd=root).decode().split()
    bad = False
    for f in files:
        rel = subprocess.check_output(['git', 'ls-files', '--full-name', f]).decode().strip() or f
        new = open(f, encoding='utf-8').read()
        try:
            old = subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=root, stderr=subprocess.DEVNULL).decode()
            same = code_fingerprint(old) == code_fingerprint(new)
        except subprocess.CalledProcessError:
            same = None
        left = remaining(new)
        status = 'OK ' if same else ('NEW' if same is None else 'CODE CHANGED')
        if same is False:
            bad = True
        print(f'{status:12} remaining={left:4}  {f}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
