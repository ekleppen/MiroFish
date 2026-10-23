"""Temporary helper for the literal translation pass. Delete when done.

Usage: python3 .translate_locale_add.py fragment.json
fragment.json = {"en": {...nested keys...}, "zh": {...same keys...}}
Merges the keys into locales/en.json and locales/zh.json under a file lock,
preserving the existing file layout. Refuses to overwrite an existing key with
a different value, and requires en/zh to define exactly the same keys.
"""
import fcntl
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCALES = os.path.join(ROOT, 'locales')


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def render(data, original):
    text = json.dumps(data, ensure_ascii=False, indent=2)

    def inline(m):
        items = re.findall(r'"(?:[^"\\]|\\.)*"', m.group(2))
        return f'{m.group(1)}[{", ".join(items)}]'

    text = re.sub(r'(: )\[\n((?:\s*"(?:[^"\\]|\\.)*",?\n)+)\s*\]', inline, text)
    if original.endswith('\n'):
        text += '\n'
    return text


def merge(target, fragment, path=''):
    for k, v in fragment.items():
        key = f'{path}{k}'
        if isinstance(v, dict):
            node = target.setdefault(k, {})
            if not isinstance(node, dict):
                raise SystemExit(f'conflict: {key} is a string in the locale file')
            merge(node, v, key + '.')
        elif k in target and target[k] != v:
            raise SystemExit(f'conflict: {key} already exists with a different value: {target[k]!r}')
        else:
            target[k] = v


def main():
    fragment = json.load(open(sys.argv[1], encoding='utf-8'))
    en_keys, zh_keys = set(flatten(fragment['en'])), set(flatten(fragment['zh']))
    if en_keys != zh_keys:
        raise SystemExit(f'en/zh key mismatch: only en={sorted(en_keys - zh_keys)} only zh={sorted(zh_keys - en_keys)}')
    for key in en_keys:
        en_ph = set(re.findall(r'\{(\w+)\}', flatten(fragment['en'])[key]))
        zh_ph = set(re.findall(r'\{(\w+)\}', flatten(fragment['zh'])[key]))
        if en_ph != zh_ph:
            raise SystemExit(f'placeholder mismatch for {key}: en={en_ph} zh={zh_ph}')

    with open(os.path.join(LOCALES, '.merge.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        loaded = {}
        for lang in ('en', 'zh'):
            path = os.path.join(LOCALES, f'{lang}.json')
            original = open(path, encoding='utf-8').read()
            data = json.loads(original)
            if render(data, original) != original:
                raise SystemExit(f'{path} layout would change; aborting')
            merge(data, fragment[lang])
            loaded[lang] = (path, render(data, original))
        for path, text in loaded.values():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
    print(f'merged {len(en_keys)} keys')


if __name__ == '__main__':
    main()
