import ast
import sys

try:
    with open('nas_framework/search_strategy.py', 'r') as f:
        code = f.read()
    ast.parse(code)
    print("OK")
except SyntaxError as e:
    print(f"Line {e.lineno}: {e.msg}")
    print(f"Text: {e.text}")
    print(f"Offset: {e.offset}")
    if e.lineno:
        lines = code.split('\n')
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        print("\nContext:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1}: {lines[i]}")
