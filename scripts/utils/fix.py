import re

import os
_root = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_root, 'app.py'), 'r', encoding='utf-8') as f:
    c = f.read()

# Fix unicode dashes
c = c.replace('\u2014', '-')
c = c.replace('\u2013', '-')

# Fix the specific broken patterns
c = c.replace("or '-').upper()", "or 'N/A').upper()")

with open(os.path.join(_root, 'app.py'), 'w', encoding='utf-8') as f:
    f.write(c)

print('FIXED - no more unicode issues')
