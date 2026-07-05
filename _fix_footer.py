import re

with open('cis_v2.py', encoding='utf-8') as f:
    content = f.read()

# Fix footer threshold text
old = 'CRIMINAL >62%  |  WATCH LIST >45%  |  CLEAR <45%'
new = 'CRIMINAL >52%  |  SUSPECT >38%  |  INNOCENT <38%'

if old in content:
    content = content.replace(old, new, 1)
    print('Footer threshold: REPLACED OK')
else:
    print('Footer old text not found, searching...')
    idx = content.find('CRIMINAL >')
    if idx >= 0:
        print('Context:', repr(content[idx:idx+80]))

# Also fix Face weights label in footer
old2 = 'Face 45% | Gait 30% | Behavior 25%'
new2 = 'Face 60% | Gait 5% | Behavior 35% (image mode)'
if old2 in content:
    content = content.replace(old2, new2, 1)
    print('Footer weights: REPLACED OK')

with open('cis_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
import ast
ast.parse(content)
print('SYNTAX OK')
