import py_compile, sys

files = [
    'cis_v2.py',
    'config/settings.py',
    'database/db_manager.py',
    'modules/face_analyzer.py',
    'modules/gait_analyzer.py',
    'modules/behavior_analyzer.py',
    'modules/fusion_engine.py',
    'modules/scanner.py',
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'  [OK]  {f}')
    except py_compile.PyCompileError as e:
        print(f'  [ERR] {f}: {e}')
        all_ok = False

print()
print('SYNTAX CHECK: PASSED' if all_ok else 'SYNTAX CHECK: FAILED')
sys.exit(0 if all_ok else 1)
