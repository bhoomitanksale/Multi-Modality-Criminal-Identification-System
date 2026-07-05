import sys
sys.path.insert(0, '.')
print('Testing imports...')
from config.settings import CRIMINAL_THRESHOLD, FUSION_WEIGHTS
print(f'  [OK] config.settings - Threshold: {CRIMINAL_THRESHOLD}, Weights: {FUSION_WEIGHTS}')

from database.db_manager import init_db, get_all_criminals, get_stats
print('  [OK] database.db_manager imported')
init_db()
print('  [OK] Database initialized')
criminals = get_all_criminals()
print(f'  [OK] Criminals in DB: {len(criminals)}')
for c in criminals:
    print(f'       - {c["name"]:20} | {c["crime_type"]:20} | {c["risk_level"]}')
stats = get_stats()
print(f'  [OK] Stats: {stats}')

from modules.face_analyzer import FaceAnalyzer, FaceFeatures
print('  [OK] face_analyzer imported')

from modules.gait_analyzer import GaitAnalyzer, MP_AVAILABLE
print(f'  [OK] gait_analyzer imported (MediaPipe: {MP_AVAILABLE})')

from modules.behavior_analyzer import BehaviorAnalyzer
print('  [OK] behavior_analyzer imported')

from modules.fusion_engine import FusionEngine, FusionResult
print('  [OK] fusion_engine imported')

from modules.scanner import Scanner, ScanResult
print('  [OK] scanner imported')

print()
print('Testing fusion engine...')
engine = FusionEngine()
result = engine.fuse(
    face_conf=0.72, gait_conf=0.58, behavior_conf=0.65,
    face_features=[0.1]*32,
    face_available=True, gait_available=True, behavior_available=True,
    scan_num=1
)
print(f'  [OK] Fusion: {result.verdict} ({result.fusion_conf:.0%}) | DB:{result.suspect_name}')
result2 = engine.fuse(
    face_conf=0.25, gait_conf=0.20, behavior_conf=0.18,
    face_available=True, gait_available=True, behavior_available=True,
    scan_num=2
)
print(f'  [OK] Fusion (clear): {result2.verdict} ({result2.fusion_conf:.0%})')
agg = engine.aggregate_scans()
print(f'  [OK] Aggregated: {agg.verdict} ({agg.fusion_conf:.0%})')
print()
print('=== All CIS v2 checks passed! ===')
