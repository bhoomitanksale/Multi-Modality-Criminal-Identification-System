import sys, sqlite3, json
sys.path.insert(0, '.')
from config.settings import DB_PATH

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 90)
print("  CIS v2 DATABASE — CRIMINAL REGISTRY")
print("=" * 90)

c = conn.cursor()
c.execute("SELECT * FROM criminals ORDER BY risk_level='HIGH' DESC, name ASC")
rows = c.fetchall()
print(f"\n  Total suspects: {len(rows)}\n")
print(f"  {'ID':<4} {'Name':<22} {'Alias':<14} {'DOB':<12} {'Gender':<7} {'Height':<7} {'Weight':<7} {'Risk':<8} {'Crime Type':<22} {'Status'}")
print("  " + "-"*88)
for r in rows:
    print(f"  {r['id']:<4} {r['name']:<22} {r['alias']:<14} {r['dob']:<12} {r['gender']:<7} {str(r['height_cm'])+'cm':<7} {str(r['weight_kg'])+'kg':<7} {r['risk_level']:<8} {r['crime_type']:<22} {r['status']}")

print("\n" + "=" * 90)
print("  FACE BIOMETRICS (stored per suspect)")
print("=" * 90)
c.execute("""
    SELECT fb.id, fb.criminal_id, cr.name, fb.eye_distance, fb.face_ratio, fb.brightness
    FROM face_biometrics fb JOIN criminals cr ON cr.id = fb.criminal_id
""")
bio_rows = c.fetchall()
print(f"\n  {'ID':<5} {'Criminal ID':<12} {'Name':<22} {'Eye Dist':<10} {'Face Ratio':<12} {'Brightness'}")
print("  " + "-"*70)
for r in bio_rows:
    print(f"  {r['id']:<5} {r['criminal_id']:<12} {r['name']:<22} {r['eye_distance']:<10} {r['face_ratio']:<12} {r['brightness']}")

print("\n" + "=" * 90)
print("  DETECTION EVENTS LOG")
print("=" * 90)
c.execute("""
    SELECT de.id, de.timestamp, cr.name, de.mode, de.face_conf, de.gait_conf,
           de.behavior_conf, de.fusion_conf, de.verdict
    FROM detection_events de
    LEFT JOIN criminals cr ON cr.id = de.criminal_id
    ORDER BY de.id DESC LIMIT 20
""")
det_rows = c.fetchall()
if not det_rows:
    print("\n  No detections recorded yet.")
else:
    print(f"\n  {'ID':<5} {'Timestamp':<20} {'Suspect':<20} {'Mode':<8} {'Face':<7} {'Gait':<7} {'Behav':<7} {'Fusion':<8} {'Verdict'}")
    print("  " + "-"*88)
    for r in det_rows:
        name = r['name'] or 'Unknown'
        ts   = r['timestamp'][:16] if r['timestamp'] else ''
        print(f"  {r['id']:<5} {ts:<20} {name:<20} {r['mode']:<8} {r['face_conf']:.0%}   {r['gait_conf']:.0%}   {r['behavior_conf']:.0%}   {r['fusion_conf']:.0%}    {r['verdict']}")

conn.close()
print("\n" + "=" * 90)
