import csv
from collections import Counter
from pathlib import Path

csv.field_size_limit(2**31 - 1)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
print("Looking in:", RAW)

print("\n=== FILE TREE ===")
files = sorted(p for p in RAW.rglob("*") if p.is_file())
for p in files:
    print(f"{p.relative_to(RAW)}  ({p.stat().st_size / 1e6:.2f} MB)")

csvs = [p for p in files if p.suffix.lower() == ".csv"]
print(f"\nFound {len(csvs)} CSV files")

print("\n=== PEEK AT FIRST CSV ===")
if csvs:
    first = csvs[0]
    with open(first, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    print("File:", first.relative_to(RAW))
    print("Columns:", header)
    for name, value in zip(header, row):
        print(f"\n--- {name} (first 300 chars) ---")
        print(value[:300])

print("\n=== PACKETS AND CLASS COUNTS PER CSV ===")
for p in csvs:
    n = 0
    classes = Counter()
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            if "class" in row:
                classes[row["class"]] += 1
    print(f"{p.relative_to(RAW)}: {n} packets, class counts: {dict(classes)}")