import re
import csv
import sys
from pathlib import Path


# Patterns
RE_GENERATION  = re.compile(r'Running generation (\d+)')
RE_AVG_FITNESS = re.compile(r"Population's average fitness:\s*([-\d.]+)")
RE_BEST        = re.compile(r'Best fitness:\s*([-\d.]+)')
RE_GEN_LINE    = re.compile(r'Gen\s+(\d+)\s*\|\s*Species:\s*(\d+)')


def parse_log(path: Path) -> list[dict]:
    records = []
    current = {}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            m = RE_GENERATION.search(line)
            if m:
                current = {'generation': int(m.group(1))}
                continue

            m = RE_AVG_FITNESS.search(line)
            if m and 'generation' in current:
                current['avg_fitness'] = float(m.group(1))
                continue

            m = RE_BEST.search(line)
            if m and 'generation' in current:
                current['best_fitness'] = float(m.group(1))
                continue

            # "Gen  0 | Species:   4 ..." — signals end of generation block
            m = RE_GEN_LINE.search(line)
            if m and 'generation' in current:
                current['species_count'] = int(m.group(2))
                # Only store complete records
                if all(k in current for k in ('avg_fitness', 'best_fitness', 'species_count')):
                    records.append(current)
                current = {}

    return records


def save_csv(records: list[dict], out_path: Path):
    fieldnames = ['generation', 'best_fitness', 'avg_fitness', 'species_count']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} generations -> {out_path}")


def main(paths: list[Path]):
    for i, log_path in enumerate(paths):
        print(f"{i}: {log_path}")

    c = input("Select log to parse: ")
    if not c.isdecimal():
        print("Invalid Choice")
        return
    c = int(c)
    if c < 0 or c >= len(paths):
        print("Invalid Choice")
        return

    log_path = paths[c]
    print(f"Parsing {log_path.name} ...")
    records = parse_log(log_path)

    if not records:
        print(f"  [WARN] No complete generation records found in {log_path.name}")
        return

    candidate = log_path.with_suffix('.csv')
    try:
        candidate.touch()
        out_path = candidate
    except OSError:
        out_path = Path(log_path.stem + '.csv')
    save_csv(records, out_path)


if __name__ == '__main__':
    logs_path = Path.cwd() / "logs"
    paths = list(logs_path.glob("*.log"))
    main(paths)
