import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(path: Path) -> dict:
    data = {"generation": [], "best_fitness": [], "avg_fitness": [], "species_count": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data["generation"].append(int(row["generation"]))
            data["best_fitness"].append(float(row["best_fitness"]))
            data["avg_fitness"].append(float(row["avg_fitness"]))
            data["species_count"].append(int(row["species_count"]))
    return data


def plot(datasets: list[tuple[str, dict]]):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("NEAT Training Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("best_fitness",  "Best Fitness",      axes[0]),
        ("avg_fitness",   "Avg Population Fitness", axes[1]),
        ("species_count", "Species Count",     axes[2]),
    ]

    for label, data in datasets:
        for key, _, ax in metrics:
            ax.plot(data["generation"], data[key], label=label, linewidth=1.5)

    for key, title, ax in metrics:
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    axes[-1].set_xlabel("Generation")
    plt.tight_layout()
    plt.show()


def main(paths: list[Path]):
    for i, p in enumerate(paths):
        print(f"  {i}: {p.stem}")

    raw = input("\nSelect CSVs to graph (comma-separated, e.g. 0,1): ").strip()
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdecimal():
            print(f"Invalid token: '{part}'")
            return
        indices.append(int(part))

    invalid = [i for i in indices if i < 0 or i >= len(paths)]
    if invalid:
        print(f"Out of range: {invalid}")
        return

    datasets = [(paths[i].stem, load_csv(paths[i])) for i in indices]
    plot(datasets)


if __name__ == "__main__":
    tables_path = Path.cwd() / "logs"
    paths = sorted(tables_path.glob("*.csv"))

    if not paths:
        print(f"No CSV files found in {tables_path}")
    else:
        main(paths)
