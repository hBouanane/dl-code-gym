from pathlib import Path

path = Path("/data/count.txt")
path.parent.mkdir(parents=True, exist_ok=True)
count = int(path.read_text()) + 1 if path.exists() else 1
path.write_text(str(count))
print(f"run number {count}")
