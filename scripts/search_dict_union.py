from pathlib import Path

needle = " " + chr(124) + " "
for root_name in ("reloc3r", "dust3r", "scripts"):
    root = Path(root_name)
    if not root.exists():
        continue
    for path in root.rglob("*.py"):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            if needle in line:
                print(f"{path}:{lineno}:{line}")
