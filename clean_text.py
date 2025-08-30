from pathlib import Path
import re

RAW_TXT_DIR = Path("data/clean")       # your extracted .txt files
CLEANED_DIR = Path("data/clean_final") # where we'll put cleaned text
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

def clean_file(input_path: Path, output_path: Path):
    cleaned_lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove page numbers (lines that are only digits)
            if re.fullmatch(r"\d+", line):
                continue

            # Remove NPNF-style headers/footers (common patterns)
            if line.startswith("NPNF") or "Christian Literature" in line:
                continue

            # Optional: remove ALL CAPS headers (likely section titles)
            if line.isupper() and len(line.split()) < 6:
                continue

            cleaned_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

def main():
    for txt_file in RAW_TXT_DIR.glob("*.txt"):
        out_path = CLEANED_DIR / txt_file.name
        print(f"Cleaning {txt_file.name} → {out_path.name}")
        clean_file(txt_file, out_path)

if __name__ == "__main__":
    main()
