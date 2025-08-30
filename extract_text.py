from pathlib import Path
from pypdf import PdfReader

# Input / Output folders
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_txt(pdf_path: Path, txt_path: Path):
    reader = PdfReader(str(pdf_path))
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                f.write(text + "\n")

def main():
    for pdf_file in RAW_DIR.glob("*.pdf"):
        txt_name = pdf_file.stem + ".txt"
        txt_path = CLEAN_DIR / txt_name
        print(f"Converting {pdf_file.name} → {txt_name}")
        pdf_to_txt(pdf_file, txt_path)

if __name__ == "__main__":
    main()
