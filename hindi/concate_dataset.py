import json
from pathlib import Path


INPUT_DIR = Path("./hindi_corpus").resolve()
OUTPUT_FILE = Path("./combined_hindi_corpus.jsonl").resolve()


def main():
    print(f"Input directory : {INPUT_DIR}")
    print(f"Directory exists: {INPUT_DIR.exists()}")
    print()

    json_files = sorted(
        INPUT_DIR.glob("*.json"),
        key=lambda p: int(p.stem)
    )

    txt_files = {
        p.stem: p
        for p in INPUT_DIR.glob("*.txt")
    }

    print(f"JSON files found: {len(json_files):,}")
    print(f"TXT files found : {len(txt_files):,}")
    print()

    total = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for json_file in json_files:
            doc_id = json_file.stem

            txt_file = txt_files.get(doc_id)

            if txt_file is None:
                print(
                    f"MISSING TXT for {json_file.name}"
                )
                skipped += 1
                continue

            try:
                with open(
                    json_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    metadata = json.load(f)

                with open(
                    txt_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    text = f.read()

                record = {
                    "doc_id": doc_id,
                    "url": metadata.get("url"),
                    "text": text,
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total += 1

                if total % 1000 == 0:
                    print(
                        f"Processed {total:,}",
                        flush=True
                    )

            except Exception as e:
                print(
                    f"ERROR processing {json_file.name}: {e}"
                )
                skipped += 1

    print()
    print("=" * 60)
    print(f"Records written : {total:,}")
    print(f"Skipped         : {skipped:,}")
    print(f"Output          : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()