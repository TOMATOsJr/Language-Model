import csv
import glob
from collections import Counter


INPUT_DIR = "hindi_chunks"


def main():
    exact = Counter()

    hindi_only = 0
    hindi_primary = 0
    hindi_secondary = 0
    hindi_middle = 0

    total_rows = 0

    csv_files = glob.glob(f"{INPUT_DIR}/*.csv")

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"Reading {csv_file}...")

        with open(
            csv_file,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                value = row.get("content_languages")

                if not value:
                    continue

                value = value.strip()

                if not value:
                    continue

                total_rows += 1

                # Count exact language combinations
                exact[value] += 1

                languages = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]

                # Only considering rows containing Hindi
                if "hin" not in languages:
                    continue

                if len(languages) == 1:
                    hindi_only += 1

                elif languages[0] == "hin":
                    hindi_primary += 1

                elif languages[-1] == "hin":
                    hindi_secondary += 1

                else:
                    hindi_middle += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("HINDI LANGUAGE POSITION")
    print("=" * 60)

    print(f"Hindi only:        {hindi_only:,}")
    print(f"Hindi primary:     {hindi_primary:,}")
    print(f"Hindi secondary:   {hindi_secondary:,}")
    print(f"Hindi middle:      {hindi_middle:,}")

    print()
    print("=" * 60)
    print("EXACT CONTENT_LANGUAGE VALUES")
    print("=" * 60)

    for value, count in exact.most_common():
        print(f"{value:30} {count:,}")

    print()
    print("=" * 60)
    print(f"Total rows: {total_rows:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
