import csv
import glob
import os


INPUT_DIR = "./dataset/hindi_chunks"
OUTPUT_FILE = "hindi_only.csv"


def main():
    csv_files = glob.glob(
        os.path.join(INPUT_DIR, "*.csv")
    )

    print(f"Found {len(csv_files)} CSV files")

    total_written = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as out:

        writer = None

        for csv_file in csv_files:
            print(f"Processing {csv_file}...")

            with open(
                csv_file,
                "r",
                encoding="utf-8",
                newline=""
            ) as f:

                reader = csv.DictReader(f)

                # Write header once
                if writer is None:
                    writer = csv.DictWriter(
                        out,
                        fieldnames=reader.fieldnames
                    )
                    writer.writeheader()

                for row in reader:

                    if row["content_languages"].strip() == "hin":
                        writer.writerow(row)
                        total_written += 1

    print()
    print("=" * 50)
    print(f"Finished")
    print(f"Hindi-only records: {total_written:,}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
