
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent

PARTITIONS = ["train", "student_test"]

REQUIRED_EXTENSIONS = [
    ".track",
    ".nodes",
    ".arcs",
    ".route"
]

OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inventory = []


def count_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


for partition in PARTITIONS:

    partition_path = BASE_DIR / partition

    if not partition_path.exists():
        print(f"WARNING: {partition} folder not found")
        continue

    trip_folders = sorted(
        folder for folder in partition_path.iterdir()
        if folder.is_dir()
    )

    print(f"\nInspecting {partition}...")
    print(f"Number of trips: {len(trip_folders)}")

    for trip_folder in trip_folders:

        trip_id = trip_folder.name

        record = {
            "partition": partition,
            "trip_id": trip_id
        }

        for extension in REQUIRED_EXTENSIONS:

            file_path = trip_folder / f"{trip_id}{extension}"

            if file_path.exists():

                record[f"{extension}_exists"] = True
                record[f"{extension}_rows"] = count_rows(file_path)
                record[f"{extension}_size_kb"] = round(
                    file_path.stat().st_size / 1024, 2
                )

            else:

                record[f"{extension}_exists"] = False
                record[f"{extension}_rows"] = 0
                record[f"{extension}_size_kb"] = 0

        inventory.append(record)


if inventory:

    output_file = OUTPUT_DIR / "dataset_inventory.csv"

    fieldnames = list(inventory[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    print("\n========== DATASET INVENTORY ==========")

    for partition in PARTITIONS:

        trips = [
            row for row in inventory
            if row["partition"] == partition
        ]

        print(f"\n{partition.upper()}")
        print(f"Total trips: {len(trips)}")

        for extension in REQUIRED_EXTENSIONS:

            total_rows = sum(
                row[f"{extension}_rows"] for row in trips
            )

            print(f"{extension} total rows: {total_rows}")

    print("\nInventory saved successfully!")
    print(f"Output: {output_file}")

else:
    print("No trip data was found.")
