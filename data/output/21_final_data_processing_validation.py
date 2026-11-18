
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

PARTITIONS = {
    "train": BASE_DIR / "train",
    "student_test": BASE_DIR / "student_test"
}

PROCESSED_GPS = BASE_DIR / "outputs" / "processed"
PROCESSED_ROAD = PROCESSED_GPS / "road_network"
REPORTS = BASE_DIR / "outputs" / "reports"

REQUIRED_REPORTS = [
    "dataset_inventory.csv",
    "gps_gap_analysis.csv",
    "gps_anomaly_details.csv",
    "gps_preprocessing_summary.csv",
    "processed_data_validation.csv",
    "all_road_networks_summary.csv",
    "road_network_validation.csv",
    "reference_route_validation.csv",
    "disconnected_route_details.csv",
    "route_disconnection_distances.csv"
]

errors = []
total_trips = 0
gps_pass = 0
nodes_pass = 0
arcs_pass = 0

print("=" * 65)
print("FINAL DATA PROCESSING VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. CHECK RAW AND PROCESSED TRIP FILES
# --------------------------------------------------

for partition, raw_folder in PARTITIONS.items():

    print(f"\nChecking partition: {partition}")

    if not raw_folder.exists():
        errors.append(f"Missing raw folder: {raw_folder}")
        print("[FAIL] Raw folder not found")
        continue

    # Search recursively inside subfolders
    trip_files = sorted(raw_folder.rglob("*.track"))

    print("Raw trips found:", len(trip_files))

    for track_file in trip_files:

        trip_id = track_file.stem
        total_trips += 1

        gps_file = (
            PROCESSED_GPS / partition / f"{trip_id}.csv"
        )

        road_folder = (
            PROCESSED_ROAD / partition / trip_id
        )

        nodes_file = (
            road_folder / f"{trip_id}_processed_nodes.csv"
        )

        arcs_file = (
            road_folder / f"{trip_id}_processed_arcs.csv"
        )

        # Check processed GPS
        try:
            if gps_file.exists():
                df = pd.read_csv(gps_file)

                if not df.empty:
                    gps_pass += 1
                else:
                    errors.append(f"Empty GPS file: {gps_file}")
            else:
                errors.append(f"Missing GPS file: {gps_file}")

        except Exception as e:
            errors.append(
                f"Unreadable GPS file: {gps_file} - {e}"
            )

        # Check processed nodes
        try:
            if nodes_file.exists():
                df = pd.read_csv(nodes_file)

                if not df.empty:
                    nodes_pass += 1
                else:
                    errors.append(f"Empty nodes file: {nodes_file}")
            else:
                errors.append(f"Missing nodes file: {nodes_file}")

        except Exception as e:
            errors.append(
                f"Unreadable nodes file: {nodes_file} - {e}"
            )

        # Check processed arcs
        try:
            if arcs_file.exists():
                df = pd.read_csv(arcs_file)

                if not df.empty:
                    arcs_pass += 1
                else:
                    errors.append(f"Empty arcs file: {arcs_file}")
            else:
                errors.append(f"Missing arcs file: {arcs_file}")

        except Exception as e:
            errors.append(
                f"Unreadable arcs file: {arcs_file} - {e}"
            )


# --------------------------------------------------
# 2. PRINT PROCESSED FILE SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("PROCESSED FILE SUMMARY")
print("=" * 65)

print("Total trips:", total_trips)
print(f"GPS files:   {gps_pass}/{total_trips}")
print(f"Node files:  {nodes_pass}/{total_trips}")
print(f"Arc files:   {arcs_pass}/{total_trips}")

if total_trips == 0:
    errors.append(
        "No raw trips found. Check train and student_test folder paths."
    )


# --------------------------------------------------
# 3. CHECK REQUIRED REPORTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("REPORT FILE CHECK")
print("=" * 65)

report_pass = 0

for report in REQUIRED_REPORTS:

    report_path = REPORTS / report

    if report_path.exists():

        try:
            df = pd.read_csv(report_path)

            if not df.empty:
                report_pass += 1
                print("[PASS]", report)
            else:
                errors.append(f"Empty report: {report}")
                print("[FAIL] Empty:", report)

        except Exception as e:
            errors.append(
                f"Unreadable report: {report} - {e}"
            )
            print("[FAIL] Unreadable:", report)

    else:
        errors.append(f"Missing report: {report}")
        print("[FAIL] Missing:", report)

print("\nReports available:", f"{report_pass}/{len(REQUIRED_REPORTS)}")


# --------------------------------------------------
# 4. CHECK EVALUATOR DATA FOLDER
# --------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATOR DATA CHECK")
print("=" * 65)

possible_evaluator_locations = [
    BASE_DIR / "evaluator_test",
    BASE_DIR / "test",
    BASE_DIR / "evaluator",
    BASE_DIR / "data" / "evaluator_test"
]

evaluator_found = False

for folder in possible_evaluator_locations:

    if folder.exists() and folder.is_dir():
        evaluator_found = True
        print("[FOUND]", folder)
        break

if not evaluator_found:
    print("[NOT FOUND] Evaluator folder not found in common locations.")
    print("Confirm its actual location with your team.")
    print("This does not affect the training/student-test checks.")


# --------------------------------------------------
# 5. FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)

all_files_pass = (
    total_trips == 90
    and gps_pass == total_trips
    and nodes_pass == total_trips
    and arcs_pass == total_trips
    and report_pass == len(REQUIRED_REPORTS)
)

if all_files_pass and not errors:

    print("[SUCCESS] ALL DATA PROCESSING CHECKS PASSED!")
    print("Training trips: 80 expected")
    print("Student testing trips: 10 expected")
    print("Processed GPS files:", gps_pass)
    print("Processed node files:", nodes_pass)
    print("Processed arc files:", arcs_pass)
    print("Required reports:", report_pass)

else:

    print("[WARNING] SOME CHECKS NEED ATTENTION.")
    print("Issues found:", len(errors))

    for error in errors:
        print("-", error)

print("\nValidation completed.")
