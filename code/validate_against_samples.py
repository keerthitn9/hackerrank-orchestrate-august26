"""
Validation script: runs the routing pipeline against dataset/sample_messages.csv
(which has known-correct action/message_type/reason/confidence/evidence_message_ids
columns already filled in) and reports how many predictions match the gold labels.

This does NOT modify main.py or output.csv. It is a read-only accuracy check
against the one part of the dataset that has real ground truth.

Run from repo root:
    python code/validate_against_samples.py
"""

import os
import sys

# Import everything from the existing pipeline without duplicating any logic.
sys.path.insert(0, os.path.dirname(__file__))
from main import (  # noqa: E402
    DATASET_DIR,
    load_all_tables,
    build_indexes,
    route_message,
)

import pandas as pd  # noqa: E402


def run_validation(dataset_dir: str) -> None:
    tables = load_all_tables(dataset_dir)
    sample_df = tables.get("sample_messages")

    sample_path = os.path.join(dataset_dir, "sample_messages.csv")
    if sample_df is None or sample_df.empty:
        # sample_messages.csv isn't part of load_all_tables's default set;
        # load it directly instead.
        sample_df = pd.read_csv(sample_path)

    idx = build_indexes(tables)

    action_correct = 0
    type_correct = 0
    both_correct = 0
    total = 0
    mismatches = []

    for _, row in sample_df.iterrows():
        gold_action = row.get("action")
        gold_type = row.get("message_type")
        if pd.isna(gold_action) or pd.isna(gold_type):
            continue  # skip any sample rows that aren't fully labeled

        # sample_messages.csv has the same columns as messages.csv plus the
        # gold-label columns; route_message only reads the messages.csv-style
        # columns, so passing the row directly works unmodified.
        predicted = route_message(row, idx)

        total += 1
        action_match = predicted["action"] == gold_action
        type_match = predicted["message_type"] == gold_type

        if action_match:
            action_correct += 1
        if type_match:
            type_correct += 1
        if action_match and type_match:
            both_correct += 1

        if not (action_match and type_match):
            mismatches.append({
                "message_id": row.get("message_id"),
                "gold_action": gold_action,
                "pred_action": predicted["action"],
                "gold_type": gold_type,
                "pred_type": predicted["message_type"],
            })

    print(f"Total labeled sample rows checked: {total}")
    print(f"Action accuracy:      {action_correct}/{total} ({100*action_correct/total:.1f}%)")
    print(f"Message type accuracy: {type_correct}/{total} ({100*type_correct/total:.1f}%)")
    print(f"Both correct:          {both_correct}/{total} ({100*both_correct/total:.1f}%)")
    print()

    if mismatches:
        print("Mismatches:")
        for m in mismatches:
            print(
                f"  {m['message_id']}: "
                f"action gold={m['gold_action']!r} pred={m['pred_action']!r} | "
                f"type gold={m['gold_type']!r} pred={m['pred_type']!r}"
            )
    else:
        print("No mismatches. All labeled sample rows matched.")


if __name__ == "__main__":
    run_validation(DATASET_DIR)
