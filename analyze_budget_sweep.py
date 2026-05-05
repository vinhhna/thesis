import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MODES = ["dense", "sparse_192", "sparse_128", "sparse_96", "sparse_64"]
ANSWER_COLUMNS = {mode: f"{mode}_answer" for mode in MODES}
CORRECT_COLUMNS = {mode: f"{mode}_correct" for mode in MODES}
KNOWN_DATASET_ISSUES = {"GQA_VAL_019"}


def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9:./]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_matches(answer, ground_truth):
    answer_norm = normalize(answer)
    gt_norm = normalize(ground_truth)
    if not gt_norm or not answer_norm:
        return False
    if answer_norm == gt_norm:
        return True
    if f" {gt_norm} " in f" {answer_norm} ":
        return True
    answer_tokens = set(answer_norm.split())
    gt_tokens = set(gt_norm.split())
    return bool(gt_tokens) and gt_tokens.issubset(answer_tokens)


def read_failure_set(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def read_outputs(path):
    outputs = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            mode = record.get("mode")
            if not case_id or not mode:
                raise ValueError(f"Missing case_id/mode at line {line_number}")
            outputs[case_id][mode] = record
    return outputs


def choose_failure_signal(case_id, source_row, correctness):
    note = normalize(source_row.get("note", ""))
    dense = correctness.get("dense")
    sparse = [correctness.get(mode) for mode in ["sparse_192", "sparse_128", "sparse_96", "sparse_64"]]
    sparse_present = [value for value in sparse if value is not None]

    if case_id in KNOWN_DATASET_ISSUES or "dataset issue" in note or "dataset_issue" in note:
        return "likely_dataset_issue"
    if "ambiguous" in note:
        return "ambiguous_question"
    if dense is False:
        return "dense_wrong"
    if dense is True and any(value is False for value in sparse_present):
        return "dense_correct_sparse_wrong"
    if correctness.get("sparse_192") is True and any(
        correctness.get(mode) is False for mode in ["sparse_128", "sparse_96", "sparse_64"]
    ):
        return "pruning_sensitive"
    all_present = [value for value in [dense] + sparse_present if value is not None]
    if all_present and all(all_present):
        return "stable_correct"
    if all_present and not any(all_present):
        return "stable_wrong"
    return "needs_manual_review"


def build_analysis_rows(failure_rows, outputs):
    analysis_rows = []
    for case_id, source_row in failure_rows.items():
        mode_records = outputs.get(case_id, {})
        analysis_row = {
            "case_id": case_id,
            "dataset": source_row.get("dataset", ""),
            "question_type": source_row.get("question_type", ""),
            "ground_truth": source_row.get("ground_truth", ""),
        }

        correctness = {}
        for mode in MODES:
            record = mode_records.get(mode, {})
            answer = record.get("answer", "")
            is_correct = answer_matches(answer, source_row.get("ground_truth", "")) if answer else None
            analysis_row[ANSWER_COLUMNS[mode]] = answer
            analysis_row[CORRECT_COLUMNS[mode]] = "" if is_correct is None else str(is_correct)
            correctness[mode] = is_correct

        analysis_row["failure_signal"] = choose_failure_signal(case_id, source_row, correctness)
        analysis_row["review_note"] = source_row.get("note", "")
        analysis_rows.append(analysis_row)
    return analysis_rows


def write_csv(path, rows):
    fieldnames = [
        "case_id",
        "dataset",
        "question_type",
        "ground_truth",
        "dense_answer",
        "sparse_192_answer",
        "sparse_128_answer",
        "sparse_96_answer",
        "sparse_64_answer",
        "dense_correct",
        "sparse_192_correct",
        "sparse_128_correct",
        "sparse_96_correct",
        "sparse_64_correct",
        "failure_signal",
        "review_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_accuracy_tables(rows):
    print("Failure signals:", dict(Counter(row["failure_signal"] for row in rows)))
    for group_name in ["dataset", "question_type"]:
        print()
        print(f"Accuracy by {group_name}:")
        groups = defaultdict(list)
        for row in rows:
            groups[row[group_name]].append(row)
        for group, group_rows in sorted(groups.items()):
            parts = []
            for mode in MODES:
                col = CORRECT_COLUMNS[mode]
                values = [row[col] == "True" for row in group_rows if row[col] in {"True", "False"}]
                if values:
                    parts.append(f"{mode}={sum(values)}/{len(values)}")
                else:
                    parts.append(f"{mode}=NA")
            print(f"{group}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Analyze Kaggle SparseVLM budget sweep outputs.")
    parser.add_argument("--failure-set", default="failure_mining_set.csv")
    parser.add_argument("--outputs", default="failure_mining_budget_sweep_outputs.jsonl")
    parser.add_argument("--out-csv", default="failure_mining_budget_sweep_analysis.csv")
    args = parser.parse_args()

    failure_rows = read_failure_set(Path(args.failure_set))
    outputs = read_outputs(Path(args.outputs))
    analysis_rows = build_analysis_rows(failure_rows, outputs)
    write_csv(Path(args.out_csv), analysis_rows)

    expected = len(failure_rows) * len(MODES)
    actual = sum(len(mode_records) for mode_records in outputs.values())
    print(f"Cases: {len(failure_rows)}")
    print(f"Output records: {actual}/{expected}")
    print(f"Wrote: {args.out_csv}")
    print_accuracy_tables(analysis_rows)


if __name__ == "__main__":
    main()
