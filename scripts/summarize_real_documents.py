"""Summary of the real-document run (data/real_document_data.csv).

    python scripts/summarize_real_documents.py

Prints overall accuracy and accuracy by question type, format, style, document
and difficulty, plus the wrong answers. Also writes data/real_wrong_answers.csv.
"""

import csv
from collections import defaultdict
from pathlib import Path

from agentops.config import DATA_DIR

DATA = Path(DATA_DIR)


def read(name):
    path = DATA / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def table(rows, key, title):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[row[key]][1] += 1
        groups[row[key]][0] += row["Answer Correct"] == "Yes"
    print(f"\n{title}")
    for name, (good, total) in sorted(groups.items(), key=lambda x: x[1][0] / x[1][1]):
        print(f"  {name:<38} {good:>3}/{total:<3} {100 * good / total:5.1f}%")


rows = read("real_document_data.csv")
after_rows = read("real_after_document_data.csv")
failures = read("real_document_failures.csv")
if not rows:
    raise SystemExit("data/real_document_data.csv not found. Run scripts/run_real_documents.py first.")

good = sum(r["Answer Correct"] == "Yes" for r in rows)
source = sum(r["Source Correct"] == "Yes" for r in rows)
print(f"Real-document runs judged: {len(rows)} | failed runs (API): {len(failures)}")
print(f"Answer correct: {good}/{len(rows)} = {100 * good / len(rows):.1f}%")
print(f"Source correct: {source}/{len(rows)} = {100 * source / len(rows):.1f}%")

table(rows, "Question Type", "By question type")
table(rows, "Format", "By format")
table(rows, "Question Style", "By style")
table(rows, "Difficulty", "By difficulty")
table(rows, "Doc File", "By document (worst first)")

wrong = [r for r in rows if r["Answer Correct"] != "Yes"]
with open(DATA / "real_wrong_answers.csv", "w", newline="", encoding="utf-8") as file:
    fields = ["Run No", "Doc File", "Question Type", "Question Style", "Source Correct", "Question", "Expected Answer"]
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(wrong)
print(f"\nWrong answers: {len(wrong)} (saved in data/real_wrong_answers.csv)")


# ---- before vs after the retrieval fixes (run with --tag after)
if after_rows:
    def key(r):
        return (r["Doc File"], r["Question"])

    def pct(items):
        return 100 * sum(r["Answer Correct"] == "Yes" for r in items) / len(items)

    print("\n" + "=" * 60)
    print(f"BEFORE: {pct(rows):.1f}% ({len(rows)} runs)   AFTER: {pct(after_rows):.1f}% ({len(after_rows)} runs)")

    def grouped(items, field):
        g = defaultdict(list)
        for r in items:
            g[r[field]].append(r)
        return g

    for field, title in (("Question Type", "By question type"), ("Doc File", "By document")):
        before_g, after_g = grouped(rows, field), grouped(after_rows, field)
        print(f"\n{title}: before -> after")
        for name in sorted(before_g, key=lambda n: pct(before_g[n])):
            if name in after_g:
                b, a = pct(before_g[name]), pct(after_g[name])
                print(f"  {name:<38} {b:5.1f}% -> {a:5.1f}%  ({a - b:+.1f})")

    before_ok = {key(r): r["Answer Correct"] == "Yes" for r in rows}
    fixed = regressed = 0
    for r in after_rows:
        if key(r) in before_ok:
            now = r["Answer Correct"] == "Yes"
            fixed += now and not before_ok[key(r)]
            regressed += (not now) and before_ok[key(r)]
    print(f"\nSame question, wrong -> right: {fixed} | right -> wrong: {regressed}")


# ---- held-out set: 250 new questions, asked once (plain wording)
held_rows = read("real_heldout_document_data.csv")
if held_rows:
    def hpct(items):
        return 100 * sum(r["Answer Correct"] == "Yes" for r in items) / len(items)

    print("\n" + "=" * 60)
    print(f"HELD-OUT (new questions): {hpct(held_rows):.1f}% ({len(held_rows)} runs judged)")
    for field, title in (("Question Type", "By question type"), ("Format", "By format"), ("Doc File", "By document")):
        groups = defaultdict(list)
        for r in held_rows:
            groups[r[field]].append(r)
        print(f"\n{title}")
        for name in sorted(groups, key=lambda n: hpct(groups[n])):
            items = groups[name]
            ok = sum(r["Answer Correct"] == "Yes" for r in items)
            print(f"  {name:<38} {ok:>3}/{len(items):<3} {hpct(items):5.1f}%")


# ---- held-out WITH the fixes vs held-out WITHOUT them (same 750 runs, switch AGENTOPS_NO_FIXES=1)
nofix_rows = read("real_heldout_nofix_document_data.csv")
if held_rows and nofix_rows:
    print("\n" + "=" * 60)
    print(f"HELD-OUT  without fixes: {hpct(nofix_rows):.1f}% ({len(nofix_rows)} runs)   with fixes: {hpct(held_rows):.1f}% ({len(held_rows)} runs)")
    for field, title in (("Question Type", "By question type"), ("Format", "By format"), ("Doc File", "By document")):
        a, b = defaultdict(list), defaultdict(list)
        for r in nofix_rows:
            a[r[field]].append(r)
        for r in held_rows:
            b[r[field]].append(r)
        print(f"\n{title}: without fixes -> with fixes")
        for name in sorted(a, key=lambda n: hpct(a[n])):
            if name in b:
                x, y = hpct(a[name]), hpct(b[name])
                print(f"  {name:<38} {x:5.1f}% -> {y:5.1f}%  ({y - x:+.1f})")
    base_ok = {(r["Doc File"], r["Question"]): r["Answer Correct"] == "Yes" for r in nofix_rows}
    up = down = 0
    for r in held_rows:
        k = (r["Doc File"], r["Question"])
        if k in base_ok:
            now = r["Answer Correct"] == "Yes"
            up += now and not base_ok[k]
            down += (not now) and base_ok[k]
    print(f"\nSame question, wrong -> right with fixes: {up} | right -> wrong: {down}")