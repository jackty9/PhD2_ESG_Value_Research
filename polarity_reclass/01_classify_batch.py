"""
Submits the polarity reclassification as an OpenAI Batch API job, mirroring
Section 6.11/6.12's exact pattern (locked prompt, JSON schema, async batch,
24h completion window). Must be run in Colab with an authenticated OpenAI
client -- NOT runnable in this sandbox (no API access here).

Population: Corporate-relevant, specificity tier >= 1 sentences from
esg_e_sg/data/df_esg_combined_specificity.csv (3,994 sentences as of this
project's current data -- re-verify this count when you run it, in case the
underlying file has changed).

Deliberately broader than the tier>=2 population the two regressors
(Positive_Specific_share, Negative_Specific_share) actually use -- per
instruction, tier>=1 is reclassified so more sentences get corrected labels,
but 03_rebuild_shares.py still filters to tier>=2 when building the two
share variables.

Run order (in Colab):
  1. Run 00_prompt_design.py's cell (or import from it) to get
     POLARITY_SYSTEM_PROMPT / polarity_schema.
  2. Run this script's cells in order: load population, write batch input,
     submit, poll status, download + parse output, merge back, save.
  3. Then 02_validation_sample.py to build the hand-coding sample.
  4. After manual coding is complete, 03_rebuild_shares.py for the kappa
     stats and share rebuild.
"""

import json as json_
from pathlib import Path

import pandas as pd

# ============================================================
# 1. Load the population: Corporate-relevant, tier >= 1
# ============================================================
data_dir = "/content/drive/MyDrive/phd/Data"  # adjust if your Drive layout differs
corpus_path = f"{data_dir}/df_ar_ceo_sentences_esgbert_scored.csv"  # or wherever
                                                                      # df_esg_combined_specificity's
                                                                      # source data lives on Drive

# NOTE: adjust this load to point at whatever file on your Drive holds the
# same columns as esg_e_sg/data/df_esg_combined_specificity.csv
# (corporate_relevance_pred, specificity_tier_pred, Sentence, Company Name,
# Year, sentence_id or an equivalent unique key). Re-verify the exact source
# file path before running -- this repo copy is a local snapshot.
df_all = pd.read_csv(corpus_path)

required_cols = ["Company Name", "Year", "Sentence", "corporate_relevance_pred", "specificity_tier_pred"]
missing_cols = [c for c in required_cols if c not in df_all.columns]
if missing_cols:
    raise ValueError(f"Source file missing required columns: {missing_cols}")

df_pop = df_all[
    (df_all["corporate_relevance_pred"] == "Corporate") & (df_all["specificity_tier_pred"] >= 1)
].copy().reset_index(drop=True)

df_pop["polarity_sentence_id"] = [f"pol_{i}" for i in range(len(df_pop))]

print(f"Population (Corporate, tier>=1): {len(df_pop)}")
print("(Expected 3,994 per this project's current data -- confirm this matches; if it does not, "
      "check whether the source corpus file is stale before submitting the batch job, same "
      "convention as Section 6.11's own expected-count check.)")


# ============================================================
# 2. Dependency guard -- same convention as 6.11.1b/6.12.1b
# ============================================================
_required_globals = ["client", "MODEL", "POLARITY_SYSTEM_PROMPT", "polarity_schema"]
_missing = [name for name in _required_globals if name not in globals()]
if _missing:
    raise NameError(
        f"Missing required objects: {_missing}. Run your existing OpenAI client/MODEL setup "
        "(same as Section 6.10.1) and import POLARITY_SYSTEM_PROMPT/polarity_schema from "
        "00_prompt_design.py first."
    )


# ============================================================
# 3. Batch I/O paths and prompt-building helper
# ============================================================
POLARITY_BATCH_SIZE = 20
POLARITY_TASK_NAME = "polarity_reclass_tier1plus"

polarity_output_dir = Path(data_dir) / "polarity_reclass_batch"
polarity_output_dir.mkdir(parents=True, exist_ok=True)


def build_polarity_user_prompt(batch_df):
    lines = ["Classify each of the following sentences.\n"]
    for _, row in batch_df.iterrows():
        lines.append(f'id={row["polarity_sentence_id"]} | sentence: "{row["Sentence"]}"')
    return "\n".join(lines)


# ============================================================
# 4. Write the Batch API input JSONL -- mirrors 6.11.3/6.12.3
# ============================================================
def write_polarity_batch_input(df, task_name, schema, system_prompt, batch_size=POLARITY_BATCH_SIZE):
    input_path = polarity_output_dir / f"{task_name}_batch_input.jsonl"
    metadata_path = polarity_output_dir / f"{task_name}_batch_metadata.csv"
    metadata_rows = []

    with open(input_path, "w", encoding="utf-8") as f:
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start + batch_size]
            custom_id = f"{task_name}_{start}_{start + len(batch_df) - 1}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "temperature": 0,
                    "top_p": 1,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_polarity_user_prompt(batch_df)},
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": f"{task_name}_classification",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                },
            }
            f.write(json_.dumps(request, ensure_ascii=False) + "\n")
            metadata_rows.append({
                "task_name": task_name, "custom_id": custom_id,
                "start_row": start, "end_row": start + len(batch_df) - 1,
                "n_sentences": len(batch_df),
            })

    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    print(f"Wrote {len(df)} sentences across {len(metadata_rows)} batch requests to {input_path}")
    return input_path


polarity_input_path = write_polarity_batch_input(
    df_pop, POLARITY_TASK_NAME, polarity_schema, POLARITY_SYSTEM_PROMPT
)


# ============================================================
# 5. Submit -- mirrors 6.11.4/6.12.4
# ============================================================
def submit_polarity_batch_job(task_name, input_path):
    uploaded_file = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=uploaded_file.id, endpoint="/v1/responses", completion_window="24h"
    )
    batch_id_path = polarity_output_dir / f"{task_name}_batch_id.txt"
    with open(batch_id_path, "w") as f:
        f.write(batch_job.id)
    print(f"Created batch job {batch_job.id}, status={batch_job.status}")
    print(f"Saved batch ID to {batch_id_path}")
    return batch_job


# polarity_batch_job = submit_polarity_batch_job(POLARITY_TASK_NAME, polarity_input_path)
# -- uncomment when ready to actually submit. Left commented so importing
# this module for the population count (Step above) doesn't accidentally
# fire the job.


# ============================================================
# 6. Status check / download / merge -- mirrors 6.11.5-6.11.7,
# 6.12.5-6.12.7 exactly. Run these only after the batch job completes
# (can take up to 24h).
# ============================================================
def check_polarity_batch_status(task_name=POLARITY_TASK_NAME):
    batch_id_path = polarity_output_dir / f"{task_name}_batch_id.txt"
    with open(batch_id_path, "r") as f:
        batch_id = f.read().strip()
    job = client.batches.retrieve(batch_id)
    print(f"status: {job.status}")
    print(f"request_counts: {job.request_counts}")
    return job


def download_polarity_batch_output(task_name=POLARITY_TASK_NAME):
    job = check_polarity_batch_status(task_name)
    if job.status != "completed":
        print(f"Not completed yet (status={job.status}). Check again later.")
        return None
    output_path = polarity_output_dir / f"{task_name}_batch_output.jsonl"
    if job.output_file_id:
        client.files.content(job.output_file_id).write_to_file(output_path)
        print(f"Saved output to {output_path}")
        return output_path
    print("No output_file_id present.")
    return None


def parse_polarity_batch_output(output_path):
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json_.loads(line)
            if item.get("error") is not None:
                continue
            body = item["response"]["body"]
            content = json_.loads(body["output"][0]["content"][0]["text"])
            for c in content["classifications"]:
                results.append(c)
    return pd.DataFrame(results)


def merge_polarity_results(df_pop, results_df):
    merged = df_pop.merge(
        results_df.rename(columns={"sentence_id": "polarity_sentence_id"}),
        on="polarity_sentence_id", how="left",
    )
    n_missing = merged["case"].isna().sum()
    print(f"Total: {len(merged)}, classified: {len(merged) - n_missing}, missing: {n_missing}")
    print("\nCase distribution:")
    print(merged["case"].value_counts(dropna=False))
    print("\nPolarity distribution (new GPT labels):")
    print(merged["polarity"].value_counts(dropna=False))
    output_path = f"{data_dir}/df_ar_ceo_sentences_polarity_reclass.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return merged
