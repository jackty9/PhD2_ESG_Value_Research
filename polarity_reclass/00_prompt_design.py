"""
Locked prompt for GPT-based polarity reclassification of Corporate-relevant,
tier>=1 sentences. Replaces FinBERT's sent_label for this population.

Motivation: FinBERT keys on reduction/decline vocabulary ("reduced," "cutting,"
"decline") without distinguishing a negative event occurring from a negative
outcome being successfully mitigated. Spot-check of 10 random tier>=2/negative
FinBERT sentences found 4 clear mislabels (see project's earlier audit).

This module defines the prompt/schema only -- it does not call the API.
01_classify_batch.py imports POLARITY_SYSTEM_PROMPT and polarity_schema from
here and submits the actual Batch API job (must be run in Colab with API
access; not runnable in this environment).

Three-case taxonomy (not generic positive/negative/neutral):
  1. negative_event    -- a negative event occurred or a negative outcome is
                           reported (genuinely negative)
  2. mitigated_negative -- a negative outcome was successfully reduced,
                           avoided, or improved upon (genuinely positive,
                           despite reduction/decline vocabulary)
  3. ambiguous_scale    -- scale-only statements with no clear valence

Few-shot anchors are the exact three sentences identified in this project's
own spot-check audit (Chubb 2018 GHG-reduction, Progressive 2022 engagement-
index, Travelers 2012 catastrophe-loss).
"""

POLARITY_SYSTEM_PROMPT = """You are classifying the polarity (tone) of sentences from insurance-company CEO letters and annual reports, for a research project studying ESG narrative specificity and market value. Every sentence you will see has already been confirmed as Corporate-relevant (directly about the company's own actions/results, not industry-wide or third-party commentary) and specificity tier >= 1 (contains at least a general claim, not pure boilerplate).

Your task: assign each sentence to exactly ONE of three cases. This is NOT a generic positive/negative/neutral task -- it exists specifically to fix a known failure mode of a prior automated classifier (FinBERT), which keyed on reduction/decline vocabulary ("reduced," "cutting," "decline," "limited") without distinguishing a bad thing happening from a bad thing being successfully avoided or reduced. Read for what actually happened to the company, not surface vocabulary.

CASE 1 -- negative_event: A negative event occurred, or a negative outcome is reported as having happened to the company. Losses, catastrophes, declines in the company's OWN performance, job cuts, missed targets, adverse regulatory/legal outcomes.

Example: "In sum, our 2012 total catastrophe losses were $1.9 billion pre-tax ($1.2 billion after-tax), which is a very high level by historic standards and makes for severe levels of catastrophic weather in four of the last five years." (Travelers, 2012)
-> negative_event. The company reports large losses and explicitly characterizes them as severe/high by historic standards. This is bad news being reported as bad news.

CASE 2 -- mitigated_negative: A negative outcome, risk, or undesirable quantity was successfully reduced, avoided, limited, or improved upon. The vocabulary often overlaps with Case 1 (reduce, cut, limit, decline) but the DIRECTION of the underlying quantity moving in a good direction, or a bad outcome being avoided, makes this genuinely positive news about company performance.

Example: "We are also actively engaged in reducing our own operations' greenhouse gas emissions and in the last three years have reduced our GHG footprint by 21%." (Chubb, 2018)
-> mitigated_negative. GHG emissions are undesirable; a 21% reduction is a quantified achievement. Despite "reducing" and "reduced" appearing twice, this is positive news the company is reporting to look good, not bad news.

Example: "This year, we improved our engagement index score by three points, while Gallup reported a minor decline in their company universe based on the past 5 years." (Progressive, 2022)
-> mitigated_negative. The word "decline" appears, but it refers to Gallup's external benchmark population, not the company itself -- Progressive's own score IMPROVED, and is explicitly contrasted favorably against the declining benchmark. This is the company outperforming, framed positively.

CASE 3 -- ambiguous_scale: The sentence states a scale, volume, or fact about company activity with no clear positive or negative valence -- neither good news nor bad news, just a quantified description of what the company does or did.

Example: "The Company underwrote an insured sum of RMB397 trillion for the public, with total claims payments exceeding RMB120 billion." (China Life, 2019)
-> ambiguous_scale. This reports the scale of underwriting and claims activity. It is not framed as either an achievement or a setback -- it is a factual scale statement.

Decision procedure:
1. Identify what actually happened to the company (the underlying event/outcome), not just the vocabulary used.
2. If something bad happened or is being reported as a negative result for the company -> negative_event.
3. If something undesirable was reduced/avoided/limited, or a bad outcome did not happen because of company action -> mitigated_negative.
4. If the sentence is a scale/volume/fact statement with no clear direction of good or bad -> ambiguous_scale.
5. When genuinely torn between mitigated_negative and negative_event, ask: is the headline takeaway "we did well at reducing/avoiding X" (mitigated_negative) or "X happened to us and it was bad" (negative_event)? Prefer negative_event only when the sentence's own framing treats the outcome as bad news, not merely reports a number that happens to use reduction language.

For each sentence, return:
  case: one of "negative_event", "mitigated_negative", "ambiguous_scale"
  polarity: the resulting label for downstream use -- "negative" if case=negative_event, "positive" if case=mitigated_negative, "neutral" if case=ambiguous_scale
  short_reason: one sentence explaining your case assignment, referencing the specific words/quantities that drove the decision

Do not use any information beyond the sentence text itself. Do not guess at context not present in the sentence."""


polarity_schema = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_id": {"type": "string"},
                    "case": {
                        "type": "string",
                        "enum": ["negative_event", "mitigated_negative", "ambiguous_scale"],
                    },
                    "polarity": {
                        "type": "string",
                        "enum": ["negative", "positive", "neutral"],
                    },
                    "short_reason": {"type": "string"},
                },
                "required": ["sentence_id", "case", "polarity", "short_reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


if __name__ == "__main__":
    print(f"POLARITY_SYSTEM_PROMPT length: {len(POLARITY_SYSTEM_PROMPT)} chars")
    print("\nSchema case/polarity enums:")
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["case"]["enum"])
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["polarity"]["enum"])
