"""
Locked prompt for GPT-based polarity reclassification of Corporate-relevant,
tier>=1 sentences. Replaces FinBERT's sent_label for this population.

REVISION 2 -- added Case 4 (plain_positive) after Revision 1's spot-check
found 14 of 15 sampled "mitigated_negative" sentences were plain positive
statements with no negative/reduction vocabulary at all -- Case 2 had
become a generic positive catch-all rather than specifically capturing the
FinBERT reduction/decline-vocabulary failure mode it was built to fix.

This module defines the prompt/schema only -- it does not call the API.
The notebook's Section 6.13.2 (PHDp2_CEOLetters_AnnualReports_TextAnalysis.ipynb)
carries the identical, authoritative copy; keep both in sync if either changes.

Four-case taxonomy:
  1. negative_event     -- a negative event occurred or a negative outcome is
                            reported (genuinely negative)
  2. mitigated_negative -- explicit negative/reduction vocabulary ("reduced,"
                            "cut," "limited," "avoided," "declined")
                            describing an outcome that is actually positive
  3. ambiguous_scale     -- scale-only statements with no clear valence
  4. plain_positive      -- ordinary positive content (achievement,
                            investment, expansion, growth) with NO negative/
                            reduction vocabulary -- added in Revision 2

Few-shot anchors: the three original sentences (Chubb 2018 GHG-reduction,
Progressive 2022 engagement-index, Travelers 2012 catastrophe-loss) plus a
fourth for Case 4 (Prudential 2018 capital-deployment), drawn directly from
Revision 1's own spot-check -- one of the 14 confirmed misclassifications.
"""

POLARITY_SYSTEM_PROMPT = """You are classifying the polarity (tone) of sentences from insurance-company CEO letters and annual reports, for a research project studying ESG narrative specificity and market value. Every sentence you will see has already been confirmed as Corporate-relevant (directly about the company's own actions/results, not industry-wide or third-party commentary) and specificity tier >= 1 (contains at least a general claim, not pure boilerplate).

Your task: assign each sentence to exactly ONE of four cases. This is NOT a generic positive/negative/neutral task -- it exists specifically to fix a known failure mode of a prior automated classifier (FinBERT), which keyed on reduction/decline vocabulary ("reduced," "cutting," "decline," "limited") without distinguishing a bad thing happening from a bad thing being successfully avoided or reduced. Read for what actually happened to the company, not surface vocabulary.

A critical distinction, added after an earlier version of this schema (three cases only) was found to misuse Case 2 as a generic positive catch-all: Case 2 (mitigated_negative) is ONLY for sentences that use negative or reduction vocabulary ("reduced," "cut," "limited," "avoided," "declined," "improved from a worse position") to describe an outcome that is actually positive. An ordinary positive statement -- an achievement, an investment, an expansion, growth, a successful launch -- that does NOT use any such negative/reduction vocabulary belongs in Case 4 (plain_positive), not Case 2. Read carefully for the actual presence of negative/reduction wording before assigning Case 2.

CASE 1 -- negative_event: A negative event occurred, or a negative outcome is reported as having happened to the company. Losses, catastrophes, declines in the company's OWN performance, job cuts, missed targets, adverse regulatory/legal outcomes.

Example: "In sum, our 2012 total catastrophe losses were $1.9 billion pre-tax ($1.2 billion after-tax), which is a very high level by historic standards and makes for severe levels of catastrophic weather in four of the last five years." (Travelers, 2012)
-> negative_event. The company reports large losses and explicitly characterizes them as severe/high by historic standards. This is bad news being reported as bad news.

CASE 2 -- mitigated_negative: A negative outcome, risk, or undesirable quantity was successfully reduced, avoided, limited, or improved upon, using explicit negative/reduction vocabulary. The vocabulary overlaps with Case 1 (reduce, cut, limit, decline, avoid) but the DIRECTION of the underlying quantity moving in a good direction, or a bad outcome being avoided, makes this genuinely positive news about company performance. Do NOT use this case for a sentence that is simply positive with no such vocabulary -- see Case 4.

Example: "We are also actively engaged in reducing our own operations' greenhouse gas emissions and in the last three years have reduced our GHG footprint by 21%." (Chubb, 2018)
-> mitigated_negative. GHG emissions are undesirable; a 21% reduction is a quantified achievement. Despite "reducing" and "reduced" appearing twice, this is positive news the company is reporting to look good, not bad news. The negative/reduction vocabulary ("reducing," "reduced") is explicitly present and central to the sentence.

Example: "This year, we improved our engagement index score by three points, while Gallup reported a minor decline in their company universe based on the past 5 years." (Progressive, 2022)
-> mitigated_negative. The word "decline" appears, but it refers to Gallup's external benchmark population, not the company itself -- Progressive's own score IMPROVED, and is explicitly contrasted favorably against the declining benchmark. This is the company outperforming, framed positively, and it explicitly names a decline (even if not the company's own) as the contrast point.

CASE 3 -- ambiguous_scale: The sentence states a scale, volume, or fact about company activity with no clear positive or negative valence -- neither good news nor bad news, just a quantified description of what the company does or did.

Example: "The Company underwrote an insured sum of RMB397 trillion for the public, with total claims payments exceeding RMB120 billion." (China Life, 2019)
-> ambiguous_scale. This reports the scale of underwriting and claims activity. It is not framed as either an achievement or a setback -- it is a factual scale statement.

CASE 4 -- plain_positive: A straightforwardly positive statement describing an achievement, expansion, investment, growth, or good outcome, with NO negative or reduction vocabulary involved anywhere in the sentence. This is the correct case for ordinary good news that isn't built around a reduce/cut/limit/avoid/decline framing.

Example: "In fact, since 2013, we have deployed about $13 billion in capital, while investing in financial wellness, digital, and data initiatives to support our long-term growth." (Prudential Financial, 2018)
-> plain_positive. This describes capital deployment and investment in growth initiatives. There is no negative or reduction vocabulary anywhere in the sentence -- "deployed," "investing," "support," and "growth" are all straightforwardly positive/neutral-to-positive words, not words describing something bad being reduced or avoided. This is ordinary good news, not Case 2.

Decision procedure:
1. Identify what actually happened to the company (the underlying event/outcome), not just the vocabulary used.
2. If something bad happened or is being reported as a negative result for the company -> negative_event.
3. Does the sentence contain explicit negative/reduction vocabulary (reduce, cut, limit, avoid, decline, or similar) describing an outcome that is actually good? -> mitigated_negative. If you cannot point to a specific negative/reduction word or phrase in the sentence, this is NOT mitigated_negative -- reconsider case 4 instead.
4. If the sentence is a scale/volume/fact statement with no clear direction of good or bad -> ambiguous_scale.
5. If the sentence is straightforwardly positive (achievement, investment, growth, expansion, successful launch) with no negative/reduction vocabulary anywhere in it -> plain_positive.
6. When genuinely torn between mitigated_negative and plain_positive, the deciding question is narrow and literal: is there an actual negative/reduction word or phrase in the sentence (reduce/cut/limit/avoid/decline/similar) that is doing the work of describing the positive outcome? If yes -> mitigated_negative. If the sentence is positive but you cannot point to such a word -> plain_positive.

For each sentence, return:
  case: one of "negative_event", "mitigated_negative", "ambiguous_scale", "plain_positive"
  polarity: the resulting label for downstream use -- "negative" if case=negative_event, "positive" if case=mitigated_negative OR case=plain_positive, "neutral" if case=ambiguous_scale
  short_reason: one sentence explaining your case assignment. For mitigated_negative specifically, quote or name the exact negative/reduction word(s) that justify the case.

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
                        "enum": ["negative_event", "mitigated_negative", "ambiguous_scale", "plain_positive"]
                    },
                    "polarity": {
                        "type": "string",
                        "enum": ["negative", "positive", "neutral"]
                    },
                    "short_reason": {"type": "string"}
                },
                "required": ["sentence_id", "case", "polarity", "short_reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["classifications"],
    "additionalProperties": False
}




if __name__ == "__main__":
    print(f"POLARITY_SYSTEM_PROMPT length: {len(POLARITY_SYSTEM_PROMPT)} chars (Revision 2, 4 cases)")
    print("\nSchema case/polarity enums:")
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["case"]["enum"])
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["polarity"]["enum"])
