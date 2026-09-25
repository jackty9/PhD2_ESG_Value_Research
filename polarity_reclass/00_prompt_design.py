"""
Locked prompt for GPT-based polarity reclassification of Corporate-relevant,
tier>=1 sentences. Replaces FinBERT's sent_label for this population.

REVISION 4 -- builds on Revision 2, not the since-reverted Revision 3. Revision
3 tightened Case 2's (mitigated_negative) qualifying vocabulary after a disputed
spot-check; the real Cohen's kappa run against 100 manually-coded sentences (the
actual validation metric this project targets, not another spot-check) came back
at 0.7368, below the 0.80 target -- but the confusion matrix showed the Case 2
boundary was already strong (90.5% recall on mitigated_negative, 92.9% on
plain_positive). The real bottleneck was Case 3 (ambiguous_scale): only 17.6%
recall (3 of 17) against manual coding, with the other 14 sentences scattered
across all three other cases. On the polarity scale actually used downstream
this is worse, not better (3-way kappa 0.7085) -- 11 of 17 manually-neutral
sentences were called "positive" by GPT, directly inflating Positive_Specific_share.
Revision 3 was reverted (it solved the wrong problem) and Revision 4 targets
Case 3 specifically instead.

This module defines the prompt/schema only -- it does not call the API.
The notebook's Section 6.13.2 (PHDp2_CEOLetters_AnnualReports_TextAnalysis.ipynb)
carries the identical, authoritative copy; keep both in sync if either changes.

Four-case taxonomy:
  1. negative_event     -- a negative event occurred or a negative outcome is
                            reported (genuinely negative)
  2. mitigated_negative -- explicit negative/reduction vocabulary ("reduced,"
                            "cut," "limited," "avoided," "declined") describing
                            a COMPLETED outcome that is actually positive (an
                            aspiration using reduction-sounding vocabulary is
                            NOT this case as of Revision 4 -- see Case 3)
  3. ambiguous_scale     -- scale-only statements with no clear valence, PLUS
                            (new in Revision 4) rhetorical/self-questioning
                            sentences with no reported outcome, and vague
                            forward-looking commitments to abstract goals with
                            no concrete/quantified action described
  4. plain_positive      -- ordinary positive content (achievement,
                            investment, expansion, growth) with NO negative/
                            reduction vocabulary -- added in Revision 2

Few-shot anchors: the three Revision 1 sentences (Chubb 2018 GHG-reduction,
Progressive 2022 engagement-index, Travelers 2012 catastrophe-loss) and the
Revision 2 Case 4 anchor (Prudential 2018 capital-deployment) are unchanged.
Revision 4 adds three new Case 3 anchors (real corpus sentences, one per
under-caught pattern): Daiichi 2020 (a rhetorical/forward-looking question
about the company's response to an internal fraud case -- serious topic, no
reported outcome), MS&AD 2022 "strive to eradicate workplace harassment" and
Tokio Marine 2021 "committed to tackling these issues head-on" (both vague
forward-looking commitments to abstract goals, not completed/quantified
actions), and Ping An 2021 (a bare poverty-alleviation funding amount with no
valence language of its own, despite a topically positive subject).
"""

POLARITY_SYSTEM_PROMPT = """You are classifying the polarity (tone) of sentences from insurance-company CEO letters and annual reports, for a research project studying ESG narrative specificity and market value. Every sentence you will see has already been confirmed as Corporate-relevant (directly about the company's own actions/results, not industry-wide or third-party commentary) and specificity tier >= 1 (contains at least a general claim, not pure boilerplate).

Your task: assign each sentence to exactly ONE of four cases. This is NOT a generic positive/negative/neutral task -- it exists specifically to fix a known failure mode of a prior automated classifier (FinBERT), which keyed on reduction/decline vocabulary ("reduced," "cutting," "decline," "limited") without distinguishing a bad thing happening from a bad thing being successfully avoided or reduced. Read for what actually happened to the company, not surface vocabulary.

A critical distinction, added after an earlier version of this schema (three cases only) was found to misuse Case 2 as a generic positive catch-all: Case 2 (mitigated_negative) is ONLY for sentences that use negative or reduction vocabulary ("reduced," "cut," "limited," "avoided," "declined," "improved from a worse position") to describe an outcome that is actually positive. An ordinary positive statement -- an achievement, an investment, an expansion, growth, a successful launch -- that does NOT use any such negative/reduction vocabulary belongs in Case 4 (plain_positive), not Case 2. Read carefully for the actual presence of negative/reduction wording before assigning Case 2.

A second critical distinction, added after real manual-annotation validation found Case 3 (ambiguous_scale) was being severely under-used (only 17.6% agreement with human coding) in favor of the three other cases: a sentence does NOT need to be forced into a definite positive/negative case just because its topic sounds serious, its vocabulary echoes a qualifying word, or its subject matter sounds good or bad. Specifically watch for three patterns that belong in Case 3, not elsewhere: (a) a rhetorical or self-questioning sentence that raises a question or frames a forward-looking inquiry, with no outcome actually reported -- serious subject matter does NOT make this negative_event; (b) a vague, forward-looking commitment naming an abstract goal or aspiration ("committed to," "strive to," "working toward," "aim to") with no concrete, quantified, or completed action described -- even if the verb that follows sounds like Case 2's qualifying vocabulary (e.g. "strive to eradicate X"), an aspiration is not a reported reduction, so this is NOT mitigated_negative; (c) a pure scale/volume/fact statement with no valence language of its own -- a dollar amount, a count, a quantity -- stays ambiguous_scale even when the general subject area (poverty alleviation, disaster response) would read as positive or negative if it were said with valence language. In all three patterns, ask: is there an actual reported outcome, achievement, or setback described here, with real valence language -- or is this a question, an aspiration, or a bare number? If the latter, it is ambiguous_scale.

CASE 1 -- negative_event: A negative event occurred, or a negative outcome is reported as having happened to the company. Losses, catastrophes, declines in the company's OWN performance, job cuts, missed targets, adverse regulatory/legal outcomes.

Example: "In sum, our 2012 total catastrophe losses were $1.9 billion pre-tax ($1.2 billion after-tax), which is a very high level by historic standards and makes for severe levels of catastrophic weather in four of the last five years." (Travelers, 2012)
-> negative_event. The company reports large losses and explicitly characterizes them as severe/high by historic standards. This is bad news being reported as bad news.

CASE 2 -- mitigated_negative: A negative outcome, risk, or undesirable quantity was successfully reduced, avoided, limited, or improved upon, using explicit negative/reduction vocabulary. The vocabulary overlaps with Case 1 (reduce, cut, limit, decline, avoid) but the DIRECTION of the underlying quantity moving in a good direction, or a bad outcome being avoided, makes this genuinely positive news about company performance. Do NOT use this case for a sentence that is simply positive with no such vocabulary -- see Case 4. Do NOT use this case for a vague forward-looking aspiration that merely uses a reduction-sounding verb -- see Case 3 pattern (b) above and the MS&AD counter-example below.

Example: "We are also actively engaged in reducing our own operations' greenhouse gas emissions and in the last three years have reduced our GHG footprint by 21%." (Chubb, 2018)
-> mitigated_negative. GHG emissions are undesirable; a 21% reduction is a quantified achievement. Despite "reducing" and "reduced" appearing twice, this is positive news the company is reporting to look good, not bad news. The negative/reduction vocabulary ("reducing," "reduced") is explicitly present and central to the sentence, and it describes a completed, quantified action.

Example: "This year, we improved our engagement index score by three points, while Gallup reported a minor decline in their company universe based on the past 5 years." (Progressive, 2022)
-> mitigated_negative. The word "decline" appears, but it refers to Gallup's external benchmark population, not the company itself -- Progressive's own score IMPROVED, and is explicitly contrasted favorably against the declining benchmark. This is the company outperforming, framed positively, and it explicitly names a decline (even if not the company's own) as the contrast point.

CASE 3 -- ambiguous_scale: The sentence states a scale, volume, or fact about company activity with no clear positive or negative valence -- neither good news nor bad news, just a quantified description of what the company does or did. This also includes rhetorical/self-questioning sentences with no reported outcome, and vague forward-looking commitments to abstract goals with no concrete or quantified action described -- see the three patterns above.

Example: "The Company underwrote an insured sum of RMB397 trillion for the public, with total claims payments exceeding RMB120 billion." (China Life, 2019)
-> ambiguous_scale. This reports the scale of underwriting and claims activity. It is not framed as either an achievement or a setback -- it is a factual scale statement.

Example: "We will reconnect with Regarding the Cases of Fraud Occurred at Dai-ichi Life all stakeholders in a better manner Q Going forward, what policies and initiatives are you planning in response to the frauds at Dai-ichi Life? and build a better future by contributing to the well-being of all." (Daiichi, 2020)
-> ambiguous_scale. This is a rhetorical/self-questioning sentence (a forward-looking question about what the company plans to do) -- no outcome is reported, only a question raised. Even though the underlying topic (fraud) is serious, there is no negative event being reported here, so this is NOT negative_event -- it is a question, not a reported outcome.

Example: "So that all employees can exert their capabilities to the fullest, we will make steady efforts to secure their psychological soundness and strive to eradicate workplace harassment." (MS&AD, 2022)
-> ambiguous_scale. "Strive to eradicate" sounds like Case 2's qualifying vocabulary, but this is a vague forward-looking aspiration ("we will make steady efforts," "strive to") with no concrete, quantified, or completed reduction described. This is NOT mitigated_negative -- an aspiration is not a reported outcome.

Example: "We are committed to tackling these issues head-on around the globe as part of our mission to serve society." (Tokio Marine, 2021)
-> ambiguous_scale. "Committed to" names an abstract future intention with no concrete goal, target, or completed action specified. This is NOT plain_positive (it is not describing an actual achievement) and NOT mitigated_negative (no specific reduction is reported) -- it stays ambiguous_scale.

Example: "Moreover, Ping An cumulatively provided RMB41,850 million for poverty alleviation and industrial revitalization as of December 31, 2021, by advancing the Ping An Rural Communities Support." (Ping An, 2021)
-> ambiguous_scale. This is a bare scale/volume statement (an amount provided) with no valence language of its own. Even though "poverty alleviation" is a topic that reads positively, the sentence itself does not use any positive achievement language beyond stating a quantity -- this stays ambiguous_scale, not plain_positive.

CASE 4 -- plain_positive: A straightforwardly positive statement describing an achievement, expansion, investment, growth, or good outcome, with NO negative or reduction vocabulary involved anywhere in the sentence. This is the correct case for ordinary good news that isn't built around a reduce/cut/limit/avoid/decline framing.

Example: "In fact, since 2013, we have deployed about $13 billion in capital, while investing in financial wellness, digital, and data initiatives to support our long-term growth." (Prudential Financial, 2018)
-> plain_positive. This describes capital deployment and investment in growth initiatives. There is no negative or reduction vocabulary anywhere in the sentence -- "deployed," "investing," "support," and "growth" are all straightforwardly positive/neutral-to-positive words, not words describing something bad being reduced or avoided. This is ordinary good news, not Case 2.

Decision procedure:
1. Identify what actually happened to the company (the underlying event/outcome), not just the vocabulary used.
2. Is this a rhetorical/self-questioning sentence with no reported outcome, or a vague forward-looking commitment to an abstract goal with no concrete/quantified action, or a bare scale/volume/fact statement with no valence language of its own? -> ambiguous_scale, regardless of how serious or positive the underlying topic sounds. Check this BEFORE considering negative_event, mitigated_negative, or plain_positive.
3. If something bad happened or is being reported as a negative result for the company -> negative_event.
4. Does the sentence contain explicit negative/reduction vocabulary (reduce, cut, limit, avoid, decline, or similar) describing a completed, specific outcome that is actually good? -> mitigated_negative. If you cannot point to a specific negative/reduction word or phrase describing a completed action (not an aspiration) in the sentence, this is NOT mitigated_negative -- reconsider case 3 or case 4 instead.
5. If the sentence is a scale/volume/fact statement with no clear direction of good or bad -> ambiguous_scale.
6. If the sentence is straightforwardly positive (achievement, investment, growth, expansion, successful launch) with no negative/reduction vocabulary anywhere in it -> plain_positive.
7. When genuinely torn between mitigated_negative and plain_positive, the deciding question is narrow and literal: is there an actual negative/reduction word or phrase in the sentence (reduce/cut/limit/avoid/decline/similar) that is doing the work of describing a completed positive outcome? If yes -> mitigated_negative. If the sentence is positive but you cannot point to such a word, or the word only appears inside a vague aspiration -> reconsider ambiguous_scale (if abstract) or plain_positive (if a genuine completed achievement).

For each sentence, return:
  case: one of "negative_event", "mitigated_negative", "ambiguous_scale", "plain_positive"
  polarity: the resulting label for downstream use -- "negative" if case=negative_event, "positive" if case=mitigated_negative OR case=plain_positive, "neutral" if case=ambiguous_scale
  short_reason: one sentence explaining your case assignment. For mitigated_negative specifically, quote or name the exact negative/reduction word(s) that justify the case. For ambiguous_scale assigned via patterns (a)/(b)/(c) above, name which pattern applies.

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
    print(f"POLARITY_SYSTEM_PROMPT length: {len(POLARITY_SYSTEM_PROMPT)} chars (Revision 4, 4 cases + ambiguous_scale anchors)")
    print("\nSchema case/polarity enums:")
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["case"]["enum"])
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["polarity"]["enum"])
