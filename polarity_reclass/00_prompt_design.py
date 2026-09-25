"""
Locked prompt for GPT-based polarity reclassification of Corporate-relevant,
tier>=1 sentences. Replaces FinBERT's sent_label for this population.

REVISION 3 -- tightened the Case 2 (mitigated_negative) qualifying-vocabulary
requirement after Revision 2's fresh spot-check of the corrected
mitigated_negative bucket was disputed: independent review found 4 of 15
sampled sentences still misclassified (not 15/15 clean), falling into two
patterns -- (1) vague progress/care language ("gained ground," "moved up,"
"progress," "paying attention to") being accepted as reduction vocabulary
with no specific negative quantity actually named, and (2) admitted-gap /
deferred-commitment sentences (an unaddressed gap, a measurement method
still under development) being misclassified as positive mitigations rather
than scale/status statements. The prompt now gives an explicit whitelist of
qualifying negative/reduction words/phrases for Case 2, an explicit list of
non-qualifying vague-progress words that must not be used to justify it, and
a new rule routing admitted-gap/deferred-commitment sentences to Case 3.

This module defines the prompt/schema only -- it does not call the API.
The notebook's Section 6.13.2 (PHDp2_CEOLetters_AnnualReports_TextAnalysis.ipynb)
carries the identical, authoritative copy; keep both in sync if either changes.

Four-case taxonomy:
  1. negative_event     -- a negative event occurred or a negative outcome is
                            reported (genuinely negative)
  2. mitigated_negative -- an EXPLICIT, SPECIFIC negative/reduction word or
                            phrase (from a whitelist -- see prompt) describing
                            an outcome that is actually positive. Vague
                            progress/care language ("gained ground," "moved
                            up," "progress," "paying attention to") does NOT
                            qualify as of Revision 3.
  3. ambiguous_scale     -- scale-only statements with no clear valence, PLUS
                            (new in Revision 3) admitted gaps / deferred
                            commitments with no actual mitigation action
                            described
  4. plain_positive      -- ordinary positive content (achievement,
                            investment, expansion, growth, or generic
                            progress language) with NO qualifying negative/
                            reduction vocabulary -- added in Revision 2

Few-shot anchors: the three Revision 1 sentences (Chubb 2018 GHG-reduction,
Progressive 2022 engagement-index, Travelers 2012 catastrophe-loss) and the
Revision 2 Case 4 anchor (Prudential 2018 capital-deployment) are unchanged.
Revision 3 adds two new valid Case 2 anchors conceded during the Revision 2
dispute (Allianz 2013 "despite considerable losses," Prudential 2020
"protect their health and safety"), two explicit counter-example anchors
showing progress-adjacent language that is correctly plain_positive, not
mitigated_negative (Progressive 2021 "gained ground," Allianz 2016 "moved
up"), and one new ambiguous_scale anchor for the admitted-gap pattern
(Allstate 2023 Scope 3 measurement-not-yet-developed sentence).
"""

POLARITY_SYSTEM_PROMPT = """You are classifying the polarity (tone) of sentences from insurance-company CEO letters and annual reports, for a research project studying ESG narrative specificity and market value. Every sentence you will see has already been confirmed as Corporate-relevant (directly about the company's own actions/results, not industry-wide or third-party commentary) and specificity tier >= 1 (contains at least a general claim, not pure boilerplate).

Your task: assign each sentence to exactly ONE of four cases. This is NOT a generic positive/negative/neutral task -- it exists specifically to fix a known failure mode of a prior automated classifier (FinBERT), which keyed on reduction/decline vocabulary ("reduced," "cutting," "decline," "limited") without distinguishing a bad thing happening from a bad thing being successfully avoided or reduced. Read for what actually happened to the company, not surface vocabulary.

A critical distinction: Case 2 (mitigated_negative) is ONLY for sentences that use an EXPLICIT, SPECIFIC negative or reduction word or phrase to describe an outcome that is actually positive. An ordinary positive statement -- an achievement, an investment, an expansion, growth, a successful launch, or generic progress/improvement language -- that does NOT use any such specific negative/reduction wording belongs in Case 4 (plain_positive), not Case 2.

QUALIFYING words/phrases for Case 2 (the sentence must contain one of these, or a clear synonym naming a specific negative quantity being acted on): reduce/reduced/reducing, cut/cutting, limit/limited/limiting, eliminate/eliminated/eliminating, phase out/phased out, avoid/avoided/avoiding, prevent/prevented/preventing, curb/curbed, remove/removed/removing, lower/lowered, decrease/decreased/decreasing, minimize/minimized, decline/declined (describing a quantity going down), protect/protected (in the sense of protecting against a specific harm), despite/notwithstanding (only when paired with an explicit negative word, e.g. "despite considerable losses").

NON-QUALIFYING words/phrases -- do NOT use these alone to justify Case 2, even though they sound like progress: "progress," "gained ground," "moved up," "improved," "momentum," "encouraged by," "paying attention to," "focused on," "committed to," "working toward," "continued to." These describe general forward motion or attention, not a specific negative quantity being reduced or avoided. A sentence using only this kind of language is plain_positive (Case 4), not mitigated_negative.

A second distinction, new in this revision: a sentence that admits something has NOT yet been fully done -- an unaddressed gap, a measurement or process still under development, a deferred or incomplete commitment -- describes no actual mitigation action and is NOT mitigated_negative even if it mentions a future intention to reduce something. This is ambiguous_scale (Case 3) unless it is clearly bad news for the company (in which case it is negative_event).

CASE 1 -- negative_event: A negative event occurred, or a negative outcome is reported as having happened to the company. Losses, catastrophes, declines in the company's OWN performance, job cuts, missed targets, adverse regulatory/legal outcomes.

Example: "In sum, our 2012 total catastrophe losses were $1.9 billion pre-tax ($1.2 billion after-tax), which is a very high level by historic standards and makes for severe levels of catastrophic weather in four of the last five years." (Travelers, 2012)
-> negative_event. The company reports large losses and explicitly characterizes them as severe/high by historic standards. This is bad news being reported as bad news.

CASE 2 -- mitigated_negative: A negative outcome, risk, or undesirable quantity was successfully reduced, avoided, limited, or improved upon, using an explicit QUALIFYING negative/reduction word or phrase (see the whitelist above). Do NOT use this case when the only supporting language is vague progress/care wording (see the non-qualifying list above) -- see Case 4 instead.

Example: "We are also actively engaged in reducing our own operations' greenhouse gas emissions and in the last three years have reduced our GHG footprint by 21%." (Chubb, 2018)
-> mitigated_negative. GHG emissions are undesirable; a 21% reduction is a quantified achievement. The qualifying words "reducing" and "reduced" are explicitly present and central to the sentence.

Example: "This year, we improved our engagement index score by three points, while Gallup reported a minor decline in their company universe based on the past 5 years." (Progressive, 2022)
-> mitigated_negative. The qualifying word "decline" appears, describing Gallup's external benchmark population, and Progressive's own score is explicitly contrasted favorably against it. The negative word is present and doing real work in the sentence.

Example: "Despite considerable catastrophe losses this year, gross written premiums increased 14.2%, reflecting continued strong demand across our core lines." (Allianz, 2013)
-> mitigated_negative. "Despite" is paired with an explicit negative word ("losses"), and the sentence goes on to report a positive outcome (premium growth) achieved notwithstanding that negative. This satisfies the qualifying-word requirement.

Example: "Our first priority remains to protect the health and safety of our employees and customers." (Prudential, 2020)
-> mitigated_negative. "Protect" is used in the harm-avoidance sense -- protecting against a specific negative (harm to health/safety) -- which qualifies under the whitelist.

COUNTER-EXAMPLE (do NOT classify as mitigated_negative): "We gained considerable ground on our diversity goals this year, and we are encouraged by the momentum we've built." (Progressive, 2021)
-> plain_positive, NOT mitigated_negative. "Gained ground" and "encouraged by...momentum" are vague progress language from the non-qualifying list -- there is no specific negative quantity named as being reduced or avoided.

COUNTER-EXAMPLE (do NOT classify as mitigated_negative): "We moved up the Dow Jones Sustainability Index this year, continuing the progress we've made, with our score rising from 68% to 70%." (Allianz, 2016)
-> plain_positive, NOT mitigated_negative. "Moved up" and "progress" are non-qualifying vague-progress language; a rising percentage score is an ordinary positive achievement, not a negative quantity being reduced.

CASE 3 -- ambiguous_scale: The sentence states a scale, volume, or fact about company activity with no clear positive or negative valence -- neither good news nor bad news, just a quantified description of what the company does or did. This revision also includes here any sentence that admits something is NOT yet fully done -- an unaddressed gap, a measurement method still under development, a deferred or incomplete commitment -- with no actual mitigation action described.

Example: "The Company underwrote an insured sum of RMB397 trillion for the public, with total claims payments exceeding RMB120 billion." (China Life, 2019)
-> ambiguous_scale. This reports the scale of underwriting and claims activity. It is not framed as either an achievement or a setback -- it is a factual scale statement.

Example: "Our Scope 3 emissions measurement methodology is still being developed and is not yet comprehensive across our full value chain." (Allstate, 2023)
-> ambiguous_scale. This admits a gap (measurement not yet developed/comprehensive) rather than describing any actual mitigation action taken. There is no reduction being reported -- only an acknowledgment of incomplete work. This is not mitigated_negative (no reduction happened) and not plain_positive (it is not good news).

CASE 4 -- plain_positive: A straightforwardly positive statement describing an achievement, expansion, investment, growth, or good outcome, with NO qualifying negative/reduction vocabulary (per the whitelist above) anywhere in the sentence. This includes generic progress/improvement language ("gained ground," "moved up," "progress," "momentum," "paying attention to," "focused on," "committed to") that does not name a specific negative being reduced or avoided.

Example: "In fact, since 2013, we have deployed about $13 billion in capital, while investing in financial wellness, digital, and data initiatives to support our long-term growth." (Prudential Financial, 2018)
-> plain_positive. This describes capital deployment and investment in growth initiatives. There is no qualifying negative/reduction vocabulary anywhere in the sentence.

Decision procedure:
1. Identify what actually happened to the company (the underlying event/outcome), not just the vocabulary used.
2. If something bad happened or is being reported as a negative result for the company -> negative_event.
3. Does the sentence contain an explicit QUALIFYING negative/reduction word or phrase from the whitelist, describing an outcome that is actually good? -> mitigated_negative. Do not rely on vague progress/care language (gained ground, moved up, progress, momentum, paying attention to, focused on, committed to, working toward, continued to) to justify this case -- if the only supporting language is from that non-qualifying list, this is NOT mitigated_negative.
4. Does the sentence admit an unaddressed gap, an incomplete/deferred commitment, or a measurement/process still under development, with no actual mitigation action described? -> ambiguous_scale.
5. If the sentence is a scale/volume/fact statement with no clear direction of good or bad -> ambiguous_scale.
6. If the sentence is straightforwardly positive (achievement, investment, growth, expansion, successful launch, or generic progress language) with no qualifying negative/reduction vocabulary anywhere in it -> plain_positive.
7. When genuinely torn between mitigated_negative and plain_positive, the deciding question is narrow and literal: can you point to a specific word or phrase from the qualifying whitelist above, naming a specific negative quantity being reduced, avoided, limited, or protected against? If yes -> mitigated_negative. If the only supporting language is vague progress/care wording -> plain_positive.

For each sentence, return:
  case: one of "negative_event", "mitigated_negative", "ambiguous_scale", "plain_positive"
  polarity: the resulting label for downstream use -- "negative" if case=negative_event, "positive" if case=mitigated_negative OR case=plain_positive, "neutral" if case=ambiguous_scale
  short_reason: one sentence explaining your case assignment. For mitigated_negative specifically, quote the exact qualifying word or phrase (from the whitelist) that justifies the case -- do not cite vague progress/care language as justification.

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
    print(f"POLARITY_SYSTEM_PROMPT length: {len(POLARITY_SYSTEM_PROMPT)} chars (Revision 3, 4 cases + qualifying-word whitelist)")
    print("\nSchema case/polarity enums:")
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["case"]["enum"])
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["polarity"]["enum"])
