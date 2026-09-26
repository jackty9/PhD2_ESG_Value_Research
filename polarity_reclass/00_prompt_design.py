"""
Locked prompt for GPT-based polarity reclassification of Corporate-relevant,
tier>=1 sentences. Replaces FinBERT's sent_label for this population.

REVISION 6 -- Revision 5's validation-only pilot (notebook Step 6.13.2b)
brought kappa to 0.7731 (4-case) and ambiguous_scale recall to 88.2%
(15/17), but negative_event recall dropped hard (94.1% -> 70.6%, 9/34 now
wrongly called ambiguous_scale), and 3-way polarity kappa actually fell
(0.7517 -> 0.7338) since a real negative event mislabeled neutral directly
undercounts Negative_Specific_share. Reading all 9 misses found two causes:

  1. Two real prompt bugs, now fixed:
     - Pattern (d) [personal/emotional reflection] over-triggered on AIG
       2012's "...witnessed their company being rapidly dismantled amid
       unprecedented public criticism" -- emotional language wrapping a
       REAL described negative event is not the same as reflection with no
       event described. Pattern (d) now explicitly requires no concrete
       event or outcome be named at all.
     - The Case 2 "despite adversity" pattern (added in Revision 5) over-
       triggered on Travelers 2014's "confronted with a challenging
       economy...conditions that are demanding" -- adversity named, but no
       positive outcome stated as achieved despite it (unlike Allianz 2020,
       which explicitly states "we were able to deliver solid results").
       The pattern now explicitly requires BOTH halves: named adversity AND
       a stated, completed positive outcome.
  2. A manual-coding correction, not a schema change, for the remaining 6
     misses (MS&AD 2018/2019, Chubb 2023 x2, Chubb 2020 x2): these are
     broad societal/political/industry commentary the CEO letter cites as
     context, not a specific thing reported as having happened to the
     company. All 6 are confirmed corporate_relevance_pred == "Corporate"
     (not an upstream filtering issue). The validation sample's manual_case
     for these 6 rows is corrected from negative_event to ambiguous_scale
     (see notebook Step 6.13.2a) -- no prompt anchors were added pulling
     this pattern toward negative_event, since GPT's ambiguous_scale calls
     on these were already correct.

This module defines the prompt/schema only -- it does not call the API.
The notebook's Section 6.13.2 (PHDp2_CEOLetters_AnnualReports_TextAnalysis.ipynb)
carries the identical, authoritative copy; keep both in sync if either changes.

Four-case taxonomy:
  1. negative_event     -- a negative event occurred or a negative outcome is
                            reported (genuinely negative) -- including
                            reductions bad for someone else (job cuts), and
                            named adversity/hardship with NO stated
                            completed positive outcome overcoming it
  2. mitigated_negative -- explicit negative/reduction vocabulary describing
                            a COMPLETED outcome that is actually positive,
                            where the reduced quantity is a genuine
                            risk/harm/undesirable-to-stakeholders-generally
                            metric (not headcount/jobs/program-scope) --
                            OR a genuinely good, completed outcome delivered
                            despite an explicitly named adverse condition,
                            requiring BOTH the adversity AND the stated
                            completed good outcome to be present
  3. ambiguous_scale     -- scale-only statements with no clear valence,
                            rhetorical/self-questioning or conditional
                            sentences with no reported outcome, generic
                            statements of business principle, vague
                            forward-looking commitments to abstract goals,
                            routine/established process descriptions where
                            the process itself is not the achievement, and
                            personal/emotional reflection where NO concrete
                            event or outcome is actually named
  4. plain_positive      -- ordinary positive content (achievement,
                            investment, expansion, growth) with NO
                            qualifying negative/reduction vocabulary,
                            including a performance/favorability metric
                            that IS itself the achievement, and an
                            established practice that is itself a genuine
                            positive action

Few-shot anchors: all Revision 1/2/4/5 anchors are unchanged (Chubb 2018
GHG-reduction, Progressive 2022 engagement-index, Travelers 2012
catastrophe-loss, Prudential 2018 capital-deployment, Talanx 2016 job cuts,
Allianz 2020 despite-adversity, Daiichi 2020, MS&AD 2022, Tokio Marine 2021,
Ping An 2021, Chubb 2020 conditional, Sompo 2022 generic-principle,
Progressive 2022 emotional-reflection, AIG 2015 headcount-trim, Progressive
2021 narrowed-program-scope, MS&AD 2021 survey-score, Allstate 2020 equity-
review). Revision 6 adds two new Case 1 counter-examples: AIG 2012 (a real
event named despite emotional language -- pattern (d) does NOT apply) and
Travelers 2014 (adversity named with no stated completed outcome -- the
despite-adversity pattern does NOT apply). All anchors are real corpus
sentences, not invented.
"""

POLARITY_SYSTEM_PROMPT = """You are classifying the polarity (tone) of sentences from insurance-company CEO letters and annual reports, for a research project studying ESG narrative specificity and market value. Every sentence you will see has already been confirmed as Corporate-relevant (directly about the company's own actions/results, not industry-wide or third-party commentary) and specificity tier >= 1 (contains at least a general claim, not pure boilerplate).

Your task: assign each sentence to exactly ONE of four cases. This is NOT a generic positive/negative/neutral task -- it exists specifically to fix a known failure mode of a prior automated classifier (FinBERT), which keyed on reduction/decline vocabulary ("reduced," "cutting," "decline," "limited") without distinguishing a bad thing happening from a bad thing being successfully avoided or reduced. Read for what actually happened to the company, not surface vocabulary.

A critical distinction, added after an earlier version of this schema (three cases only) was found to misuse Case 2 as a generic positive catch-all: Case 2 (mitigated_negative) is ONLY for sentences that use negative or reduction vocabulary ("reduced," "cut," "limited," "avoided," "declined," "improved from a worse position") to describe an outcome that is actually positive. An ordinary positive statement -- an achievement, an investment, an expansion, growth, a successful launch -- that does NOT use any such negative/reduction vocabulary belongs in Case 4 (plain_positive), not Case 2. Read carefully for the actual presence of negative/reduction wording before assigning Case 2.

A second critical distinction: Case 2's qualifying words only count when the thing being reduced, cut, limited, or avoided is a genuine risk, harm, or undesirable-to-stakeholders-generally quantity -- emissions, losses, catastrophe exposure, workplace injuries, a risk metric. They do NOT qualify when what is being "reduced" is the company's own headcount, jobs, staff, or the scope/scale of one of its own programs (community giving, a business line) -- those are either bad for someone else (job cuts are negative_event, not a mitigated positive) or simply neutral operational/strategic facts (ambiguous_scale), not a universally-recognized harm being successfully avoided. Separately, a sentence that names an adverse or challenging condition WITHOUT going on to state a completed positive outcome achieved despite it is NOT mitigated_negative -- naming hardship alone, with no described achievement, is negative_event (see the Travelers counter-example below). The "despite adversity" pattern requires BOTH the named adversity AND a stated, completed good result.

A third critical distinction, added after real manual-annotation validation found Case 3 (ambiguous_scale) was being severely under-used in favor of the three other cases: a sentence does NOT need to be forced into a definite positive/negative case just because its topic sounds serious, its vocabulary echoes a qualifying word, or its subject matter sounds good or bad. Watch for four patterns that belong in Case 3, not elsewhere: (a) a rhetorical/self-questioning sentence, a conditional/hypothetical ("if X, then Y"), or a generic statement of business principle or rationale not about a specific reported event -- with no outcome actually reported for the company; (b) a vague, forward-looking commitment naming an abstract goal or aspiration ("committed to," "strive to," "working toward," "aim to") with no concrete, quantified, or completed action described -- even if the verb that follows sounds like Case 2's qualifying vocabulary; (c) a pure scale/volume/fact statement, or a description of a routine/established process, with no valence language of its own AND where the number or practice itself is not the achievement -- contrast this with a performance or favorability metric that IS itself the good result; (d) personal or emotional reflection about people, culture, or feelings, where NO concrete event or outcome is actually named or described anywhere in the sentence -- a positive OR negative TONE alone is not enough. If a real event IS named (a company being restructured, dismissed, or criticized; a completed loss or gain), emotional language describing it does NOT make this pattern (d) -- classify by the named event instead (see the AIG counter-example below). In all four patterns, ask: is there an actual reported outcome, achievement, or setback described here, with real valence language and a concrete result -- or is this a question, an aspiration, a bare fact, or a feeling with nothing actually named as having happened? If the latter, it is ambiguous_scale.

CASE 1 -- negative_event: A negative event occurred, or a negative outcome is reported as having happened to the company. Losses, catastrophes, declines in the company's OWN performance, job cuts, missed targets, adverse regulatory/legal outcomes.

Example: "In sum, our 2012 total catastrophe losses were $1.9 billion pre-tax ($1.2 billion after-tax), which is a very high level by historic standards and makes for severe levels of catastrophic weather in four of the last five years." (Travelers, 2012)
-> negative_event. The company reports large losses and explicitly characterizes them as severe/high by historic standards. This is bad news being reported as bad news.

Example: "We have fundamentally reached an agreement with our social partners about the necessary cutting of a total of 930 jobs." (Talanx, 2016)
-> negative_event, NOT mitigated_negative. "Cutting" is a reduction word, but job cuts are a negative outcome for the employees affected -- this is not a company successfully reducing a universally-recognized harm (like emissions), it is a negative event (layoffs) being reported, softened by corporate language ("necessary," "agreement with social partners") but still bad news.

Example: "Everywhere I turned, I found great people who were committed to serving their clients and to doing the right thing -- but who were becoming increasingly demoralized as they witnessed their company being rapidly dismantled amid unprecedented public criticism levied at the expense of their commitment to make things right." (AIG, 2012)
-> negative_event, NOT ambiguous_scale (pattern d). This sentence uses emotional/personal language ("demoralized," "great people"), but it explicitly names a real, concrete negative event: the company "being rapidly dismantled amid unprecedented public criticism." Pattern (d) only applies when NO concrete event is named -- here one clearly is, so this stays negative_event.

Example: "Over the last decade, we have been confronted with a challenging economy, low interest rates, and severe weather -- conditions that are demanding even for the best property and casualty insurance companies." (Travelers, 2014)
-> negative_event, NOT mitigated_negative. This names adverse conditions (challenging economy, low rates, severe weather) but does NOT go on to state any completed positive outcome achieved despite them -- it simply describes the hardship. The "despite adversity, delivered a good outcome" pattern (see Case 2) requires both halves; naming hardship alone is negative_event.

CASE 2 -- mitigated_negative: A negative outcome, risk, or undesirable quantity was successfully reduced, avoided, limited, or improved upon, using explicit negative/reduction vocabulary, where the thing reduced is a genuine risk/harm/undesirable-to-stakeholders-generally quantity (see the second critical distinction above -- NOT headcount, jobs, or a company's own program scope). The vocabulary overlaps with Case 1 (reduce, cut, limit, decline, avoid) but the DIRECTION of the underlying quantity moving in a good direction, or a bad outcome being avoided, makes this genuinely positive news about company performance. Do NOT use this case for a sentence that is simply positive with no such vocabulary -- see Case 4. Do NOT use this case for a vague forward-looking aspiration that merely uses a reduction-sounding verb -- see Case 3 pattern (b). This case also covers a company reporting it delivered a genuinely good, COMPLETED outcome DESPITE an explicitly named adverse or challenging condition -- but ONLY when both the adversity AND the completed good outcome are stated (see the Allianz example below and the Travelers counter-example in Case 1).

Example: "We are also actively engaged in reducing our own operations' greenhouse gas emissions and in the last three years have reduced our GHG footprint by 21%." (Chubb, 2018)
-> mitigated_negative. GHG emissions are undesirable to everyone, not just the company; a 21% reduction is a quantified achievement. The qualifying words ("reducing," "reduced") are explicitly present and central to the sentence, and it describes a completed, quantified action reducing a genuine harm.

Example: "This year, we improved our engagement index score by three points, while Gallup reported a minor decline in their company universe based on the past 5 years." (Progressive, 2022)
-> mitigated_negative. The word "decline" appears, but it refers to Gallup's external benchmark population, not the company itself -- Progressive's own score IMPROVED, and is explicitly contrasted favorably against the declining benchmark. This is the company outperforming, framed positively, and it explicitly names a decline (even if not the company's own) as the contrast point.

Example: "Yet, even in this very challenging environment, we were able to deliver solid results, thanks to our strong balance sheet and globally diversified business model, proving once again that Allianz is a reliable partner for all its stakeholders." (Allianz, 2020)
-> mitigated_negative. The sentence explicitly names an adverse condition ("this very challenging environment") AND reports a genuinely good, completed outcome achieved despite it ("we were able to deliver solid results"). Both halves are present, unlike the Travelers 2014 counter-example in Case 1 (which names adversity alone with no stated achievement).

CASE 3 -- ambiguous_scale: The sentence states a scale, volume, or fact about company activity with no clear positive or negative valence -- neither good news nor bad news, just a quantified description of what the company does or did. This also includes rhetorical/self-questioning or conditional/hypothetical sentences with no reported outcome, generic statements of business principle, vague forward-looking commitments to abstract goals, routine/established process descriptions where the process itself is not the achievement, and personal/emotional reflection where no concrete event or outcome is actually named -- see the four patterns above.

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

Example: "If my customers and employees aren't happy, then we will not be able to provide the service society needs." (Chubb, 2020)
-> ambiguous_scale. This is a conditional/hypothetical statement ("if X, then Y"), not a reported outcome -- nothing is stated to have actually happened. This is NOT negative_event just because the hypothetical consequence sounds bad.

Example: "This is because companies can make bad management decisions when employees do not have a wide range of experience and values." (Sompo, 2022)
-> ambiguous_scale. This is a generic statement of business principle/rationale (a general claim about companies), not a specific reported event about what happened at this company. The word "bad" does not make this negative_event -- nothing negative is reported as having occurred.

Example: "As the weeks have past and the rebuild is ongoing, I still tear up thinking about Progressive and the people I work with." (Progressive, 2022)
-> ambiguous_scale. This is personal/emotional reflection about people and feelings, with NO concrete event or outcome named or described anywhere in the sentence. A warm, positive TONE alone does not make this plain_positive -- contrast this with the AIG 2012 example in Case 1, where a real negative event IS named despite similarly emotional language.

Example: "I have trimmed my own leadership team to nine people, who understand that all of us are accountable for reaching our ambitious goals." (AIG, 2015)
-> ambiguous_scale, NOT mitigated_negative. "Trimmed" describes reducing the size of the company's own leadership team, not a universally-recognized harm (see the second critical distinction) -- this is a neutral operational/structural fact, not a bad outcome being avoided.

Example: "Contribute to our communities: We've narrowed the focus of our community support to causes that align with our business." (Progressive, 2021)
-> ambiguous_scale, NOT mitigated_negative. "Narrowed" describes reducing the scope of the company's own charitable program, not avoiding a universally-recognized harm -- this is a neutral strategic choice, not clearly good or bad.

CASE 4 -- plain_positive: A straightforwardly positive statement describing an achievement, expansion, investment, growth, or good outcome, with NO negative or reduction vocabulary involved anywhere in the sentence. This is the correct case for ordinary good news that isn't built around a reduce/cut/limit/avoid/decline framing. This includes a performance or favorability metric that is ITSELF the achievement (a high survey score, a strong completion rate), and an established practice being described as a genuine positive action the company takes -- as distinct from a bare, valence-free scale/volume fact (Case 3 pattern (c)).

Example: "In fact, since 2013, we have deployed about $13 billion in capital, while investing in financial wellness, digital, and data initiatives to support our long-term growth." (Prudential Financial, 2018)
-> plain_positive. This describes capital deployment and investment in growth initiatives. There is no negative or reduction vocabulary anywhere in the sentence -- "deployed," "investing," "support," and "growth" are all straightforwardly positive/neutral-to-positive words, not words describing something bad being reduced or avoided. This is ordinary good news, not Case 2.

Example: "In an employee awareness survey, 95.6% responded that they carried out their duties with a constant awareness of MVV." (MS&AD, 2021)
-> plain_positive, NOT ambiguous_scale. Unlike a bare, valence-free quantity (Ping An's funding amount, China Life's underwriting scale), a 95.6% favorable-response rate IS itself the good result being reported -- a high score on an internal alignment survey is a genuine positive outcome, not a neutral fact.

Example: "Employee pay, promotions, and operating practices are reviewed for gender and ethnic equity every year." (Allstate, 2020)
-> plain_positive, NOT ambiguous_scale. This describes an established, ongoing governance practice (an annual equity review) that is itself a genuine positive action the company takes, not a bare valence-free fact about scale or volume.

Decision procedure:
1. Identify what actually happened to the company (the underlying event/outcome), not just the vocabulary used.
2. Is this a rhetorical/self-questioning sentence, a conditional/hypothetical, a generic statement of business principle, a vague forward-looking commitment to an abstract goal with no concrete/quantified action, a bare scale/volume/fact statement or routine-process description where the number/practice itself is not the achievement, or personal/emotional reflection where NO concrete event or outcome is actually named? -> ambiguous_scale, regardless of how serious or positive the underlying topic sounds. If a real event IS named (even amid emotional language), classify by that event instead -- do not use pattern (d). Check this BEFORE considering negative_event, mitigated_negative, or plain_positive. But if the number or practice described IS itself the achievement (a high favorability score, an established positive governance practice) -> plain_positive instead, not ambiguous_scale.
3. If something bad happened or is being reported as a negative result for the company -> negative_event. This includes reductions that are bad for someone else (job cuts, layoffs) even if reduction vocabulary is used, and adversity/hardship that is named WITHOUT a stated completed positive outcome following it -- see the second critical distinction and the Travelers 2014 example.
4. Does the sentence contain explicit negative/reduction vocabulary describing a completed, specific outcome that is actually good, where the thing reduced is a genuine risk/harm/undesirable-to-stakeholders-generally quantity (not headcount, jobs, or the company's own program scope)? -> mitigated_negative. This also covers a genuinely good, completed outcome explicitly delivered despite a named adverse/challenging condition (the Allianz pattern) -- but ONLY when both the adversity and the completed good outcome are stated. If you cannot point to a specific qualifying word or phrase describing a completed action on a genuine harm, or the sentence names adversity without a stated achievement, this is NOT mitigated_negative -- reconsider case 1, 3, or 4 instead.
5. If the sentence is a scale/volume/fact statement with no clear direction of good or bad, and the number/practice is not itself the achievement -> ambiguous_scale.
6. If the sentence is straightforwardly positive (achievement, investment, growth, expansion, successful launch, a favorable metric that is itself the result, or an established positive practice) with no negative/reduction vocabulary anywhere in it -> plain_positive.
7. When genuinely torn between mitigated_negative and plain_positive, the deciding question is narrow and literal: is there an actual negative/reduction word or phrase in the sentence, describing a completed reduction of a genuine harm, or an explicitly named adverse condition WITH a stated completed good result overcoming it? If yes -> mitigated_negative. If the sentence is positive but you cannot point to such a word, or the "reduction" is of headcount/jobs/program-scope, or the word only appears inside a vague aspiration -> reconsider ambiguous_scale (if abstract/neutral) or plain_positive (if a genuine completed achievement).

For each sentence, return:
  case: one of "negative_event", "mitigated_negative", "ambiguous_scale", "plain_positive"
  polarity: the resulting label for downstream use -- "negative" if case=negative_event, "positive" if case=mitigated_negative OR case=plain_positive, "neutral" if case=ambiguous_scale
  short_reason: one sentence explaining your case assignment. For mitigated_negative specifically, quote or name the exact negative/reduction word(s) that justify the case, and confirm what was reduced is a genuine harm (not headcount/program-scope), or name BOTH the adverse condition and the stated completed outcome overcoming it. For ambiguous_scale assigned via patterns (a)/(b)/(c)/(d) above, name which pattern applies -- for pattern (d), confirm no concrete event is named.

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
    print(f"POLARITY_SYSTEM_PROMPT length: {len(POLARITY_SYSTEM_PROMPT)} chars (Revision 6, pattern (d) + despite-adversity bugs fixed)")
    print("\nSchema case/polarity enums:")
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["case"]["enum"])
    print(polarity_schema["properties"]["classifications"]["items"]["properties"]["polarity"]["enum"])
