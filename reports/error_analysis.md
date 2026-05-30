# Error Analysis

## 1. Most common false positive patterns

Manual review of 10 TF-IDF false positives suggests that not-credible statements are often predicted as credible when they contain concrete numeric claims, institutional language, or matter-of-fact policy wording. Several examples look plausible without the missing PolitiFact context, which is a hard setting for statement-only modeling.

- false | katrina-shankland | jobs: wisconsin pace double number layoffs year
- false | facebook-posts | federal-budget,military,poverty: says percent federal spending goes military percent goes food agriculture including food stamps
- pants-fire | donald-trump | immigration: number illegal immigrants could million could million
- false | sarah-palin | crime,criminal-justice: rosemary lehmberg travis county das office convened grand jury indicted rick perry
- barely-true | drew-springer | environment: says bag litter increased san francisco banned singleuse shopping bags

## 2. Most common false negative patterns

Manual review of 10 TF-IDF false negatives suggests that credible statements are often predicted as not credible when they use adversarial political framing, mention polarizing subjects, or resemble common campaign-attack phrasing. This is consistent with a lexical model picking up topic and rhetoric shortcuts.

- true | rick-perry | immigration: building wall usmexico border take literally years
- true | hillary-clinton | health-care: year people die america dont health care
- mostly-true | chris-sgro | crime,gays-and-lesbians,sexuality: public safety issues cities allow transgender people use bathroom gender identify
- true | lindsey-graham | sotomayor-nomination,supreme-court: time someone like scalia ginsburg got plus votes
- true | mike-pence | climate-change,energy: contends president obama literally said capandtrade proposals pass utility rates words would necessarily skyrocket

## 3. Examples where the original label is debatable

Some errors are short, context-dependent claims where the truth label depends on date, jurisdiction, or a narrow definition. Those examples are difficult for statement-only models because the model cannot verify the external evidence behind the claim.

## 4. What TF-IDF learned well

TF-IDF improved macro F1 from 0.355 for the dummy baseline to 0.610. It learned repeatable lexical cues and remains the easiest model to inspect through top n-grams and per-statement feature contributions.

## 5. What the embedding model added

Sentence embeddings with Logistic Regression reached macro F1 0.616. Compared with TF-IDF, this tests whether pretrained semantic similarity helps on short claims without fine-tuning. In this run it is best treated as a semantic baseline rather than a clear replacement for TF-IDF.

## 6. What Sentence embeddings + GBM helped explain

Sentence embeddings + GBM reached macro F1 0.624. The TF-IDF-vs-GBM comparison was {'both_correct': 425, 'tfidf_correct_embeddings_gbm_wrong': 133, 'tfidf_wrong_embeddings_gbm_correct': 148, 'both_wrong': 208}. This shows whether a nonlinear model on semantic embeddings corrects a meaningful number of TF-IDF mistakes or mostly shifts the error profile.

## 7. Main limitations

1. LIAR contains short statements, not full articles.
2. Binary mapping loses nuance.
3. Dropping `half-true` makes the task cleaner but less complete.
4. Political-domain bias is likely.
5. No real-time fact checking is performed.
