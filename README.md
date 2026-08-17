# Telco Customer Churn

A churn model for the IBM Telco dataset. The framing question isn't "who will churn" but
"who should get a retention offer," which turns out to change most of the decisions
downstream.

The short version of the result: the model works fine as a model (ROC-AUC around 0.85) and
fails the business bar I set for it before starting. It captures roughly 24% of the value
available between doing nothing and perfect foresight, against a 40% threshold I committed to
in advance. I've left that verdict in rather than moving the goalpost, because the reason it
fails is the interesting part.

## Why it fails, briefly

Every retention decision here comes from one inequality. Contact a customer when

```
p × ε × V  >  C        ⟺        p > C / (ε × V)
```

where `p` is predicted churn probability, `V` is the margin you lose if they leave, `C` is
what the offer costs, and `ε` is the probability the offer actually works.

`ε` isn't in the dataset. It can't be estimated from it either — you'd need a randomised
experiment where some at-risk customers got offers and some didn't. I assumed 0.30.

That assumption caps everything. Even a model with perfect foresight only recovers about
16 USD per customer, because 70% of the people you correctly identify leave anyway. So the
entire prize is small, and the measurement noise around it is large relative to the prize.

Worse, when I swept `ε` across a plausible range, the share of value captured *peaks near the
assumed value and falls off on both sides*. At low `ε` there's nothing to win. At high `ε`,
contacting gets cheap enough that "just offer to everyone on a month-to-month contract" —
a one-line SQL query — captures most of what's there, and the model's edge shrinks.

That's the finding I'd defend in an interview: the value of prediction here is bounded and
non-monotonic in a parameter nobody has measured, so the next thing to spend money on is an
experiment, not a better model.

## Running it

```bash
git clone https://github.com/rbennum/telco-churn
cd telco-churn
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"
streamlit run app.py
```

The app loads `models/final_pipeline.joblib` and lets you score a single customer, upload a
CSV, or drag the `ε` slider and watch the conclusion move.

## Layout

```
notebooks/          01 ingest → 02 eda → 03 baseline → 04 model → 05 evaluation
src/telco_churn/
  config.py         seeds, paths, artifact resolution
  business.py       the cost model — V, C, thresholds, oracle, policy costs
  cleaning.py       raw → model-ready, shared by notebooks and app
app.py              streamlit
models/             fitted pipeline + final metrics + model card
```

The cost model lives in the package rather than in a notebook because the app and the
evaluation have to agree on it. Two copies of that arithmetic is how an app ends up
recommending contacts the notebook says are unprofitable.

Notebooks run in order and hand off through JSON — each writes what the next one reads, and
each asserts the shape of what it loaded. Run them out of order and they'll tell you.

## What's in each notebook

**01 — ingest.** Framing, the cost model, leakage screening, pandera schema, train/test split.
The split happens before anything statistical, and `test.parquet` isn't attached to notebooks
02–04 at all, so the "don't touch the test set" rule is a missing file rather than good
intentions.

**02 — EDA.** Training data only. The section I'd point at is the segment analysis that ranks
by recoverable margin instead of churn rate — month-to-month fibre is 30% of customers and 63%
of the money. Also where I found that tenure protects against churn on month-to-month contracts
and mildly *increases* it on two-year ones, which an additive model can't represent.

**03 — baseline.** Logistic regression with the minimum preprocessing that makes it run, then
five feature-engineering hypotheses tested against it. Four of five did nothing. The fifth was
cheaper by 0.16 USD per customer with folds disagreeing on sign, so I kept the simpler option.
Also settles the label permutation test: shuffled labels score 0.4965 against a true 0.8455,
so no leakage.

**04 — model selection.** Four model families, nested cross-validation, calibration, and the
threshold treated as a hyperparameter — which it is, even though it doesn't feel like one.
Selecting the threshold on the validation fold instead of inside it would have looked 0.31 USD
per customer better than reality. On a 100k customer base that's about 31,000 USD of benefit
that doesn't exist.

Nothing beat logistic regression. Gradient boosting was slower and no better, despite a
Box-Tidwell test saying two features violate linearity of the logit. 213 validation queries
bought 0.013 USD per customer over the notebook-03 baseline.

**05 — evaluation.** The test set, opened once, after writing down what each possible outcome
would mean. Then error analysis, slice performance including protected attributes, permutation
importance, conformal prediction, and the `ε` sweep.

## Numbers

Pull the exact figures from `models/final_metrics.json`; these are the ones worth knowing.

| | |
|---|---|
| Rows / features | 7,043 / 19 |
| Split | 80/20 stratified, seed 29 |
| Model | Logistic regression, uncalibrated, cost-derived threshold |
| Test ROC-AUC | ~0.85 |
| Headroom captured (test) | 23.6%, 95% CI [16.3%, 30.3%] |
| Same, cross-validated | 23.5% |
| Verdict | DOES NOT CLEAR (bar was 40%) |
| Best naive policy | contact everyone on month-to-month |

The thing I'm most pleased about is the third and fourth rows agreeing. After 213 looks at the
validation folds you'd normally expect the test number to come in worse. It didn't, which is
what nested CV and pre-registration are for.

## Limitations

The dataset is a single snapshot with no dates, so out-of-time validation is impossible and
I can't simulate drift. Anyone deploying this needs monitoring I couldn't pre-validate.

The model is predictive, not causal. It ranks who's likely to leave. It doesn't show that
contacting them helps. Ascarza's work on retention futility (JMR 2018) found the highest-risk
customers often aren't the best targets, and some people churn *because* an offer reminded them
to reconsider a subscription they'd stopped thinking about.

It's noticeably worse on two-year contracts — ROC-AUC 0.64 against 0.85 overall, because there
are barely any churn events to learn from. Those are also the customers with the most value at
risk, so the model is weakest exactly where mistakes are most expensive. Every one of the ten
costliest errors was a two-year fibre customer the model was confident about.

Conformal prediction gives 90.1% coverage overall, which is the target, and 86.9% on
month-to-month. The guarantee is marginal, not conditional — it was never a promise about
subgroups, and the shortfall lands on the segment that matters most.

All the cost parameters are made up. They're plausible industry figures, not anyone's actual
margins, and every dollar figure in the repo is conditional on them.

## What I'd do next

Run the offers as a randomised experiment on a high-risk slice. That measures `ε` directly,
and it produces the outcome labels you need to train an uplift model, which is the right tool
for this decision. Ranking risk was never quite the question.

## Notes

Dataset is the IBM Telco sample, via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
It isn't committed here, so download it yourself if you want to re-run the notebooks.