"""Telco churn retention targeting — decision support app.

Loads the pipeline produced by notebook 04 and imports every cost calculation from
telco_churn.business, so the app and the evaluation notebook cannot disagree.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from telco_churn.business import (
    COST_ASSUMPTIONS,
    breakeven_effectiveness,
    cost_per_customer,
    offer_cost,
    optimal_threshold,
    oracle_cost,
    policy_from_proba,
    value_at_risk,
)
from telco_churn.cleaning import (
    CATEGORY_LEVELS,
    FEATURE_COLS,
    TARGET,
    prepare_raw,
    validate_for_scoring,
)

st.set_page_config(page_title="Churn Retention Targeting", page_icon="◫", layout="wide")

# The pipeline was FITTED under this setting, so every step recorded pandas feature
# names. Without it here, ColumnTransformer hands bare arrays down the chain and
# sklearn warns that the names are missing. This is part of the artifact's contract,
# not a display preference.
sklearn.set_config(transform_output="pandas")

MODEL_PATHS = [
    Path("models/final_pipeline.joblib"),
    Path("artifacts/final_pipeline.joblib"),
    Path("final_pipeline.joblib"),
    Path("outputs/final_pipeline.joblib"),
]
METRIC_PATHS = [
    Path("models/final_metrics.json"),
    Path("artifacts/final_metrics.json"),
    Path("final_metrics.json"),
    Path("outputs/final_metrics.json"),
]


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            return joblib.load(p), p
    return None, None


@st.cache_data
def load_metrics():
    for p in METRIC_PATHS:
        if p.exists():
            return json.loads(p.read_text())
    return None


model, model_path = load_model()
metrics = load_metrics()

if model is None:
    st.error(
        "Could not find `final_pipeline.joblib`.\n\n"
        "Searched: " + ", ".join(f"`{p}`" for p in MODEL_PATHS) + "\n\n"
        "Run notebook 04 and commit the artifact, or place it at one of those paths."
    )
    st.stop()


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Cost assumptions")
st.sidebar.caption(
    "None of these come from the data. They are illustrative industry figures, and "
    "every monetary result in this app is conditional on them."
)

eps = st.sidebar.slider(
    "Offer effectiveness (ε)", 0.05, 0.65, float(COST_ASSUMPTIONS["offer_effectiveness"]),
    0.01,
    help="Probability that an offer actually retains a customer who would otherwise "
         "leave. This parameter is absent from the dataset and can only be measured "
         "by a randomised experiment.",
)
margin = st.sidebar.slider(
    "Gross margin (g)", 0.15, 0.65, float(COST_ASSUMPTIONS["gross_margin"]), 0.01)
discount = st.sidebar.slider(
    "Offer discount rate (r)", 0.05, 0.50, float(COST_ASSUMPTIONS["offer_discount_rate"]), 0.01)
duration = st.sidebar.slider(
    "Offer duration, months (d)", 1, 12, int(COST_ASSUMPTIONS["offer_duration_months"]))
contact_cost = st.sidebar.number_input(
    "Fixed contact cost (k), USD", 0.0, 50.0,
    float(COST_ASSUMPTIONS["fixed_contact_cost"]), 0.5)

A = {
    **COST_ASSUMPTIONS,
    "offer_effectiveness": eps,
    "gross_margin": margin,
    "offer_discount_rate": discount,
    "offer_duration_months": duration,
    "fixed_contact_cost": contact_cost,
}

st.sidebar.divider()
st.sidebar.caption(f"Model: `{model_path}`")
if metrics:
    st.sidebar.caption(
        f"Test ROC-AUC {metrics['test_metrics']['roc_auc']:.3f} · "
        f"verdict **{metrics['verdict']}**"
    )


# ----------------------------------------------------------------- header
st.title("Retention Offer Targeting")
st.caption(
    "Decides which customers should receive a retention offer, by comparing the "
    "expected cost of contacting against the expected cost of doing nothing."
)

with st.expander("Read this before acting on any output", expanded=False):
    st.markdown(
        """
**This model ranks risk. It does not prove that contacting high-risk customers is
profitable.** Those are different claims, and only the first is supported by the data.
Customers most likely to leave are often the hardest to persuade, and some churn
*because* an offer prompted them to reconsider a subscription running on autopilot.

**The verdict against its own success criterion was DOES NOT CLEAR.** The model captured
about 24% of the value available between doing nothing and perfect foresight, against a
40% bar set before modelling began. It is deployed here as a decision aid, not as a
validated business case.

**It is materially weaker for long-contract customers.** ROC-AUC on two-year contracts is
0.64 against 0.85 overall, because there are too few churn events to learn from — yet
those customers carry the largest value at risk. Long-contract, high-bill accounts should
not be actioned from this score without human review.

**The recommended next step is a randomised experiment**, not a better model. That
measures offer effectiveness directly and produces the labels needed for an uplift model,
which is the correct tool for this decision.
        """
    )

tab_single, tab_batch, tab_portfolio = st.tabs(
    ["Single customer", "Batch scoring", "Portfolio & sensitivity"])


# ----------------------------------------------------------------- helpers
def score_frame(frame: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    """Attach probability, threshold, economics and decision to a clean frame."""
    proba = model.predict_proba(frame[FEATURE_COLS])[:, 1]
    mc, ct = frame["MonthlyCharges"], frame["Contract"]

    out = frame.copy()
    out["churn_probability"] = proba
    out["threshold"] = optimal_threshold(mc, ct, assumptions)
    out["value_at_risk"] = value_at_risk(mc, ct, assumptions)
    out["offer_cost"] = offer_cost(mc, assumptions)
    out["breakeven_eps"] = breakeven_effectiveness(mc, ct, assumptions)
    out["contact"] = policy_from_proba(proba, mc, ct, assumptions)
    # Expected saving from contacting: p * eps * V - C. Positive means worth doing.
    out["expected_gain"] = (
        proba * assumptions["offer_effectiveness"] * out["value_at_risk"] - out["offer_cost"])
    return out


# ----------------------------------------------------------------- single
with tab_single:
    st.subheader("Score one customer")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        contract = st.selectbox("Contract", CATEGORY_LEVELS["Contract"])
        monthly = st.number_input("Monthly charges (USD)", 1.0, 200.0, 70.0, 0.05)
        total_default = round(float(tenure) * float(monthly), 2)
        total = st.number_input("Total charges (USD)", 0.0, 20000.0, total_default, 0.05,
                                help="Defaults to tenure × monthly charges.")
        paperless = st.selectbox("Paperless billing", CATEGORY_LEVELS["PaperlessBilling"])
        payment = st.selectbox("Payment method", CATEGORY_LEVELS["PaymentMethod"])

    with c2:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", CATEGORY_LEVELS["gender"])
        senior = st.selectbox("Senior citizen", CATEGORY_LEVELS["SeniorCitizen"])
        partner = st.selectbox("Partner", CATEGORY_LEVELS["Partner"])
        dependents = st.selectbox("Dependents", CATEGORY_LEVELS["Dependents"])
        phone = st.selectbox("Phone service", CATEGORY_LEVELS["PhoneService"])
        lines = st.selectbox("Multiple lines", CATEGORY_LEVELS["MultipleLines"])

    with c3:
        st.markdown("**Internet services**")
        internet = st.selectbox("Internet service", CATEGORY_LEVELS["InternetService"])
        svc_opts = (["No internet service"] if internet == "No" else ["No", "Yes"])
        security = st.selectbox("Online security", svc_opts)
        backup = st.selectbox("Online backup", svc_opts)
        protection = st.selectbox("Device protection", svc_opts)
        support = st.selectbox("Tech support", svc_opts)
        tv = st.selectbox("Streaming TV", svc_opts)
        movies = st.selectbox("Streaming movies", svc_opts)

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": lines if phone == "Yes" else "No phone service",
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }])

    scored = score_frame(row, A).iloc[0]
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn probability", f"{scored['churn_probability']:.1%}")
    m2.metric("Contact threshold", f"{scored['threshold']:.1%}",
              help="Contacting becomes cheaper than doing nothing above this probability. "
                   "Derived from the cost structure, not tuned.")
    m3.metric("Value at risk", f"${scored['value_at_risk']:,.0f}")
    m4.metric("Offer cost", f"${scored['offer_cost']:,.2f}")

    if scored["contact"]:
        st.success(
            f"**Send the offer.** Expected saving USD {scored['expected_gain']:,.2f} per "
            f"customer. Probability {scored['churn_probability']:.1%} exceeds the "
            f"{scored['threshold']:.1%} threshold where contacting starts paying."
        )
    else:
        st.info(
            f"**Do not send the offer.** Expected loss USD {-scored['expected_gain']:,.2f} "
            f"if contacted. Probability {scored['churn_probability']:.1%} is below the "
            f"{scored['threshold']:.1%} threshold."
        )

    if contract == "Two year":
        st.warning(
            "Two-year contract: the model discriminates poorly here (ROC-AUC 0.64 against "
            "0.85 overall) while these accounts carry the highest value at risk. Treat this "
            "score as weak evidence and review manually."
        )

    with st.expander("How this decision was reached"):
        st.markdown(
            "Contacting is worthwhile when the expected saved margin exceeds the "
            "offer cost, which rearranges into a threshold on the probability:"
        )
        # st.latex rather than $$...$$ inside markdown: Streamlit parses paired $
        # signs as inline maths, so currency values in the table below would pair
        # up with the maths delimiters and destroy both.
        # Note the trailing space in the middle literal: adjacent string literals
        # concatenate with no separator, so \quad would otherwise fuse with the
        # next token into \quadp.
        st.latex(r"p \times \varepsilon \times V > C"
                 r"\quad\Longleftrightarrow\quad "
                 r"p > p^{*} = \frac{C}{\varepsilon V}")

        st.dataframe(
            pd.DataFrame([
                {"Quantity": "p — churn probability",
                 "Value": f"{scored['churn_probability']:.4f}",
                 "Source": "model"},
                {"Quantity": "V — value at risk",
                 "Value": f"USD {scored['value_at_risk']:,.2f}",
                 "Source": "monthly charges x horizon x margin"},
                {"Quantity": "C — offer cost",
                 "Value": f"USD {scored['offer_cost']:,.2f}",
                 "Source": "discount x monthly x months + contact cost"},
                {"Quantity": "epsilon — offer effectiveness",
                 "Value": f"{eps:.2f}",
                 "Source": "ASSUMED — not measurable from this data"},
                {"Quantity": "p* — contact threshold",
                 "Value": f"{scored['threshold']:.4f}",
                 "Source": "C / (epsilon x V)"},
            ]),
            width="stretch", hide_index=True,
        )

        st.caption(
            f"Break-even effectiveness for this customer is "
            f"{scored['breakeven_eps']:.3f}. Below that value, contacting loses money "
            f"even if you knew for certain they would leave. The threshold is derived "
            f"from the cost structure, not tuned on data, so it cannot overfit."
        )


# ----------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Score a customer file")
    st.caption(
        "Upload the raw Kaggle CSV or any file with the same columns. Cleaning matches "
        "notebook 01 exactly, because the app imports it from the package rather than "
        "reimplementing it."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        clean = prepare_raw(raw)
        ok, problems = validate_for_scoring(clean)

        if not ok:
            st.error("File cannot be scored:")
            for p in problems:
                st.write(f"- {p}")
        else:
            scored = score_frame(clean, A)
            st.session_state["scored_batch"] = scored

            n_contact = int(scored["contact"].sum())
            spend = float(scored.loc[scored["contact"] == 1, "offer_cost"].sum())
            gain = float(scored.loc[scored["contact"] == 1, "expected_gain"].sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Customers", f"{len(scored):,}")
            k2.metric("To contact", f"{n_contact:,}", f"{n_contact/len(scored):.1%}")
            k3.metric("Campaign spend", f"${spend:,.0f}")
            k4.metric("Expected net gain", f"${gain:,.0f}",
                      help="Sum over contacted customers of p·ε·V − C. Conditional on the "
                           "assumed effectiveness in the sidebar.")

            st.divider()
            st.markdown("**Contact list, ranked by expected gain**")
            cols = ["churn_probability", "threshold", "expected_gain", "value_at_risk",
                    "offer_cost", "tenure", "Contract", "InternetService", "MonthlyCharges"]
            queue = (scored[scored["contact"] == 1]
                     .sort_values("expected_gain", ascending=False)[cols])
            st.dataframe(queue.head(200).style.format({
                "churn_probability": "{:.1%}", "threshold": "{:.1%}",
                "expected_gain": "${:,.2f}", "value_at_risk": "${:,.0f}",
                "offer_cost": "${:,.2f}", "MonthlyCharges": "${:,.2f}",
            }), width="stretch")

            st.download_button(
                "Download full scored file (CSV)",
                scored.to_csv(index=False).encode(),
                "scored_customers.csv", "text/csv",
            )

            if TARGET in clean.columns:
                st.divider()
                st.markdown("**Outcomes are present in this file, so the policy can be priced**")
                y = clean[TARGET]
                mc, ct = clean["MonthlyCharges"], clean["Contract"]
                model_cost = cost_per_customer(y, scored["contact"], mc, ct, A)
                nobody = cost_per_customer(y, np.zeros(len(y)), mc, ct, A)
                m2m = cost_per_customer(y, (ct == "Month-to-month").astype(int), mc, ct, A)
                orc = oracle_cost(y, mc, ct, A)
                naive = min(nobody, m2m)
                hd = naive - orc

                comp = pd.DataFrame([
                    {"policy": "Oracle (perfect foresight)", "cost_per_customer": orc},
                    {"policy": "This model", "cost_per_customer": model_cost},
                    {"policy": "Contact all month-to-month", "cost_per_customer": m2m},
                    {"policy": "Contact nobody", "cost_per_customer": nobody},
                ]).sort_values("cost_per_customer")
                st.dataframe(comp.style.format({"cost_per_customer": "${:,.2f}"}),
                             width="stretch", hide_index=True)

                if hd > 1e-9:
                    st.metric("Headroom captured", f"{(naive - model_cost)/hd:.1%}",
                              help="Share of the gap between the best naive policy and "
                                   "perfect foresight that the model closes.")
                else:
                    st.warning(
                        "Headroom is zero at these assumptions: offers lose money even for "
                        "customers you know will churn, so no model changes the outcome."
                    )
    else:
        st.info("Upload a file to score. The Kaggle `WA_Fn-UseC_-Telco-Customer-Churn.csv` "
                "works as-is.")


# ----------------------------------------------------------------- portfolio
with tab_portfolio:
    st.subheader("What the conclusion depends on")
    st.caption(
        "The cost model contains one parameter that does not appear in the data and cannot "
        "be estimated from it: offer effectiveness. Everything downstream inherits it."
    )

    scored = st.session_state.get("scored_batch")
    if scored is None:
        st.info("Score a file in the Batch tab first — this analysis runs on that population.")
    else:
        has_truth = TARGET in scored.columns
        mc, ct = scored["MonthlyCharges"], scored["Contract"]
        proba = scored["churn_probability"].to_numpy()

        grid = np.round(np.arange(0.05, 0.66, 0.025), 4)
        rows = []
        for e in grid:
            a = {**A, "offer_effectiveness": e}
            act = policy_from_proba(proba, mc, ct, a)
            v, c = value_at_risk(mc, ct, a), offer_cost(mc, a)
            rec = {
                "epsilon": e,
                "contacted_%": 100 * float(np.mean(act)),
                "campaign_spend": float(c[act == 1].sum()),
                "expected_gain": float((proba * e * v - c)[act == 1].sum()),
            }
            if has_truth:
                y = scored[TARGET]
                naive = min(cost_per_customer(y, np.zeros(len(y)), mc, ct, a),
                            cost_per_customer(y, (ct == "Month-to-month").astype(int), mc, ct, a))
                orc = oracle_cost(y, mc, ct, a)
                hd = naive - orc
                mcost = cost_per_customer(y, act, mc, ct, a)
                rec["headroom_captured_%"] = 100 * (naive - mcost) / hd if hd > 1e-9 else 0.0
            rows.append(rec)

        sens = pd.DataFrame(rows)

        if has_truth:
            st.markdown("**Share of achievable value the model captures, across ε**")
            chart = sens.set_index("epsilon")[["headroom_captured_%"]]
            st.line_chart(chart, height=280)

            at_eps = sens.iloc[(sens["epsilon"] - eps).abs().argmin()]
            peak = sens.loc[sens["headroom_captured_%"].idxmax()]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"At ε = {eps:.2f}", f"{at_eps['headroom_captured_%']:.1f}%")
            c2.metric("Peak", f"{peak['headroom_captured_%']:.1f}%",
                      f"at ε = {peak['epsilon']:.3f}")
            c3.metric("Reaches 40% bar?",
                      "Yes" if sens["headroom_captured_%"].max() >= 40 else "No")

            st.markdown(
                "Capture typically **peaks and then falls** as effectiveness rises. As offers "
                "become more effective, contacting gets cheap enough that a crude segment rule "
                "improves faster than the model does — precision matters less when contacting "
                "everyone is nearly free. The model's advantage is squeezed from both ends: at "
                "low ε there is nothing to win, at high ε a trivial rule wins most of it."
            )

        st.divider()
        st.markdown("**Campaign size and expected return, across ε**")
        st.line_chart(sens.set_index("epsilon")[["contacted_%"]], height=220)
        st.line_chart(sens.set_index("epsilon")[["campaign_spend", "expected_gain"]], height=260)

        st.divider()
        st.markdown("**Break-even effectiveness by contract**")
        be = (scored.assign(breakeven=breakeven_effectiveness(mc, ct, A))
              .groupby("Contract", observed=False)["breakeven"]
              .agg(["count", "mean", "max"]).round(4))
        st.dataframe(be, width="stretch")
        st.caption(
            f"Below these values, contacting loses money even for a certain churner. "
            f"Current assumption is ε = {eps:.2f}. If a real-world study puts effectiveness "
            "below the figures above, the problem is the economics of the offer, not the model."
        )