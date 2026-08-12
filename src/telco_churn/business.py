"""Business cost model for retention-offer targeting.

Single source of truth: notebooks and the Streamlit app must both import from
here. Two implementations of this logic is how an app ends up recommending
contacts that the evaluation notebook says are unprofitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- assumptions
# Illustrative figures based on common telecom industry ranges, NOT internal
# company data. The IBM dataset contains no costs, margins, or campaign outcomes.
# Every financial conclusion is conditional on these numbers, so the evaluation
# notebook must run a sensitivity analysis over offer_effectiveness and gross_margin.

COST_ASSUMPTIONS: dict = {
    "gross_margin": 0.40,  # g — gross margin on service revenue
    "offer_discount_rate": 0.20,  # r — 20% off the monthly bill
    "offer_duration_months": 3,  # d — discount runs for 3 months
    "fixed_contact_cost": 2.00,  # k — USD, agent time plus channel per contact
    "offer_effectiveness": 0.30,  # epsilon — share of would-be churners retained
    "horizon_by_contract": {  # H — months remaining if the customer stays
        "Month-to-month": 12,
        "One year": 18,
        "Two year": 24,
    },
    "currency": "USD",
}


def value_at_risk(monthly_charges, contract, a: dict = COST_ASSUMPTIONS):
    """V_i = M_i * H_i * g — margin lost if this customer leaves.

    Raises on unknown contract types rather than silently producing NaN, which
    would propagate into cost figures and look like a modelling result.
    """
    horizon = pd.Series(contract).map(a["horizon_by_contract"])
    if horizon.isna().any():
        unknown = sorted(set(pd.Series(contract)[horizon.isna()]))
        raise ValueError(f"No horizon defined for contract type(s): {unknown}")
    return (
        np.asarray(monthly_charges, dtype=float)
        * horizon.to_numpy(float)
        * a["gross_margin"]
    )


def offer_cost(monthly_charges, a: dict = COST_ASSUMPTIONS):
    """C_i = r * M_i * d + k — cost of extending a retention offer."""
    mc = np.asarray(monthly_charges, dtype=float)
    return (
        a["offer_discount_rate"] * mc * a["offer_duration_months"]
        + a["fixed_contact_cost"]
    )


def optimal_threshold(monthly_charges, contract, a: dict = COST_ASSUMPTIONS):
    """p*_i = C_i / (epsilon * V_i) — probability at which contacting is cheaper.

    Derivation: expected cost of contacting is p(1-eps)V + C; of not contacting,
    pV. Setting the first below the second gives p > C / (eps * V).
    See Elkan (2001), "The Foundations of Cost-Sensitive Learning", IJCAI.
    """
    v = value_at_risk(monthly_charges, contract, a)
    c = offer_cost(monthly_charges, a)
    return np.clip(c / (a["offer_effectiveness"] * v), 0.0, 1.0)


def breakeven_effectiveness(monthly_charges, contract, a: dict = COST_ASSUMPTIONS):
    """C_i / V_i — the epsilon below which contacting a known churner loses money."""
    return offer_cost(monthly_charges, a) / value_at_risk(monthly_charges, contract, a)


def policy_cost(y_true, action, monthly_charges, contract, a: dict = COST_ASSUMPTIONS):
    """Total cost of a contact policy. Lower is better.

    TN 0 | FP C_i | FN V_i | TP (1-eps)V_i + C_i
    """
    y = np.asarray(y_true, dtype=float)
    act = np.asarray(action, dtype=float)
    if y.shape != act.shape:
        raise ValueError(f"y_true {y.shape} and action {act.shape} must align")
    v = value_at_risk(monthly_charges, contract, a)
    c = offer_cost(monthly_charges, a)
    return float(
        np.sum(
            act * (c + (1.0 - a["offer_effectiveness"]) * v * y) + (1.0 - act) * v * y
        )
    )


def cost_per_customer(
    y_true, action, monthly_charges, contract, a: dict = COST_ASSUMPTIONS
):
    """Primary project metric: expected cost per customer, in currency."""
    return policy_cost(y_true, action, monthly_charges, contract, a) / len(
        np.asarray(y_true)
    )


def policy_from_proba(y_proba, monthly_charges, contract, a: dict = COST_ASSUMPTIONS):
    """Cost-optimal contact decisions from calibrated probabilities.

    Uses the per-customer threshold, so a low-bill short-contract customer must
    look likelier to churn before contacting them pays off.
    """
    return (
        np.asarray(y_proba, dtype=float)
        > optimal_threshold(monthly_charges, contract, a)
    ).astype(int)
