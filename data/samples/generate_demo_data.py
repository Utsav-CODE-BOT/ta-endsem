"""
Generates synthetic multi-domain SaaS / E-commerce customer feedback dataset.
Labeled explicitly as: DEMO / SYNTHETIC DATA per Section 49.
Used to evaluate cross-domain robustness (RQ5) alongside the real US Airline benchmark.
Features:
- Parameterized natural language variations (eliminating verbatim duplicate text)
- Injected incident window (v4.2 payment gateway 502 surge in days 18-27)
- Distractor factors and multiple functional categories
- Controlled missingness and diverse customer identifiers
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

ERROR_CODES = ["502", "504", "ERR_PAY_GATEWAY", "TIMEOUT_104", "AUTH_FAIL_9"]
DEVICES = ["iPhone 15", "Pixel 8", "Galaxy S23", "iPad Pro", "Chrome 128", "Safari 17"]
PAYMENT_METHODS = ["credit card", "debit card", "Apple Pay", "Google Pay", "UPI", "PayPal"]
AMOUNTS = ["$15.00", "$49.99", "$120.00", "$8.50", "$250.00"]

# Template generators with dynamic slots
def generate_payment_negative():
    err = random.choice(ERROR_CODES)
    dev = random.choice(DEVICES)
    amt = random.choice(AMOUNTS)
    method = random.choice(PAYMENT_METHODS)
    templates = [
        f"Payment of {amt} failed repeatedly on checkout with error {err} right after updating on {dev}.",
        f"Cannot complete {method} checkout: screen hung and threw {err} gateway timeout.",
        f"My {method} was charged {amt} but order failed with error {err}. App v4.2 is broken!",
        f"Payment processing timed out ({err}) at step 3 of checkout on {dev}. Money debited from bank.",
        f"Checkout button is completely unresponsive when paying {amt} via {method} after the v4.2 update.",
    ]
    return random.choice(templates), "payment", "negative", "Mobile App v4.2", "Payment Gateway"

def generate_payment_positive():
    amt = random.choice(AMOUNTS)
    method = random.choice(PAYMENT_METHODS)
    templates = [
        f"Payment of {amt} with {method} went through seamlessly in less than 2 seconds.",
        f"Checkout was smooth and instant with {method}. Clean purchase experience.",
        f"Order completed without any friction using {method}, great receipt breakdown.",
    ]
    return random.choice(templates), "payment", "positive", "Mobile App v4.1", "Payment Gateway"

def generate_login_negative():
    dev = random.choice(DEVICES)
    templates = [
        f"App crashes instantly on {dev} every time I attempt Google OAuth authentication.",
        f"Biometric FaceID login fails continuously on {dev} after installing the latest update.",
        f"Password reset link directs to a blank white screen. Completely locked out of account.",
        f"Two-factor authentication code is never received via SMS. Cannot access my dashboard.",
    ]
    return random.choice(templates), "login", "negative", "Mobile App v4.2", "Auth Service"

def generate_login_positive():
    dev = random.choice(DEVICES)
    templates = [
        f"Biometric login on {dev} works flawlessly. Instant access to my profile.",
        f"Smooth sign-in flow and Google single sign-on worked on the first attempt.",
        f"Clean authentication screen, very easy to switch between work profiles.",
    ]
    return random.choice(templates), "login", "positive", "Mobile App v4.1", "Auth Service"

def generate_delivery_negative():
    days = random.randint(3, 9)
    templates = [
        f"Shipment arrived {days} days past the promised delivery window and outer box was badly crushed.",
        f"Tracking status shows 'Delivered to front porch' but package is missing. Courier unresponsive.",
        f"Package misrouted to incorrect distribution hub. Delay has exceeded one full week.",
    ]
    return random.choice(templates), "delivery", "negative", "Logistics Partner B", "Delivery Depot"

def generate_delivery_positive():
    templates = [
        f"Arrived a day earlier than scheduled. Courteous delivery driver and excellent packaging.",
        f"Real-time GPS tracking of courier was super accurate. Package delivered in pristine condition.",
        f"Rapid same-day dispatch and clear status SMS notifications throughout transit.",
    ]
    return random.choice(templates), "delivery", "positive", "Logistics Partner A", "Delivery Fleet"

def generate_support_negative():
    mins = random.randint(30, 75)
    templates = [
        f"Support phone agent placed me on hold for {mins} minutes then disconnected the call abruptly.",
        f"Submitted billing refund ticket 12 days ago with zero response. Unacceptable customer care.",
        f"Automated support chatbot is stuck in an unhelpful loop and refuses to connect a human agent.",
    ]
    return random.choice(templates), "customer_service", "negative", "Support Desk", "Agent Team"

def generate_support_positive():
    templates = [
        f"Support representative resolved my invoice discrepancy within 4 minutes. Outstanding care!",
        f"Friendly live chat specialist helped walk me through subscription renewal immediately.",
        f"Help desk was very responsive and credited my account promptly without hassle.",
    ]
    return random.choice(templates), "customer_service", "positive", "Support Desk", "Live Chat"

def generate_performance():
    templates = [
        ("Dashboard analytics take over 25 seconds to render data tables. Extreme interface lag.", "performance", "negative", "Web Portal v3.1", "Database Cluster"),
        ("Severe background battery drain and device overheating whenever app v4.2 is open.", "performance", "negative", "Mobile App v4.2", "Mobile Client"),
        ("The newly optimized web dashboard loads in under 500ms. Noticeably snappier.", "performance", "positive", "Web Portal v3.1", "Frontend App"),
    ]
    return random.choice(templates)


def generate_dataset(n_samples: int = 500) -> pd.DataFrame:
    start_date = datetime(2026, 8, 1, 9, 0)
    rows = []

    # Category generator pools
    baseline_generators = [
        (generate_payment_positive, 0.15),
        (generate_payment_negative, 0.12),
        (generate_login_positive, 0.12),
        (generate_login_negative, 0.10),
        (generate_delivery_positive, 0.15),
        (generate_delivery_negative, 0.12),
        (generate_support_positive, 0.12),
        (generate_support_negative, 0.08),
        (generate_performance, 0.04),
    ]

    for i in range(n_samples):
        # Days 0 to 27 (4 weeks)
        day_offset = random.randint(0, 27)
        current_date = start_date + timedelta(
            days=day_offset,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        # Injected incident: In days 18-27, payment negative complaints surge by >300%
        if day_offset >= 18:
            # 60% probability of payment failure during surge window
            if random.random() < 0.60:
                gen_fn = generate_payment_negative
            else:
                gen_fn = random.choices(
                    [g[0] for g in baseline_generators],
                    weights=[g[1] for g in baseline_generators],
                    k=1
                )[0]
        else:
            gen_fn = random.choices(
                [g[0] for g in baseline_generators],
                weights=[g[1] for g in baseline_generators],
                k=1
            )[0]

        text, category, sentiment, product, component = gen_fn()

        # Controlled missingness: 2% missing timestamp, 1% missing rating
        ts_str = current_date.strftime("%Y-%m-%d %H:%M:%S") if random.random() > 0.02 else None
        rating = (1 if sentiment == "negative" else 5) if random.random() > 0.01 else None

        rows.append({
            "id": f"saas_{i+1:04d}",
            "text": text,
            "timestamp": ts_str,
            "source": "SaaS Helpdesk & Store Reviews",
            "sentiment": sentiment,
            "rating": rating,
            "product": product,
            "category": category,
            "component": component,
            "dataset_type": "DEMO / SYNTHETIC DATA",
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset(500)
    out_dir = "data/samples"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ecommerce_saas_feedback_demo.csv")
    df.to_csv(out_path, index=False)
    unique_texts = df["text"].nunique()
    print(f"Generated {len(df)} records ({unique_texts} unique text strings) in {out_path}.")
    print("Dataset explicitly tagged: DEMO / SYNTHETIC DATA")

    # Truthful measurement of injected incident dynamics (FIX 12)
    valid_ts_df = df[df["timestamp"].notna()].copy()
    valid_ts_df["dt"] = pd.to_datetime(valid_ts_df["timestamp"])
    start_dt = valid_ts_df["dt"].min()
    valid_ts_df["day_offset"] = (valid_ts_df["dt"] - start_dt).dt.days

    baseline_mask = valid_ts_df["day_offset"] < 18
    incident_mask = valid_ts_df["day_offset"] >= 18

    n_base = int(baseline_mask.sum())
    n_inc = int(incident_mask.sum())

    pay_neg_base = int(((valid_ts_df["category"] == "payment") & (valid_ts_df["sentiment"] == "negative") & baseline_mask).sum())
    pay_neg_inc = int(((valid_ts_df["category"] == "payment") & (valid_ts_df["sentiment"] == "negative") & incident_mask).sum())

    prev_base = pay_neg_base / max(1, n_base)
    prev_inc = pay_neg_inc / max(1, n_inc)
    growth_pct = ((prev_inc - prev_base) / max(1e-5, prev_base)) * 100.0

    print(f"Empirical Injected Dynamics:")
    print(f"  Baseline Period (Days 0-17): {pay_neg_base} payment failures / {n_base} total ({prev_base * 100:.2f}%)")
    print(f"  Incident Period (Days 18-27): {pay_neg_inc} payment failures / {n_inc} total ({prev_inc * 100:.2f}%)")
    print(f"  Measured Prevalence Growth: {growth_pct:+.2f}%")
