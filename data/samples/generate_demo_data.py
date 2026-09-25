"""
Generates synthetic multi-domain SaaS / E-commerce customer feedback dataset.
Labeled explicitly as: DEMO / SYNTHETIC DATA per Section 49.
Used to evaluate cross-domain robustness (RQ5) alongside the real US Airline benchmark.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

COMPLAINT_TEMPLATES = [
    # Payment failures
    ("Payment failed on checkout with error 502 after v4.2 update.", "payment", "negative", "Mobile App v4.2", "Payment Gateway"),
    ("Credit card transaction rejected repeatedly on checkout.", "payment", "negative", "Mobile App v4.2", "Payment Gateway"),
    ("UPI transaction timed out but money was debited from my bank!", "payment", "negative", "Mobile App v4.2", "UPI Gateway"),
    ("Cannot complete payment, the buy button is unresponsive.", "payment", "negative", "Web Portal v3.1", "Checkout System"),
    ("Payment went through smoothly, quick checkout experience.", "payment", "positive", "Mobile App v4.1", "Payment Gateway"),
    ("Checkout was instant with Apple Pay, very convenient.", "payment", "positive", "Mobile App v4.2", "Apple Pay"),

    # Login and Auth
    ("App crashes every time I try to log in with Google OAuth.", "login", "negative", "Mobile App v4.2", "Auth Service"),
    ("Password reset link is broken and sends me to a blank screen.", "login", "negative", "Web Portal v3.1", "Auth Service"),
    ("FaceID login fails on iOS after the latest update.", "login", "negative", "Mobile App v4.2", "iOS Client"),
    ("Login was seamless and biometrics worked flawlessly.", "login", "positive", "Mobile App v4.1", "Auth Service"),

    # Delivery & Shipping
    ("Package was delivered 4 days late and box was crushed.", "delivery", "negative", "Logistics Partner B", "Delivery Depot"),
    ("Tracking number shows delivered but package is nowhere to be found.", "delivery", "negative", "Logistics Partner B", "Tracking API"),
    ("Super fast 1-day delivery, driver was polite and punctual.", "delivery", "positive", "Logistics Partner A", "Delivery Fleet"),
    ("Package arrived in perfect condition, great tracking updates.", "delivery", "positive", "Logistics Partner A", "Delivery Fleet"),

    # Customer Support
    ("Customer support agent hung up after placing me on hold for 40 minutes.", "customer_service", "negative", "Support Desk", "Agent Team"),
    ("No response to my refund ticket for over two weeks!", "customer_service", "negative", "Support Desk", "Ticketing System"),
    ("Support rep Sarah resolved my billing question within 5 minutes, fantastic service!", "customer_service", "positive", "Support Desk", "Agent Team"),
    ("Very helpful live chat agent helped reconfigure my subscription.", "customer_service", "positive", "Support Desk", "Live Chat"),

    # Performance & Speed
    ("Dashboard takes 30 seconds to load analytics reports.", "performance", "negative", "Web Portal v3.1", "Database Cluster"),
    ("High CPU usage and battery drain while app is open in background.", "performance", "negative", "Mobile App v4.2", "Mobile Client"),
    ("The new platform version is noticeably snappier and loads instantly.", "performance", "positive", "Web Portal v3.1", "Frontend App"),
]

start_date = datetime(2026, 8, 1, 9, 0)
rows = []

for i in range(500):
    # Inject temporal surge for payment failures in weeks 3 and 4
    day_offset = random.randint(0, 27)
    current_date = start_date + timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    
    if day_offset >= 18:
        # High surge of payment failures after "v4.2 update"
        weights = [0.35, 0.25, 0.20, 0.05, 0.02, 0.03] + [0.01] * (len(COMPLAINT_TEMPLATES) - 6)
        tmpl = random.choices(COMPLAINT_TEMPLATES, weights=weights, k=1)[0]
    else:
        tmpl = random.choice(COMPLAINT_TEMPLATES)

    text, aspect, sentiment, entity, component = tmpl
    rows.append({
        "id": f"saas_{i+1:04d}",
        "text": text,
        "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "SaaS Helpdesk & Store Reviews",
        "sentiment": sentiment,
        "rating": 1 if sentiment == "negative" else 5,
        "product": entity,
        "category": aspect,
        "component": component,
        "dataset_type": "DEMO / SYNTHETIC DATA",
    })

df = pd.DataFrame(rows)
os.makedirs("data/samples", exist_ok=True)
output_path = "data/samples/ecommerce_saas_feedback_demo.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} records in {output_path} (Explicitly tagged DEMO / SYNTHETIC DATA)")
