from fastapi import FastAPI
from pydantic import BaseModel
#from blacklist import BLACKLISTED_IPS, BLACKLISTED_DEVICES
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# ------------------------------
# Models & data
# ------------------------------
class Transaction(BaseModel):
    user_id: str
    amount: float
    country: str
    device: str
    ip_address: str
    timestamp: str

user_transactions = defaultdict(list)

# Blacklist check
BLACKLISTED_IPS = {"192.168.1.100", "10.0.0.5"}
BLACKLISTED_DEVICES = {"unknown", "fraud-device-001"}

# Velocity check
def check_velocity(user_id, timestamp, limit=3, minutes=5):
    now = datetime.fromisoformat(timestamp)
    user_transactions[user_id] = [t for t in user_transactions[user_id] if now - t < timedelta(minutes=minutes)]
    user_transactions[user_id].append(now)
    return len(user_transactions[user_id]) > limit

# ML model
df = pd.read_csv("historical.csv")
X = pd.get_dummies(df[["amount","country"]])
y = df["is_fraud"]
model = LogisticRegression()
model.fit(X, y)

def ml_risk_score(transaction):
    x_test = pd.get_dummies(pd.DataFrame([{"amount": transaction.amount, "country": transaction.country}]))
    x_test = x_test.reindex(columns=X.columns, fill_value=0)
    return model.predict_proba(x_test)[0][1] * 100  # percentage risk

# ------------------------------
# Endpoint
# ------------------------------
@app.post("/Score")
def score_transaction(transaction: Transaction):
    risk_score = 0

    # ---- Rule-based checks ----
    if transaction.amount > 200:
        risk_score += 30
    if transaction.country not in ["US"]:
        risk_score += 20
    if transaction.device in BLACKLISTED_DEVICES:
        risk_score += 40
    if transaction.ip_address in BLACKLISTED_IPS:
        risk_score += 40
    if check_velocity(transaction.user_id, transaction.timestamp):
        risk_score += 30

    # ---- Step 6: Add ML score ----
    ml_score = ml_risk_score(transaction)
    risk_score += ml_score * 0.5  # weight ML score 50%

    # ---- Decision ----
    decision = "approve"
    if risk_score > 50:
        decision = "review"
    if risk_score > 80:
        decision = "decline"

    return {"risk_score": risk_score, "decision": decision}