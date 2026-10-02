"""Generates a labelled synthetic ticket set (used when no real CSV is supplied)."""
import random
import pandas as pd

random.seed(7)
CATS = {
    "Billing": [
        "I was charged twice for my subscription this month",
        "my invoice shows the wrong amount",
        "there is an unknown charge on my credit card statement",
        "the payment page keeps failing when I try to update my card",
        "I need a copy of my last three invoices for accounting",
    ],
    "Refund Request": [
        "I want a refund for the product I bought last week",
        "please refund my order, it arrived damaged",
        "I returned the item two weeks ago and still have no refund",
        "the item was not as described, I would like my money back",
        "cancel my order and send the refund to my original payment method",
    ],
    "Technical Issue": [
        "the app crashes every time I open the dashboard",
        "I get a 500 error when I upload a file",
        "the website is extremely slow and pages time out",
        "the sync feature stopped working after the update",
        "the API returns an error and our integration is broken",
    ],
    "Account Access": [
        "I cannot log in to my account, the password reset email never arrives",
        "my account was locked after several login attempts",
        "two factor authentication code is not working",
        "someone may have hacked my account, I see unknown logins",
        "I forgot my username and cannot recover my account",
    ],
    "Product Inquiry": [
        "does the premium plan include priority support",
        "what is the difference between the basic and pro versions",
        "is this product compatible with my operating system",
        "do you offer discounts for students or nonprofits",
        "when will the new feature be released",
    ],
    "Cancellation": [
        "I want to cancel my subscription effective immediately",
        "how do I close my account and delete my data",
        "please stop my membership renewal",
        "I am switching to another provider and want to terminate my plan",
        "cancel my plan before the next billing cycle",
    ],
}
PRIO_CUES = {
    "critical": ["URGENT production is down", "this is an emergency, all our users are affected",
                 "critical outage, we are losing money every minute", "security breach in progress"],
    "high": ["this is blocking my work today", "please respond as soon as possible",
             "I have a deadline tomorrow", "this is a serious problem for our team"],
    "medium": ["I would appreciate help when you can", "this has been bothering me for a few days",
               "please look into this", "could someone check this"],
    "low": ["no rush at all", "just a minor question whenever you have time",
            "not urgent, just curious", "when convenient, thanks"],
}
OPEN = ["Hi,", "Hello team,", "Hey,", "To whom it may concern,", ""]
CLOSE = ["Thanks.", "Regards.", "Thank you for your help.", "", "Please advise."]


def build(n=3000):
    rows = []
    for _ in range(n):
        cat = random.choice(list(CATS))
        prio = random.choices(list(PRIO_CUES), weights=[1, 2, 3, 2])[0]
        text = " ".join([random.choice(OPEN), random.choice(CATS[cat]) + ".",
                         random.choice(PRIO_CUES[prio]) + ".", random.choice(CLOSE)]).strip()
        if random.random() < 0.06:                 # label noise -> honest, not 100%
            cat = random.choice(list(CATS))
        rows.append((text, cat, prio))
    return pd.DataFrame(rows, columns=["text", "category", "priority"])


if __name__ == "__main__":
    build().to_csv("data/synthetic_tickets.csv", index=False)
    print("wrote data/synthetic_tickets.csv")
