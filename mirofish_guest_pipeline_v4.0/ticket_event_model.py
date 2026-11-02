"""Ticket taxonomy/organic-message event model for the hotel simulator.

Notes:
- Recommendation proportions are P(service_type, category, item | detected recommendation)
  in a three-property sample, *not* probabilities of a guest experiencing an issue.
- organic_message_event() uses the user-supplied ~18% detection frequency per
  eligible *guest message*, not per guest or per stay.
- Do not use that baseline as the fixed response or disclosure rate for
  different status-check treatments. Those rates are experimental outcomes.
"""

import json
import random
from pathlib import Path

import pandas as pd


class TicketEventModel:
    def __init__(self, csv_path, default_message_rate=0.18, property_rules_path=None):
        if not (0 <= default_message_rate <= 1):
            raise ValueError("Default per-message detection rate must be in [0, 1]")
        df = pd.read_csv(csv_path, keep_default_na=False)
        required = {"service_type", "category", "item", "recommendations"}
        if required - set(df.columns):
            raise ValueError(f"Missing ticket columns: {sorted(required - set(df.columns))}")
        df["recommendations"] = pd.to_numeric(df["recommendations"], errors="coerce")
        df = df[df["recommendations"].gt(0)].copy()
        if df.empty:
            raise ValueError("Ticket CSV has no positive recommendation weights")
        self.items = df.to_dict("records")
        self.default_message_rate = float(default_message_rate)
        self.rules = {}
        if property_rules_path:
            self.rules = json.loads(Path(property_rules_path).read_text(encoding="utf-8"))
            if not isinstance(self.rules, dict):
                raise ValueError("Property rules must be a JSON object keyed by company_id")

    def property_settings(self, company_id):
        entry = self.rules.get(str(company_id), {})
        if not isinstance(entry, dict):
            raise ValueError(f"Property rules for {company_id} must be an object")
        rate = float(entry.get("organic_message_ticket_rate", self.default_message_rate))
        if not (0 <= rate <= 1):
            raise ValueError(f"Invalid organic_message_ticket_rate for {company_id}: {rate}")
        return rate, entry

    def eligible_items(self, company_id):
        _, rule = self.property_settings(company_id)
        includes = set(rule.get("include_categories", []))
        excludes = set(rule.get("exclude_categories", []))
        excluded_items = set(rule.get("exclude_items", []))
        eligible = [r for r in self.items if
                    (not includes or r["category"] in includes) and
                    r["category"] not in excludes and
                    r["item"] not in excluded_items]
        return eligible

    def draw_type(self, company_id, rng):
        """Sample conditional recommendation *type*, not issue prevalence."""
        options = self.eligible_items(company_id)
        if not options:
            return None
        item = rng.choices(options, weights=[int(r["recommendations"]) for r in options], k=1)[0]
        return {
            "service_type": item["service_type"],
            "category": item["category"],
            "item": item["item"] or None,
            "source": "ticket_detection_three_property_pool",
            "sampled_conditionally_on": "a modeled detection or controlled scenario",
        }

    def organic_message_event(self, company_id, rng):
        """Draw for ONE eligible guest-initiated message in the future simulator.

        Calling this for a stay rather than a message would be a unit error.
        Never use it to fix the disclosure rate of a randomized status-check arm.
        """
        message_rate, _ = self.property_settings(company_id)
        if rng.random() >= message_rate:
            return None
        return self.draw_type(company_id, rng)
