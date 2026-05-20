#!/usr/bin/env python3
import json
import logging
from typing import List, Optional, Any, Dict
from latency_injector import LatencyInjector

class CallLogger:
    def __init__(self):
        self.calls = []
    def log(self, function_name, args, result):
        self.calls.append({"function": function_name, "args": args, "result": result})

# ── Travel & Identity ───────────────────────────────────────────
def search_flights(destination: str, date: str, **kwargs) -> dict:
    return {"status": "success", "flights": [{"flight_id": "FL123", "destination": destination, "date": date, "price": 450.0}]}

def book_flight(passenger_name: str, flight_id: Optional[str] = "FL123", **kwargs) -> dict:
    return {"status": "success", "booking_ref": "B789", "passenger": passenger_name}

def update_identity_doc(doc_type: str, doc_number: str, **kwargs) -> dict:
    return {"status": "success", "updated_doc": doc_type, "masked_number": doc_number[-4:]}

# ── Finance & Billing ───────────────────────────────────────────
def get_card_benefits(card_type: str, **kwargs) -> dict:
    return {"status": "success", "card_type": card_type, "benefits": ["2% Cashback", "No Foreign Transaction Fee"]}

def get_exchange_rate(amount: float, from_currency: str, to_currency: str, **kwargs) -> dict:
    rate = 1.1 if from_currency == "EUR" else 0.9
    return {"status": "success", "converted_amount": float(amount) * rate, "rate": rate}

def modify_autopay(bill_type: str, source_account: str, **kwargs) -> dict:
    return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source_account}

# ── Housing & Location ───────────────────────────────────────────
def search_apartments(city: str, bedrooms: int, max_price: float, **kwargs) -> dict:
    return {"status": "success", "city": city, "results": [{"id": "APT1", "price": max_price - 100, "beds": bedrooms}]}

def calculate_commute(origin_address: str, destination_address: str, mode: str = "driving", **kwargs) -> dict:
    return {"status": "success", "duration_mins": 25, "mode": mode}

def update_search_filter(filter_name: str, value: Any, **kwargs) -> dict:
    return {"status": "success", "filter_updated": filter_name, "new_value": value}

# ── E-Commerce Support ───────────────────────────────────────────
def track_order(order_id: str, **kwargs) -> dict:
    return {"status": "success", "order_id": order_id, "shipping_status": "Out for delivery"}

def search_products(query: str, max_price: Optional[float] = None, **kwargs) -> dict:
    price = max_price - 10 if max_price else 99.99
    return {"status": "success", "products": [{"product_id": "PROD1", "name": f"{query} Premium", "price": price}]}

def add_to_cart(product_id: str, quantity: int, **kwargs) -> dict:
    return {"status": "success", "product_id": product_id, "quantity": quantity, "cart_total": 99.99 * quantity}

class MockAPIRegistry:
    FUNCTIONS = {
        "search_flights": search_flights,
        "book_flight": book_flight,
        "update_identity_doc": update_identity_doc,
        "get_card_benefits": get_card_benefits,
        "get_exchange_rate": get_exchange_rate,
        "modify_autopay": modify_autopay,
        "search_apartments": search_apartments,
        "calculate_commute": calculate_commute,
        "update_search_filter": update_search_filter,
        "track_order": track_order,
        "search_products": search_products,
        "add_to_cart": add_to_cart,
    }

    def __init__(self, latency_profile="instant", enable_logging=True):
        self.injector = LatencyInjector(profile=latency_profile)
        self.logger = CallLogger() if enable_logging else None

    def call(self, function_name: str, **kwargs) -> dict:
        func = self.FUNCTIONS.get(function_name)
        if func is None: return {"status": "error", "message": f"Unknown function: {function_name}"}
        self.injector.inject(function_name)
        result = func(**kwargs)
        if self.logger: self.logger.log(function_name, kwargs, result)
        return result
