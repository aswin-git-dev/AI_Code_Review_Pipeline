# calculator.py
# Demo module for STTP Workshop — AI Code Review & Auto-Doc Pipeline
# INTENTIONAL FLAWS: no docstrings, magic numbers, unclear variable names

def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def emi(principal, annual_rate, years):
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if annual_rate < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Loan tenure must be positive")
    
    monthly_rate = annual_rate / 12 / 100
    num_months = years * 12
    
    if monthly_rate == 0:
        return principal / num_months
    
    emi_value = (principal * monthly_rate * (1 + monthly_rate) ** num_months) / (
        (1 + monthly_rate) ** num_months - 1
    )
    return emi_value
    
def factorial(n):
    if n < 0:
        raise ValueError("Negative input")
    if n == 0 or n == 1:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def celsius_to_fahrenheit(c):
    return (c * 9 / 5) + 32


def discount_price(price, pct):
    if pct < 0 or pct > 100:
        raise ValueError("Invalid percentage")
    d = price * pct / 100
    return price - d


def bmi(w, h):
    if h <= 0:
        raise ValueError("Height must be positive")
    val = w / (h ** 2)
    if val < 18.5:
        return "Underweight"
    elif val < 25:
        return "Normal"
    elif val < 30:
        return "Overweight"
    else:
        return "Obese"
