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
