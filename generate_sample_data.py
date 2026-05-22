import csv
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

categories = {
    "Electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "Charging Dock", "Laptop Stand"],
    "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Vacuum Cleaner", "Scented Candle", "Silicon SpatulaSet"],
    "Clothing": ["Cotton T-Shirt", "Denim Jacket", "Athletic Socks", "Woolen Sweater", "Baseball Cap"],
    "Sports & Outdoors": ["Yoga Mat", "Water Bottle", "Dumbbells (Pair)", "Camping Tent", "Running Shoes"],
    "Beauty & Personal Care": ["Face Moisturizer", "Sunscreen SPF 50", "Electric Toothbrush", "Lip Balm", "Hair Dryer"]
}

regions = ["North", "South", "East", "West"]
customer_first = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
customer_last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

start_date = datetime(2025, 1, 1)

with open("backend/sales_data_sample.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["order_id", "order_date", "customer_name", "product_category", "product_name", "revenue", "quantity", "region", "is_mobile"])
    
    for i in range(1, 551):
        order_id = f"ORD{1000 + i}"
        # Generate date distributed over 2025 Q1 and Q2
        days_offset = random.randint(0, 180)
        order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        
        customer_name = f"{random.choice(customer_first)} {random.choice(customer_last)}"
        product_category = random.choice(list(categories.keys()))
        product_name = random.choice(categories[product_category])
        
        # Quantity between 1 and 5
        quantity = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 7, 3])[0]
        
        # Base price depends on category and product to look realistic
        if product_category == "Electronics":
            base_price = random.uniform(30, 150)
        elif product_category == "Home & Kitchen":
            base_price = random.uniform(15, 120)
        elif product_category == "Clothing":
            base_price = random.uniform(10, 60)
        elif product_category == "Sports & Outdoors":
            base_price = random.uniform(15, 80)
        else: # Beauty
            base_price = random.uniform(8, 45)
            
        revenue = round(base_price * quantity, 2)
        region = random.choice(regions)
        is_mobile = random.choice([0, 1])
        
        writer.writerow([order_id, order_date, customer_name, product_category, product_name, revenue, quantity, region, is_mobile])

print("Successfully generated backend/sales_data_sample.csv with 550 rows!")
