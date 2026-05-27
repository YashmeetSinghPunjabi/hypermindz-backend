import os
import sqlite3
import pandas as pd
import re
from fastapi import HTTPException
import db_manager

# ==========================================
# 1. PROGRAMMATIC SECURITY ENFORCEMENT
# ==========================================

class SQLSecurityValidator:
    """
    Enforces a strict programmatic read-only sandbox.
    Blocks hazardous commands and structural mutations.
    """
    FORBIDDEN_KEYWORDS = {'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'REPLACE', 'ATTACH', 'DETACH'}

    @classmethod
    def validate_query(cls, sql: str) -> str:
        # Clean up code blocks if any slipped through
        cleaned_sql = sql.strip().strip("`").replace("sql\n", "").strip()
        upper_sql = cleaned_sql.upper()

        # Rule 1: Block obvious stacked queries (semi-colon injection)
        if ";" in cleaned_sql and cleaned_sql.rstrip(";").count(";") > 0:
            raise HTTPException(status_code=400, detail="Security Violation: Multi-statement executions are forbidden.")

        # Rule 2: Token keyword assertion to block modification
        tokens = set(re.findall(r'\b\w+\b', upper_sql))
        intercepted = cls.FORBIDDEN_KEYWORDS.intersection(tokens)
        if intercepted:
            raise HTTPException(status_code=400, detail=f"Security Violation: Structural mutations or database attachments are blocked: {list(intercepted)}")

        # Rule 3: Only permit safe analytical reading queries
        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
            raise HTTPException(status_code=400, detail="Security Violation: Only explicit SELECT analytical queries are allowed.")

        return cleaned_sql

# ==========================================
# 2. HELPER UTILITIES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_user_db_path(user_id: str, file_id: str) -> str:
    """Maps user file directly to an isolated SQLite file path."""
    return os.path.join(BASE_DIR, f"db_{user_id}_{file_id}.sqlite")

def get_db_connection(user_id: str, file_id: str) -> sqlite3.Connection:
    db_path = get_user_db_path(user_id, file_id)
    return sqlite3.connect(db_path)

def extract_schema_context(conn: sqlite3.Connection, table_name: str) -> tuple:
    """Inspects structural layout and fetches small data footprint for LLM positioning."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns_info = cursor.fetchall()
    schema_str = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])
    
    # Fetch a 3-row sample footprint so Gemini knows what the data actually looks like
    df_sample = pd.read_sql_query(f"SELECT * FROM `{table_name}` LIMIT 3;", conn)
    sample_rows = df_sample.to_dict(orient="records")
    return schema_str, sample_rows

def ensure_sample_files_exist():
    """Generates the three domain-specific sample CSV datasets if they do not exist."""
    # 1. Sales Data Sample
    sales_path = os.path.join(BASE_DIR, "sales_data_sample.csv")
    if not os.path.exists(sales_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(42)
        categories = {
            "Electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "Charging Dock", "Laptop Stand"],
            "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Vacuum Cleaner", "Scented Candle", "Silicon SpatulaSet"],
            "Clothing": ["Cotton T-Shirt", "Denim Jacket", "Athletic Socks", "Woolen Sweater", "Baseball Cap"],
            "Sports & Outdoors": ["Yoga Mat", "Water Bottle", "Dumbbells (Pair)", "Camping Tent", "Running Shoes"],
            "Beauty & Personal Care": ["Face Moisturizer", "Sunscreen SPF 50", "Electric Toothbrush", "Lip Balm", "Hair Dryer"]
        }
        regions = ["North", "South", "East", "West"]
        customer_first = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
        customer_last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        start_date = datetime(2025, 1, 1)
        with open(sales_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["order_id", "order_date", "customer_name", "product_category", "product_name", "revenue", "quantity", "region", "is_mobile"])
            for i in range(1, 201):
                order_id = f"ORD{1000 + i}"
                days_offset = random.randint(0, 90)
                order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                customer_name = f"{random.choice(customer_first)} {random.choice(customer_last)}"
                product_category = random.choice(list(categories.keys()))
                product_name = random.choice(categories[product_category])
                quantity = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
                base_price = random.uniform(10, 100)
                revenue = round(base_price * quantity, 2)
                region = random.choice(regions)
                is_mobile = random.choice([0, 1])
                writer.writerow([order_id, order_date, customer_name, product_category, product_name, revenue, quantity, region, is_mobile])

    # 2. Marketing Campaign Performance Data Sample
    marketing_path = os.path.join(BASE_DIR, "marketing_data_sample.csv")
    if not os.path.exists(marketing_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(100)
        channels = ["Google Ads", "Meta Ads", "LinkedIn Ads", "TikTok Ads", "YouTube Ads"]
        campaign_names = {
            "Google Ads": ["Search_Brand_Core", "Search_NonBrand_Generic", "PerformanceMax_Retail"],
            "Meta Ads": ["Prospecting_Lookalike_US", "Retargeting_AddToCart", "Instagram_Stories_Promo"],
            "LinkedIn Ads": ["B2B_DecisionMakers", "JobHolders_TechSkill"],
            "TikTok Ads": ["InFeedVideo_GenZ_Trend", "SparkAds_Influencer"],
            "YouTube Ads": ["PreRoll_BrandAwareness", "BumperAds_SummerSale"]
        }
        start_date = datetime(2025, 3, 1)
        with open(marketing_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["campaign_id", "date", "campaign_name", "channel", "spend", "impressions", "clicks", "conversions", "revenue"])
            for i in range(1, 151):
                campaign_id = f"CMP{2000 + i}"
                days_offset = random.randint(0, 60)
                date_str = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                channel = random.choice(channels)
                campaign_name = random.choice(campaign_names[channel])
                spend = round(random.uniform(50, 1500), 2)
                cpc = random.uniform(0.15, 4.5)
                clicks = max(1, int(spend / cpc))
                ctr = random.uniform(0.005, 0.05)
                impressions = int(clicks / ctr)
                conv_rate = random.uniform(0.01, 0.08)
                conversions = int(clicks * conv_rate)
                avg_order_value = random.uniform(40, 180)
                revenue = round(conversions * avg_order_value, 2)
                writer.writerow([campaign_id, date_str, campaign_name, channel, spend, impressions, clicks, conversions, revenue])

    # 3. Product Engagement Data Sample
    product_path = os.path.join(BASE_DIR, "product_data_sample.csv")
    if not os.path.exists(product_path):
        import csv
        import random
        from datetime import datetime, timedelta
        random.seed(200)
        devices = ["iOS", "Android", "Web"]
        features = ["dashboard", "catalog", "settings", "query_compiler", "export_pdf"]
        start_date = datetime(2025, 4, 1)
        with open(product_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "date", "device", "session_duration_mins", "pages_viewed", "actions_performed", "active_feature", "churned"])
            for i in range(1, 151):
                user_id = f"USR{5000 + random.randint(1, 50)}"
                days_offset = random.randint(0, 30)
                date_str = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                device = random.choice(devices)
                session_duration_mins = round(random.uniform(2.5, 45.0), 1)
                pages_viewed = random.randint(1, int(session_duration_mins * 0.8) + 2)
                actions_performed = random.randint(pages_viewed, pages_viewed * 5)
                active_feature = random.choice(features)
                churned = random.choices([0, 1], weights=[92, 8])[0]
                writer.writerow([user_id, date_str, device, session_duration_mins, pages_viewed, actions_performed, active_feature, churned])

def seed_sample_data(user_id: str):
    """Automatically seeds the three domain-specific sample datasets for a user."""
    ensure_sample_files_exist()
    
    datasets = [
        {
            "filename": "sales_data_sample.csv",
            "file_id": "sample_ecommerce_sales",
            "display_name": "sales_data_sample.csv (Seeded E-Commerce)",
            "table_name": "data_sample_ecommerce_sales"
        },
        {
            "filename": "marketing_data_sample.csv",
            "file_id": "sample_marketing_campaigns",
            "display_name": "marketing_data_sample.csv (Seeded Ad Performance)",
            "table_name": "data_sample_marketing_campaigns"
        },
        {
            "filename": "product_data_sample.csv",
            "file_id": "sample_product_engagement",
            "display_name": "product_data_sample.csv (Seeded App Engagement)",
            "table_name": "data_sample_product_engagement"
        }
    ]
    
    for ds in datasets:
        unique_file_id = f"{user_id}_{ds['file_id']}"
        if db_manager.get_file_by_id(unique_file_id, user_id):
            continue
            
        sample_csv_path = os.path.join(BASE_DIR, ds["filename"])
        if not os.path.exists(sample_csv_path):
            continue
            
        try:
            df = pd.read_csv(sample_csv_path)
            table_name = ds["table_name"]
            
            df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]
            df.columns = [''.join(e for e in c if e.isalnum() or e == '_') for c in df.columns]
            
            db_path = get_user_db_path(user_id, unique_file_id)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            db_manager.add_file(
                user_id=user_id,
                file_name=ds["display_name"],
                table_name=table_name,
                row_count=len(df),
                columns=list(df.columns),
                sample_rows=df.head(3).to_dict(orient="records"),
                custom_file_id=unique_file_id
            )
        except Exception as e:
            print(f"Error seeding sample dataset {ds['filename']}: {e}")
