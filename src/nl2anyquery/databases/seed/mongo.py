"""MongoDB synthetic database seeder."""

import random
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient

from nl2anyquery.core.config import settings

random.seed(42)

CITIES = ["New York", "Chicago", "San Francisco", "Austin", "Seattle", "Boston", "Denver", "Miami"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Sports", "Beauty"]
ORDER_STATUSES = ["completed", "pending", "cancelled", "shipped"]
TICKET_STATUSES = ["open", "in_progress", "resolved", "closed"]
TICKET_PRIORITIES = ["low", "medium", "high", "urgent"]
DEPARTMENTS = ["Support", "Engineering", "Sales", "Operations"]
ROLES = ["Specialist", "Senior Specialist", "Team Lead", "Manager"]

FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Anderson"]


def generate_mongo_seed_data() -> dict[str, list[dict]]:
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # 1. Customers (100)
    customers = []
    for i in range(1, 101):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        city = random.choice(CITIES)
        customers.append({
            "_id": f"cust_{i:04d}",
            "name": name,
            "email": f"user_{i}@example.com",
            "phone": f"+1-555-{1000+i:04d}",
            "address": {
                "city": city,
                "country": "USA",
                "zip_code": f"{10000 + i}",
            },
            "created_at": start_date + timedelta(days=random.randint(0, 365)),
        })

    # 2. Products (50)
    products = []
    for i in range(1, 51):
        cat = random.choice(CATEGORIES)
        products.append({
            "_id": f"prod_{i:04d}",
            "name": f"{cat} Item {i}",
            "category": cat,
            "price": round(random.uniform(9.99, 499.99), 2),
            "stock_quantity": random.randint(10, 500),
            "created_at": start_date,
        })

    # 3. Employees (20)
    employees = []
    for i in range(1, 21):
        employees.append({
            "_id": f"emp_{i:04d}",
            "name": f"Employee {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "email": f"emp_{i}@company.internal",
            "department": random.choice(DEPARTMENTS),
            "role": random.choice(ROLES),
            "city": random.choice(CITIES),
            "hire_date": start_date - timedelta(days=random.randint(100, 1000)),
        })

    # 4. Orders (500) - Document-oriented with embedded items
    orders = []
    for i in range(1, 501):
        cust_id = f"cust_{random.randint(1, 100):04d}"
        order_date = start_date + timedelta(days=random.randint(50, 450), hours=random.randint(0, 23))
        status = random.choice(ORDER_STATUSES)
        shipping_city = random.choice(CITIES)

        num_items = random.randint(2, 4)
        chosen_pids = random.sample(range(1, 51), num_items)
        items = []
        total_amount = 0.0
        for pid_int in chosen_pids:
            prod_info = products[pid_int - 1]
            qty = random.randint(1, 5)
            u_price = prod_info["price"]
            total_amount += qty * u_price
            items.append({
                "product_id": prod_info["_id"],
                "product_name": prod_info["name"],
                "quantity": qty,
                "price": u_price,
            })

        orders.append({
            "_id": f"order_{i:04d}",
            "customer_id": cust_id,
            "created_at": order_date,
            "status": status,
            "total_amount": round(total_amount, 2),
            "shipping_city": shipping_city,
            "items": items,
        })

    # 5. Support Tickets (300)
    support_tickets = []
    for i in range(1, 301):
        cust_id = f"cust_{random.randint(1, 100):04d}"
        emp_id = f"emp_{random.randint(1, 20):04d}"
        status = random.choice(TICKET_STATUSES)
        priority = random.choice(TICKET_PRIORITIES)
        created_at = start_date + timedelta(days=random.randint(60, 450))
        resolved_at = created_at + timedelta(days=random.randint(1, 7)) if status in ("resolved", "closed") else None

        support_tickets.append({
            "_id": f"ticket_{i:04d}",
            "customer_id": cust_id,
            "assigned_employee_id": emp_id,
            "subject": f"Issue regarding order {random.randint(1, 500)}",
            "status": status,
            "priority": priority,
            "created_at": created_at,
            "resolved_at": resolved_at,
        })

    return {
        "customers": customers,
        "products": products,
        "employees": employees,
        "orders": orders,
        "support_tickets": support_tickets,
    }


def seed_mongo(uri: str | None = None, database: str | None = None) -> None:
    uri = uri or settings.mongodb_uri
    db_name = database or settings.mongodb_database
    if not uri:
        raise ValueError("MongoDB URI is required. Set MONGODB_URI in .env.")

    print(f"Connecting to MongoDB: {uri}, DB: {db_name}")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]

    print("Generating synthetic document data...")
    data = generate_mongo_seed_data()

    for coll_name, docs in data.items():
        print(f"Dropping collection '{coll_name}'...")
        db[coll_name].drop()
        print(f"Inserting {len(docs)} documents into '{coll_name}'...")
        db[coll_name].insert_many(docs)

    client.close()
    print("MongoDB seed complete!")
