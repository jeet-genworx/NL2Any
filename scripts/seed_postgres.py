"""Repeatable synthetic PostgreSQL seed script."""

import random
from datetime import datetime, timedelta, timezone
import psycopg

from nl2anyquery.core.config import settings

# Fixed seed for repeatable synthetic data
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


def generate_postgres_seed_data() -> dict[str, list[tuple]]:
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # 1. Customers (100)
    customers = []
    for i in range(1, 101):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        email = f"user_{i}@example.com"
        phone = f"+1-555-{1000+i:04d}"
        city = random.choice(CITIES)
        created_at = start_date + timedelta(days=random.randint(0, 365))
        customers.append((i, name, email, phone, city, created_at))

    # 2. Products (50)
    products = []
    for i in range(1, 51):
        cat = random.choice(CATEGORIES)
        name = f"{cat} Item {i}"
        price = round(random.uniform(9.99, 499.99), 2)
        stock = random.randint(10, 500)
        created_at = start_date
        products.append((i, name, cat, price, stock, created_at))

    # 3. Employees (20)
    employees = []
    for i in range(1, 21):
        name = f"Employee {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        email = f"emp_{i}@company.internal"
        dept = random.choice(DEPARTMENTS)
        role = random.choice(ROLES)
        city = random.choice(CITIES)
        hire_date = start_date - timedelta(days=random.randint(100, 1000))
        employees.append((i, name, email, dept, role, city, hire_date))

    # 4. Orders (500)
    orders = []
    for i in range(1, 501):
        cust_id = random.randint(1, 100)
        order_date = start_date + timedelta(days=random.randint(50, 450), hours=random.randint(0, 23))
        status = random.choice(ORDER_STATUSES)
        shipping_city = random.choice(CITIES)
        orders.append((i, cust_id, order_date, status, 0.0, shipping_city))

    # 5. Order Items (1500)
    order_items = []
    order_totals: dict[int, float] = {i: 0.0 for i in range(1, 501)}
    item_id = 1
    for o_id in range(1, 501):
        # 2 to 4 items per order
        num_items = random.randint(2, 4)
        chosen_products = random.sample(range(1, 51), num_items)
        for p_id in chosen_products:
            qty = random.randint(1, 5)
            unit_price = products[p_id - 1][3]
            line_total = round(qty * unit_price, 2)
            order_totals[o_id] += line_total
            order_items.append((item_id, o_id, p_id, qty, unit_price))
            item_id += 1

    # Update order total amounts
    updated_orders = [
        (o[0], o[1], o[2], o[3], round(order_totals[o[0]], 2), o[5])
        for o in orders
    ]

    # 6. Support Tickets (300)
    support_tickets = []
    for i in range(1, 301):
        cust_id = random.randint(1, 100)
        emp_id = random.randint(1, 20)
        status = random.choice(TICKET_STATUSES)
        priority = random.choice(TICKET_PRIORITIES)
        subject = f"Issue regarding order {random.randint(1, 500)}"
        created_at = start_date + timedelta(days=random.randint(60, 450))
        resolved_at = created_at + timedelta(days=random.randint(1, 7)) if status in ("resolved", "closed") else None
        support_tickets.append((i, cust_id, emp_id, subject, status, priority, created_at, resolved_at))

    return {
        "customers": customers,
        "products": products,
        "employees": employees,
        "orders": updated_orders,
        "order_items": order_items,
        "support_tickets": support_tickets,
    }


def seed_postgres(dsn: str | None = None) -> None:
    dsn = dsn or settings.postgres_dsn
    if not dsn:
        raise ValueError("PostgreSQL connection string is required. Set POSTGRES_DSN in .env.")

    print(f"Connecting to PostgreSQL: {dsn}")
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Drop tables in reverse dependency order
            print("Dropping existing tables...")
            cur.execute("""
                DROP TABLE IF EXISTS support_tickets CASCADE;
                DROP TABLE IF EXISTS order_items CASCADE;
                DROP TABLE IF EXISTS orders CASCADE;
                DROP TABLE IF EXISTS employees CASCADE;
                DROP TABLE IF EXISTS products CASCADE;
                DROP TABLE IF EXISTS customers CASCADE;
            """)

            # Create tables
            print("Creating relational schema tables...")
            cur.execute("""
                CREATE TABLE customers (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(150) NOT NULL,
                    phone VARCHAR(50),
                    city VARCHAR(50),
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );

                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    price NUMERIC(10, 2) NOT NULL,
                    stock_quantity INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );

                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(150) NOT NULL,
                    department VARCHAR(50) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    city VARCHAR(50),
                    hire_date TIMESTAMP WITH TIME ZONE NOT NULL
                );

                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
                    order_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    total_amount NUMERIC(10, 2) NOT NULL,
                    shipping_city VARCHAR(50)
                );

                CREATE TABLE order_items (
                    id INTEGER PRIMARY KEY,
                    order_id INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
                    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                    quantity INTEGER NOT NULL,
                    unit_price NUMERIC(10, 2) NOT NULL
                );

                CREATE TABLE support_tickets (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
                    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
                    subject VARCHAR(200) NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    priority VARCHAR(30) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    resolved_at TIMESTAMP WITH TIME ZONE
                );
            """)

            print("Generating synthetic data...")
            data = generate_postgres_seed_data()

            print(f"Inserting {len(data['customers'])} customers...")
            cur.executemany(
                "INSERT INTO customers (id, name, email, phone, city, created_at) VALUES (%s, %s, %s, %s, %s, %s);",
                data["customers"],
            )

            print(f"Inserting {len(data['products'])} products...")
            cur.executemany(
                "INSERT INTO products (id, name, category, price, stock_quantity, created_at) VALUES (%s, %s, %s, %s, %s, %s);",
                data["products"],
            )

            print(f"Inserting {len(data['employees'])} employees...")
            cur.executemany(
                "INSERT INTO employees (id, name, email, department, role, city, hire_date) VALUES (%s, %s, %s, %s, %s, %s, %s);",
                data["employees"],
            )

            print(f"Inserting {len(data['orders'])} orders...")
            cur.executemany(
                "INSERT INTO orders (id, customer_id, order_date, status, total_amount, shipping_city) VALUES (%s, %s, %s, %s, %s, %s);",
                data["orders"],
            )

            print(f"Inserting {len(data['order_items'])} order items...")
            cur.executemany(
                "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES (%s, %s, %s, %s, %s);",
                data["order_items"],
            )

            print(f"Inserting {len(data['support_tickets'])} support tickets...")
            cur.executemany(
                "INSERT INTO support_tickets (id, customer_id, employee_id, subject, status, priority, created_at, resolved_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                data["support_tickets"],
            )

    print("PostgreSQL seed complete!")


if __name__ == "__main__":
    seed_postgres()
