"""FinOpsIQ database synthetic seeder.

Unlike the demo Postgres and Mongo seeders, this one does **not** create its
tables. `finopsiq_be` carries a real, Alembic-managed schema of 68 tables
across three namespaces, and the canonical TOML that query processing reads was
extracted from it; recreating any of that here would mean two definitions of
the same database drifting apart. The tables are taken as they are and filled.

What it fills is the accounts-payable / accounts-receivable core -- the tables a
question is actually likely to be about:

    auth        organisations, users, roles, user_roles, plans
    finopsiq    tenant_profiles, vendors, customers, products, currency_rates,
                invoices (+ line items), purchase_orders (+ line items),
                contracts, payment_history, invoice_reminder_state, reminders,
                discrepancies_details, discrepancy_resolution

Everything else -- Alembic bookkeeping, the tax-rate caches, the audit logs, the
connector and mailbox plumbing, the document-parsing tables -- is left empty on
purpose. Those are operational scaffolding rather than business facts, and rows
in them would be noise in a result set without making any question answerable.

Two tenants are seeded rather than one, so that tenant_id is a real distinction
a query can get wrong rather than a column with one value in it. Status
vocabularies are taken from the CHECK constraints the schema already declares
(genbooks_requests for payables, its ar_status for receivables), so the data
speaks the application's own language instead of an invented one.

Re-runnable: every table it owns is emptied first, in foreign-key order, and no
table it does not own is touched.
"""

import json
import random
import uuid
from datetime import date, datetime, timedelta, timezone

import psycopg
from psycopg.types.json import Jsonb

from backend.src.config import settings

random.seed(20251013)

# A fixed namespace makes every id reproducible, so a re-seed produces the same
# database and a question that worked yesterday still works today.
NS = uuid.UUID("6f1b8d2e-5a43-4c7e-9f10-2b7c4d8e9a31")


def _id(*parts: object) -> uuid.UUID:
    return uuid.uuid5(NS, ":".join(str(p) for p in parts))


START = datetime(2025, 1, 6, tzinfo=timezone.utc)
TODAY = datetime(2026, 10, 13, tzinfo=timezone.utc)

TENANTS = [
    {
        "slug": "genworx",
        "name": "GenWorx Technologies",
        "legal": "GenWorx Technologies Pvt Ltd",
        "domain": "genworx.ai",
        "country": "India",
        "state": "Telangana",
        "currency": "INR",
        "industry": "Software",
        "timezone": "Asia/Kolkata",
    },
    {
        "slug": "northwind-labs",
        "name": "Northwind Labs",
        "legal": "Northwind Laboratories Inc",
        "domain": "northwindlabs.com",
        "country": "United States",
        "state": "California",
        "currency": "USD",
        "industry": "Biotechnology",
        "timezone": "America/Los_Angeles",
    },
]

VENDOR_NAMES = [
    "Acme Cloud Services", "Blue River Logistics", "Cedarline Office Supplies",
    "Delta Secure Printing", "Eastgate Facilities", "Fairmount Consulting",
    "Granite Data Systems", "Harborview Travel", "Ironwood Legal Partners",
    "Juniper Analytics", "Keystone Hardware", "Lakeshore Marketing",
    "Meridian Staffing", "Northpoint Telecom",
]

CUSTOMER_NAMES = [
    "Apex Retail Group", "Brightway Health", "Copperfield Energy",
    "Dunemark Shipping", "Everline Media", "Foxglove Pharma",
    "Greenstone Bank", "Hollis Manufacturing", "Inlet Robotics",
    "Jasper Education", "Kirkwall Insurance", "Lowtide Foods",
]

PRODUCTS = [
    ("CLD-COMPUTE", "Managed cloud compute, per vCPU-month"),
    ("CLD-STORAGE", "Object storage, per TB-month"),
    ("CLD-EGRESS", "Network egress, per TB"),
    ("LIC-SEAT-STD", "Software licence, standard seat, annual"),
    ("LIC-SEAT-ENT", "Software licence, enterprise seat, annual"),
    ("SUP-PREMIUM", "Premium support retainer, monthly"),
    ("SUP-ONCALL", "After-hours on-call coverage, monthly"),
    ("CON-ARCH", "Solution architecture consulting, per day"),
    ("CON-DATA", "Data engineering consulting, per day"),
    ("CON-SEC", "Security review engagement, per day"),
    ("HW-LAPTOP-14", "Developer laptop, 14-inch"),
    ("HW-MONITOR-27", "Monitor, 27-inch 4K"),
    ("HW-DOCK", "Docking station"),
    ("HW-HEADSET", "Noise-cancelling headset"),
    ("OFF-DESK", "Height-adjustable desk"),
    ("OFF-CHAIR", "Ergonomic task chair"),
    ("OFF-PAPER", "Printer paper, per box"),
    ("TRN-ONBOARD", "Onboarding training, per cohort"),
    ("TRN-CERT", "Certification voucher"),
    ("MKT-EVENT", "Event sponsorship package"),
    ("MKT-CONTENT", "Content production retainer, monthly"),
    ("LGL-RETAINER", "Legal retainer, monthly"),
    ("FAC-CLEANING", "Facilities cleaning, monthly"),
    ("FAC-SECURITY", "Site security, monthly"),
]

# Vocabularies the schema itself declares, on genbooks_requests.status and
# .ar_status. Reusing them keeps the seeded data in the application's language.
AP_STATUSES = ["yet_to_be_processed", "needs_review", "overdue", "ready_to_pay", "paid", "rejected"]
AR_STATUSES = ["draft", "sent", "overdue", "partially_paid", "paid", "attention_required"]
CURRENCIES = ["USD", "INR", "EUR", "GBP", "SGD"]
USD_RATES = {"USD": 1.0, "INR": 0.0120, "EUR": 1.0850, "GBP": 1.2700, "SGD": 0.7400}

DISCREPANCY_TYPES = [
    "price_mismatch", "quantity_mismatch", "missing_purchase_order",
    "duplicate_invoice", "tax_mismatch", "total_mismatch",
]
RESOLUTION_STATUSES = ["Awaiting_Review", "Email_Sent", "Closed", "Manual_Review"]
REMINDER_STATUSES = ["SENT", "FAILED", "SKIPPED"]
REMINDER_STATE_STATUSES = ["ACTIVE", "PAID", "ON_HOLD", "VOID"]

FIRST = ["Aarav", "Priya", "Rohan", "Meera", "Sanjay", "Divya", "Noah", "Elena",
         "Marcus", "Chloe", "Tomas", "Ingrid", "Yusuf", "Lena"]
LAST = ["Villuri", "Shah", "Nair", "Kapoor", "Reddy", "Bose", "Whitaker", "Moreau",
        "Okafor", "Lindqvist", "Alvarez", "Novak", "Haddad", "Sorensen"]
PLATFORM_ROLES = [
    ("owner", "Full control of the organisation, including billing"),
    ("finance_admin", "Approves payables and resolves discrepancies"),
    ("accountant", "Records payments and reconciles invoices"),
    ("viewer", "Read-only access to documents and reports"),
]
PLANS = [
    ("starter", 10, ["finopsiq"]),
    ("growth", 50, ["finopsiq", "genbooks"]),
    ("enterprise", 500, ["finopsiq", "genbooks", "connector_service"]),
]


def _money(low: float, high: float) -> float:
    return round(random.uniform(low, high), 2)


def _addr(city: str, country: str) -> dict:
    return {
        "line1": f"{random.randint(1, 400)} {random.choice(['Park', 'Hill', 'Market', 'Station', 'Lake'])} Road",
        "city": city,
        "state": random.choice(["CA", "NY", "TS", "KA", "TX"]),
        "postal_code": f"{random.randint(10000, 99999)}",
        "country": country,
    }


def generate() -> dict[str, list[tuple]]:
    """Build every row set, with the arithmetic between them made to agree."""
    rows: dict[str, list[tuple]] = {}

    # ---------------- auth ----------------
    plans = [
        (_id("plan", name), name, seats, Jsonb(products), name == "growth", Jsonb({"tier": name}))
        for name, seats, products in PLANS
    ]
    rows["plans"] = plans

    roles = [(_id("role", name), name, desc) for name, desc in PLATFORM_ROLES]
    rows["roles"] = roles

    orgs = []
    for t in TENANTS:
        orgs.append((
            _id("org", t["slug"]), t["name"], t["slug"], "active",
            "growth" if t["slug"] == "genworx" else "enterprise", "active",
            50 if t["slug"] == "genworx" else 500,
            f"billing@{t['domain']}",
            Jsonb({"industry": t["industry"], "country": t["country"]}),
        ))
    rows["organisations"] = orgs

    users, user_roles = [], []
    for t_i, t in enumerate(TENANTS):
        for u_i in range(6):
            first, last = FIRST[(t_i * 6 + u_i) % len(FIRST)], LAST[(t_i * 7 + u_i) % len(LAST)]
            uid = _id("user", t["slug"], u_i)
            email = f"{first.lower()}.{last.lower()}@{t['domain']}"
            role = PLATFORM_ROLES[0][0] if u_i == 0 else PLATFORM_ROLES[1 + (u_i % 3)][0]
            users.append((
                uid, email, f"{first} {last}", _id("org", t["slug"]), "local",
                "active" if u_i < 5 else "pending",
                START + timedelta(days=random.randint(0, 120)),
            ))
            user_roles.append((_id("urole", t["slug"], u_i), uid, _id("role", role)))
    rows["users"] = users
    rows["user_roles"] = user_roles

    # ---------------- tenant profiles ----------------
    rows["tenant_profiles"] = [
        (
            _id("org", t["slug"]), t["name"], t["domain"], f"https://www.{t['domain']}",
            f"GST{random.randint(10**9, 10**10 - 1)}", f"TIN{random.randint(10**7, 10**8 - 1)}",
            t["legal"], t["timezone"], "en", t["country"], t["state"], t["currency"],
            t["industry"],
        )
        for t in TENANTS
    ]

    # ---------------- currency rates ----------------
    rates = []
    for weeks_back in range(10):
        rate_date = (TODAY - timedelta(weeks=weeks_back)).date()
        for cur in CURRENCIES:
            drift = 1.0 if cur == "USD" else random.uniform(0.985, 1.015)
            rates.append((cur, round(USD_RATES[cur] * drift, 6), "openexchangerates", rate_date))
    rows["currency_rates"] = rates

    # ---------------- products ----------------
    products = [
        (i, code, desc) for i, (code, desc) in enumerate(PRODUCTS, start=1)
    ]
    rows["products"] = products
    unit_price = {i: _money(45, 4800) for i, _, _ in products}

    # ---------------- vendors & customers ----------------
    vendors, customers = [], []
    vendor_ids: dict[str, list] = {t["slug"]: [] for t in TENANTS}
    customer_ids: dict[str, list] = {t["slug"]: [] for t in TENANTS}

    for t in TENANTS:
        tid = _id("org", t["slug"])
        city = "Hyderabad" if t["slug"] == "genworx" else "San Jose"
        for v_i, name in enumerate(VENDOR_NAMES):
            vid = _id("vendor", t["slug"], v_i)
            vendor_ids[t["slug"]].append(vid)
            slug = name.lower().replace(" ", "")
            vendors.append((
                vid, tid, name, name, f"+1-555-{2000 + v_i:04d}",
                f"ap@{slug}.com", f"TIN-{t['slug'][:3].upper()}-V{v_i:03d}",
                _money(0, 85000), True, v_i % 5 != 0,
                Jsonb(_addr(city, t["country"])),
                f"Remit to {name}, account {random.randint(10**7, 10**8 - 1)}",
            ))
        for c_i, name in enumerate(CUSTOMER_NAMES):
            cid = _id("customer", t["slug"], c_i)
            customer_ids[t["slug"]].append(cid)
            slug = name.lower().replace(" ", "")
            customers.append((
                cid, tid, name, name, name, f"+1-555-{3000 + c_i:04d}",
                f"billing@{slug}.com", f"TIN-{t['slug'][:3].upper()}-C{c_i:03d}",
                _money(0, 120000), c_i % 11 != 0, c_i % 4 != 0,
                Jsonb(_addr(city, t["country"])), Jsonb(_addr(city, t["country"])),
            ))
    rows["vendors"] = vendors
    rows["customers"] = customers

    # ---------------- invoices & line items ----------------
    invoices, inv_lines = [], []
    payments, reminder_states, reminders = [], [], []
    invoice_meta: list[dict] = []
    seq = 0

    for t in TENANTS:
        tid = _id("org", t["slug"])
        for n in range(70):
            seq += 1
            is_ap = n % 2 == 0          # payable: we owe a vendor
            iid = _id("invoice", t["slug"], n)
            vendor = random.choice(vendor_ids[t["slug"]])
            customer = random.choice(customer_ids[t["slug"]])
            inv_date = (START + timedelta(days=random.randint(0, 630))).date()
            due = inv_date + timedelta(days=random.choice([15, 30, 45, 60]))
            currency = t["currency"] if random.random() < 0.65 else random.choice(CURRENCIES)
            status = random.choice(AP_STATUSES if is_ap else AR_STATUSES)

            # Lines first, so the header's arithmetic is derived rather than invented.
            line_count = random.randint(1, 4)
            subtotal, total_qty = 0.0, 0.0
            chosen = random.sample(range(1, len(PRODUCTS) + 1), line_count)
            for ln, pid in enumerate(chosen, start=1):
                qty = float(random.randint(1, 40))
                price = round(unit_price[pid] * random.uniform(0.92, 1.08), 2)
                line_total = round(qty * price, 2)
                subtotal += line_total
                total_qty += qty
                inv_lines.append((
                    _id("invline", t["slug"], n, ln), iid, tid, ln, "SERVICE" if pid <= 10 else "GOODS",
                    pid, PRODUCTS[pid - 1][1], PRODUCTS[pid - 1][0], qty, price, line_total,
                    False, customer, CUSTOMER_NAMES[customer_ids[t["slug"]].index(customer)],
                ))

            subtotal = round(subtotal, 2)
            tax_rate = random.choice([0.0, 0.05, 0.09, 0.18])
            tax_amount = round(subtotal * tax_rate, 2)
            shipping = round(random.choice([0, 0, 0, 125, 480]), 2)
            total = round(subtotal + tax_amount + shipping, 2)

            paid_so_far = 0.0
            if status in ("paid",):
                paid_so_far = total
            elif status in ("partially_paid", "ready_to_pay"):
                paid_so_far = round(total * random.uniform(0.2, 0.75), 2)
            balance_due = round(total - paid_so_far, 2)

            fx = USD_RATES[currency]
            invoices.append((
                iid, tid, f"PO-{t['slug'][:3].upper()}-{1000 + n}", vendor, customer,
                "email" if n % 3 else "connector", True,
                "AP" if is_ap else "AR", status,
                f"{'BILL' if is_ap else 'INV'}-{t['slug'][:3].upper()}-{2025000 + seq}",
                inv_date, due, currency, round(1 / fx, 6) if fx else 1,
                total_qty, subtotal, False, shipping,
                Jsonb([{"name": "GST" if currency == "INR" else "VAT",
                        "rate": tax_rate * 100, "amount": tax_amount}]),
                total, balance_due,
                f"ap@{VENDOR_NAMES[vendor_ids[t['slug']].index(vendor)].lower().replace(' ', '')}.com",
                f"billing@{CUSTOMER_NAMES[customer_ids[t['slug']].index(customer)].lower().replace(' ', '')}.com",
                f"{'Vendor bill' if is_ap else 'Customer invoice'} for {line_count} line item(s)",
                round(total * fx, 2), inv_date, "openexchangerates",
            ))
            invoice_meta.append({
                "id": iid, "tenant": tid, "slug": t["slug"], "n": n, "is_ap": is_ap,
                "total": total, "paid": paid_so_far, "balance": balance_due,
                "currency": currency, "status": status, "due": due,
                "customer": customer, "vendor": vendor, "date": inv_date,
                "number": f"{'BILL' if is_ap else 'INV'}-{t['slug'][:3].upper()}-{2025000 + seq}",
            })

    rows["invoices"] = invoices
    rows["invoice_line_items"] = inv_lines

    # ---------------- payments ----------------
    for m in invoice_meta:
        if m["paid"] <= 0:
            continue
        instalments = 1 if m["paid"] == m["total"] and random.random() < 0.6 else random.randint(1, 3)
        remaining, balance = m["paid"], m["total"]
        # A natural-key index covers (tenant, invoice, amount, date, action), so
        # instalments are walked forward in time rather than scattered: an even
        # split of a round total would otherwise collide with itself.
        paid_on = m["due"] - timedelta(days=random.randint(10, 30))
        for k in range(instalments):
            amount = round(remaining / (instalments - k), 2) if k < instalments - 1 else round(remaining, 2)
            remaining = round(remaining - amount, 2)
            balance = round(balance - amount, 2)
            payments.append((
                _id("payment", m["slug"], m["n"], k), m["tenant"], m["id"], amount,
                paid_on, "payment" if k == 0 else "part_payment", m["currency"], balance,
                f"accounts@{TENANTS[0 if m['slug'] == 'genworx' else 1]['domain']}",
            ))
            paid_on = paid_on + timedelta(days=random.randint(7, 25))
    rows["payment_history"] = payments

    # ---------------- reminder state & reminders ----------------
    for m in invoice_meta:
        # Receivables are what get chased, and only the ones still owing.
        if m["is_ap"] or m["balance"] <= 0:
            continue
        if random.random() > 0.55:
            continue
        overdue_days = (TODAY.date() - m["due"]).days
        state = "PAID" if m["balance"] == 0 else random.choice(REMINDER_STATE_STATUSES[:3])
        level = min(3, max(0, overdue_days // 15))
        reminder_states.append((
            m["id"], m["tenant"], state,
            TODAY + timedelta(days=random.randint(1, 20)), level,
            f"billing@{CUSTOMER_NAMES[0].lower().replace(' ', '')}.com",
            CUSTOMER_NAMES[random.randrange(len(CUSTOMER_NAMES))],
            m["number"], m["total"], m["paid"], m["balance"],
            datetime.combine(m["due"], datetime.min.time(), tzinfo=timezone.utc), m["currency"],
        ))
        for lvl in range(1, level + 1):
            reminders.append((
                _id("reminder", m["slug"], m["n"], lvl), m["id"], m["tenant"], lvl,
                f"Level {lvl} reminder",
                ["friendly", "firm", "final"][min(lvl - 1, 2)],
                f"ar_reminder_l{lvl}",
                f"billing@{CUSTOMER_NAMES[random.randrange(len(CUSTOMER_NAMES))].lower().replace(' ', '')}.com",
                random.choice(CUSTOMER_NAMES),
                random.choices(REMINDER_STATUSES, weights=[8, 1, 1])[0],
                datetime.combine(m["due"], datetime.min.time(), tzinfo=timezone.utc)
                + timedelta(days=15 * lvl),
                lvl,
            ))
    rows["invoice_reminder_state"] = reminder_states
    rows["reminders"] = reminders

    # ---------------- purchase orders & line items ----------------
    pos, po_lines = [], []
    payable = [m for m in invoice_meta if m["is_ap"]]
    for p_i, m in enumerate(random.sample(payable, k=min(50, len(payable)))):
        po_id = _id("po", m["slug"], p_i)
        po_date = m["date"] - timedelta(days=random.randint(5, 45))
        subtotal, total_qty = 0.0, 0.0
        for ln, pid in enumerate(random.sample(range(1, len(PRODUCTS) + 1), random.randint(1, 3)), start=1):
            qty = float(random.randint(1, 40))
            price = round(unit_price[pid] * random.uniform(0.9, 1.05), 2)
            line_total = round(qty * price, 2)
            subtotal += line_total
            total_qty += qty
            po_lines.append((
                _id("poline", m["slug"], p_i, ln), po_id, m["tenant"], ln,
                "SERVICE" if pid <= 10 else "GOODS", pid,
                PRODUCTS[pid - 1][0], PRODUCTS[pid - 1][1], qty, price, line_total,
                False, m["customer"], round(qty * random.uniform(0.6, 1.0), 2),
            ))
        subtotal = round(subtotal, 2)
        total = round(subtotal * 1.09, 2)
        delivered = random.random() < 0.7
        pos.append((
            po_id, m["tenant"], "email", m["id"], m["vendor"], m["customer"], True,
            random.choice(["open", "closed", "cancelled", "partially_received"]),
            random.choice(["billed", "unbilled", "partially_billed"]),
            "received" if delivered else "pending",
            f"PO-{m['slug'][:3].upper()}-{1000 + p_i}", po_date,
            po_date + timedelta(days=random.randint(7, 40)),
            po_date + timedelta(days=random.randint(8, 50)) if delivered else None,
            m["currency"], total_qty, subtotal, False, total,
            VENDOR_NAMES[random.randrange(len(VENDOR_NAMES))],
            f"Purchase order raised ahead of {m['number']}",
        ))
    rows["purchase_orders"] = pos
    rows["po_line_items"] = po_lines

    # ---------------- contracts ----------------
    contracts = []
    for c_i, m in enumerate(random.sample(invoice_meta, k=16)):
        signed = m["date"] - timedelta(days=random.randint(30, 400))
        term = random.choice([12, 24, 36])
        annual = _money(25000, 480000)
        contracts.append((
            _id("contract", m["slug"], c_i),
            f"RFP-{2025}-{100 + c_i}",
            f"Master Services Agreement {c_i + 1}",
            f"CTR-{m['slug'][:3].upper()}-{500 + c_i}",
            signed, signed + timedelta(days=7), term,
            signed + timedelta(days=7 + 30 * term),
            round(annual * term / 12, 2), round(annual * 1.06, 2), annual,
            f"Fees escalate {random.choice([3, 4, 5])}% at each renewal.",
            f"Net {random.choice([30, 45, 60])} from invoice date.",
            m["tenant"], m["vendor"], m["customer"], m["id"],
            random.choice(["MSA", "SOW", "NDA", "SUBSCRIPTION"]),
        ))
    rows["contracts"] = contracts

    # ---------------- discrepancies ----------------
    discrepancies, resolutions = [], []
    for d_i, m in enumerate(random.sample(invoice_meta, k=40)):
        match_id = _id("match", m["slug"], d_i)
        has = random.random() < 0.72
        dtype = random.choice(DISCREPANCY_TYPES)
        delta = _money(50, 9000)
        discrepancies.append((
            _id("discrepancy", m["slug"], d_i), m["tenant"], match_id,
            random.choice(["invoice_po", "invoice_contract", "invoice_only"]), has,
            round(random.uniform(0.41, 0.99), 2),
            Jsonb({"header": round(random.uniform(0.5, 1.0), 2),
                   "line_items": round(random.uniform(0.3, 1.0), 2),
                   "totals": round(random.uniform(0.4, 1.0), 2)}),
            Jsonb([{"field": dtype, "expected": round(m["total"] - delta, 2),
                    "found": m["total"], "delta": delta}] if has else []),
            Jsonb({"invoice_number_present": True, "due_date_present": True,
                   "vendor_resolved": True}),
            random.random() < 0.25,
            Jsonb({"purchase_order": bool(m["is_ap"]), "contract": random.random() < 0.4}),
            Jsonb({"type": dtype, "severity": random.choice(["low", "medium", "high"]),
                   "amount_delta": delta} if has else {"type": None}),
            Jsonb({"engine": "match-v2", "duration_ms": random.randint(120, 4200)}),
            random.choice(["PENDING", "TRIGGERED", "SKIPPED"]),
        ))
        if has and random.random() < 0.6:
            status = random.choice(RESOLUTION_STATUSES)
            trigger = TODAY - timedelta(days=random.randint(1, 180))
            resolutions.append((
                _id("resolution", m["slug"], d_i), match_id,
                "AP" if m["is_ap"] else "AR", dtype,
                f"ap@{VENDOR_NAMES[random.randrange(len(VENDOR_NAMES))].lower().replace(' ', '')}.com",
                f"finance.admin@{TENANTS[0 if m['slug'] == 'genworx' else 1]['domain']}",
                status, random.randint(0, 3), trigger,
                trigger + timedelta(days=random.randint(1, 20)) if status == "Closed" else None,
                f"{dtype.replace('_', ' ').capitalize()} of {delta} raised with the vendor."
                if status != "Awaiting_Review" else None,
            ))
    rows["discrepancies_details"] = discrepancies
    rows["discrepancy_resolution"] = resolutions

    return rows


# Reverse foreign-key order. Only tables this seeder fills appear here: a table
# it does not populate is a table it must not empty, because something else may
# have put rows there on purpose. Cleared row by row rather than truncated --
# TRUNCATE ... CASCADE would silently reach past this list into the document,
# audit and connector tables that reference these rows, and a re-seed is not
# allowed to destroy anything outside its own scope. A DELETE that something
# else still depends on fails loudly instead, which is the wanted outcome.
CLEAR_ORDER = [
    "finopsiq.discrepancy_resolution",
    "finopsiq.discrepancies_details",
    "finopsiq.reminders",
    "finopsiq.invoice_reminder_state",
    "finopsiq.payment_history",
    "finopsiq.contracts",
    "finopsiq.po_line_items",
    "finopsiq.purchase_orders",
    "finopsiq.invoice_line_items",
    "finopsiq.invoices",
    "finopsiq.customers",
    "finopsiq.vendors",
    "finopsiq.products",
    "finopsiq.currency_rates",
    "finopsiq.tenant_profiles",
    "auth.user_roles",
    "auth.users",
    "auth.organisations",
    "auth.roles",
    "auth.plans",
]

INSERTS: list[tuple[str, str, str]] = [
    ("plans", "auth.plans", "(id, name, seat_limit, products, is_platform_default, metadata)"),
    ("roles", "auth.roles", "(id, name, description)"),
    ("organisations", "auth.organisations",
     "(id, name, slug, status, plan, plan_status, seat_limit, billing_email, metadata)"),
    ("users", "auth.users",
     "(id, email, full_name, organisation_id, auth_provider, status, created_at)"),
    ("user_roles", "auth.user_roles", "(id, user_id, role_id)"),
    ("tenant_profiles", "finopsiq.tenant_profiles",
     "(tenant_id, organization_name, domain, company_website, gst_number, tin_number,"
     " legal_org_name, timezone, language, country, state, currency, industry)"),
    ("currency_rates", "finopsiq.currency_rates", "(currency, usd_rate, source, rate_date)"),
    ("products", "finopsiq.products", "(product_id, product_code, default_description)"),
    ("vendors", "finopsiq.vendors",
     "(vendor_id, tenant_id, company_name, display_name, phone, email, tin, balance,"
     " is_active, is_taxable, vendor_address, remittance_address)"),
    ("customers", "finopsiq.customers",
     "(customer_id, tenant_id, company_name, display_name, name, phone, email, tin, balance,"
     " is_active, is_taxable, billing_address, shipping_address)"),
    ("invoices", "finopsiq.invoices",
     "(invoice_id, tenant_id, purchase_order_number, vendor_id, customer_id, source, show_status,"
     " invoice_type, status, invoice_number, invoice_date, due_date, currency, exchange_rate,"
     " total_quantity, subtotal, apply_tax_after_discount, shipping_charges, taxes, total_amount,"
     " balance_due, vendor_email, customer_email, description, amount_usd, fx_rate_date, fx_source)"),
    ("invoice_line_items", "finopsiq.invoice_line_items",
     "(invoice_line_item_id, invoice_id, tenant_id, line_number, line_item_type, product_id,"
     " description, item_name, quantity, unit_price, line_total, apply_tax_after_discount,"
     " customer_id, customer_name)"),
    ("purchase_orders", "finopsiq.purchase_orders",
     "(po_id, tenant_id, source, invoice_id, vendor_id, customer_id, show_status, order_status,"
     " billed_status, received_status, po_number, po_date, expected_delivery_date, delivery_date,"
     " currency, total_quantity, subtotal, apply_tax_after_discount, total_amount, vendor_name, notes)"),
    ("po_line_items", "finopsiq.po_line_items",
     "(po_line_item_id, po_id, tenant_id, line_number, line_item_type, product_id, item_name,"
     " description, quantity, unit_price, line_total, apply_tax_after_discount, customer_id,"
     " processed_quantity)"),
    ("contracts", "finopsiq.contracts",
     "(contract_id, request_for_proposal_number, official_document_title, assigned_contract_number,"
     " agreement_signed_date, start_date, term_in_months, expiry_date, total_contract_amount_value,"
     " renewal_contract_amount_value, initial_annual_fee_value, price_escalation_terms_description,"
     " payment_terms_description, tenant_id, vendor_id, customer_id, invoice_id, contract_type)"),
    ("payment_history", "finopsiq.payment_history",
     "(payment_id, tenant_id, invoice_id, amount, payment_date, action, currency, balance_after,"
     " recorded_by)"),
    ("invoice_reminder_state", "finopsiq.invoice_reminder_state",
     "(invoice_id, tenant_id, status, next_reminder_at, next_level, customer_email, customer_name,"
     " invoice_number, total_amount, amount_paid, amount_due, due_date, currency)"),
    ("reminders", "finopsiq.reminders",
     "(reminder_id, invoice_id, tenant_id, level, reminder_label, reminder_tone, template_used,"
     " recipient_email, recipient_name, status, sent_at, attempt)"),
    ("discrepancies_details", "finopsiq.discrepancies_details",
     "(discrepancies_id, tenant_id, match_id, match_type, has_discrepancy, confidence_score,"
     " confidence_breakdown, comparison_discrepancies_details, basic_validation_details,"
     " missing_relevant_documents, matched_details, discrepancy_summary, processing_metadata,"
     " dra_trigger_status)"),
    ("discrepancy_resolution", "finopsiq.discrepancy_resolution",
     "(resolution_id, match_id, invoice_type, discrepancy_type, vendor_email, finance_admin_id,"
     " status, retry_count, trigger_time, resolution_time, resolution_notes)"),
]


def seed_finops() -> None:
    """Fill the FinOpsIQ core tables with a coherent synthetic book of business."""
    dsn = settings.finops_dsn
    if not dsn:
        raise RuntimeError("FINOPS_DSN is not configured; nothing to seed.")

    print("Generating synthetic FinOpsIQ data...")
    data = generate()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            print("Clearing the tables this seeder owns...")
            for table in CLEAR_ORDER:
                cur.execute(f"DELETE FROM {table};")

            for key, table, columns in INSERTS:
                values = data[key]
                if not values:
                    continue
                placeholders = ", ".join(["%s"] * len(values[0]))
                print(f"Inserting {len(values):>5} rows into {table}...")
                cur.executemany(
                    f"INSERT INTO {table} {columns} VALUES ({placeholders});", values
                )

            # The products primary key is a sequence, and rows were inserted with
            # explicit ids so the line items could reference them. Left alone, the
            # next application insert would collide with row 1.
            cur.execute(
                "SELECT setval(pg_get_serial_sequence('finopsiq.products', 'product_id'),"
                " (SELECT max(product_id) FROM finopsiq.products));"
            )

    total = sum(len(data[k]) for k, _, _ in INSERTS)
    print(f"FinOpsIQ seed complete: {total} rows across {len(INSERTS)} tables.")
