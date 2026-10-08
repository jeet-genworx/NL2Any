"""
Script to populate FinOps IQ database with realistic, high-complexity synthetic data
across auth, connector_service, and finopsiq schemas.
Skips: File storage tables, document storage tables, and embedding tables.
"""

import sys
import uuid
import json
import random
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
import psycopg

DSN = "postgresql://jeetthakur@localhost:5432/finops"

def generate_synthetic_data():
    conn = psycopg.connect(DSN, autocommit=False)
    cur = conn.cursor()

    try:
        print("Connected to database. Beginning synthetic data population...")

        # ---------------------------------------------------------
        # 1. AUTH: PLANS
        # ---------------------------------------------------------
        print("\n--- Populating auth.plans ---")
        plans_data = [
            ("Starter", 5, ["invoicing", "connectors"], True, {"tier": "starter", "annual_discount": 0.1}),
            ("Growth", 20, ["invoicing", "connectors", "discrepancies", "reminders"], False, {"tier": "growth", "annual_discount": 0.15}),
            ("Business", 50, ["invoicing", "connectors", "discrepancies", "reminders", "contracts", "auditing"], False, {"tier": "business", "annual_discount": 0.2}),
            ("Enterprise", 250, ["all_modules", "custom_connectors", "ai_matching", "sla_guarantee"], False, {"tier": "enterprise", "support": "24/7"}),
            ("Enterprise Unlimited", 1000, ["all_modules", "dedicated_vpc", "custom_llm", "priority_support"], False, {"tier": "unlimited", "dedicated_account_manager": True})
        ]
        
        plan_ids = {}
        for name, seats, prods, default, meta in plans_data:
            cur.execute("""
                INSERT INTO auth.plans (id, name, seat_limit, products, is_platform_default, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (name) DO UPDATE SET seat_limit = EXCLUDED.seat_limit
                RETURNING id;
            """, (uuid.uuid4(), name, seats, json.dumps(prods), default, json.dumps(meta)))
            plan_ids[name] = cur.fetchone()[0]
        print(f"  auth.plans populated: {len(plan_ids)} plans.")

        # ---------------------------------------------------------
        # 2. AUTH: ORGANISATIONS & TENANT PROFILES
        # ---------------------------------------------------------
        print("\n--- Populating auth.organisations & finopsiq.tenant_profiles ---")
        genworx_org_id = uuid.UUID('13d3cf63-7430-49a9-af74-95f2291c5df9')
        test_org_id = uuid.UUID('e4aee2aa-8e2e-4031-9ef8-427da731df34')
        acme_org_id = uuid.UUID('a1111111-1111-4111-a111-111111111111')
        cloudscale_org_id = uuid.UUID('b2222222-2222-4222-b222-222222222222')
        fintech_org_id = uuid.UUID('c3333333-3333-4333-c333-333333333333')

        new_orgs = [
            (acme_org_id, "Acme Global Industries", "acme-global", "active", "Enterprise", 250, "billing@acmeglobal.com"),
            (cloudscale_org_id, "CloudScale AI Infrastructure", "cloudscale-ai", "active", "Business", 50, "finance@cloudscale.ai"),
            (fintech_org_id, "FinTech Apex Systems", "fintech-apex", "active", "Enterprise Unlimited", 1000, "invoices@fintechapex.com")
        ]

        for oid, oname, oslug, ostatus, oplan, oseats, obill in new_orgs:
            cur.execute("""
                INSERT INTO auth.organisations (id, name, slug, status, plan, seat_limit, billing_email, plan_status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', now() - interval '180 days', now())
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, plan = EXCLUDED.plan;
            """, (oid, oname, oslug, ostatus, oplan, oseats, obill))

        cur.execute("""
            UPDATE auth.organisations 
            SET plan = 'Enterprise Unlimited', seat_limit = 500, plan_status = 'active', billing_email = 'finance@genworx.ai'
            WHERE id = %s;
        """, (genworx_org_id,))
        cur.execute("""
            UPDATE auth.organisations 
            SET plan = 'Growth', seat_limit = 20, plan_status = 'active', billing_email = 'test-finance@genworx.ai'
            WHERE id = %s;
        """, (test_org_id,))

        tenants = [
            (genworx_org_id, "Genworx AI Systems", "genworx.ai", "https://genworx.ai", "27AABCG1234F1Z1"),
            (test_org_id, "Test Org Corp", "test-org.com", "https://test-org.com", "06AAACT0000A1Z5"),
            (acme_org_id, "Acme Global Industries", "acmeglobal.com", "https://acmeglobal.com", "29AAAAA0000A1Z5"),
            (cloudscale_org_id, "CloudScale AI Infrastructure", "cloudscale.ai", "https://cloudscale.ai", "33BBBBB1111B1Z6"),
            (fintech_org_id, "FinTech Apex Systems", "fintechapex.com", "https://fintechapex.com", "27CCCCC2222C1Z7")
        ]

        for tid, tname, tdom, tweb, tgst in tenants:
            cur.execute("""
                INSERT INTO finopsiq.tenant_profiles (
                    tenant_id, organization_name, domain, company_website, gst_number,
                    currency, timezone, max_retries, cooldown_seconds,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, 'USD', 'America/New_York', 3, 300, now() - interval '200 days', now())
                ON CONFLICT (tenant_id) DO UPDATE SET 
                    organization_name = EXCLUDED.organization_name,
                    domain = EXCLUDED.domain,
                    gst_number = EXCLUDED.gst_number;
            """, (tid, tname, tdom, tweb, tgst))

        print("  auth.organisations & finopsiq.tenant_profiles updated.")

        # ---------------------------------------------------------
        # 3. AUTH: ORG_TOKEN_VERSION & DEFAULT_IDP_CONFIG
        # ---------------------------------------------------------
        print("\n--- Populating auth.org_token_version & auth.default_idp_config ---")
        for tid, _, _, _, _ in tenants:
            cur.execute("""
                INSERT INTO auth.org_token_version (org_id, version, updated_at)
                VALUES (%s, 1, now())
                ON CONFLICT (org_id) DO NOTHING;
            """, (tid,))

        idp_configs = [
            ("google", "accounts.google.com", "google-oauth-client-id-prod-01", "https://auth.finopsiq.com/callback/google", True),
            ("azure-ad", "login.microsoftonline.com", "azure-client-id-ent-02", "https://auth.finopsiq.com/callback/azure", True),
            ("okta", "okta.corp.finopsiq.com", "okta-client-id-sso-03", "https://auth.finopsiq.com/callback/okta", True)
        ]
        for prov, tenant_str, cid, redir, en in idp_configs:
            cur.execute("""
                INSERT INTO auth.default_idp_config (id, provider, tenant_id, client_id, redirect_uri, is_enabled, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (provider) DO UPDATE SET client_id = EXCLUDED.client_id;
            """, (uuid.uuid4(), prov, tenant_str, cid, redir, en))

        # ---------------------------------------------------------
        # 4. AUTH: ORG_GROUPS & USERS & GROUP_MEMBERS
        # ---------------------------------------------------------
        print("\n--- Populating auth.org_groups, auth.users & auth.org_group_members ---")
        group_templates = [
            ("FinOps & Cost Governance", "Team managing cloud cost allocation, unit economics and budgets"),
            ("IT Procurement & Vendor Management", "Contract negotiations, MSA management and vendor relations"),
            ("Cloud Architecture & SRE", "Infrastructure architects, Kubernetes and cloud resource managers"),
            ("Accounts Payable & Finance", "Invoice auditing, discrepancy resolutions and payment processing"),
            ("Executive Leadership", "CFO, VP Engineering, and leadership with full executive oversight")
        ]

        tenant_group_map = {}
        for tid, _, _, _, _ in tenants:
            for gname, gdesc in group_templates:
                cur.execute("""
                    INSERT INTO auth.org_groups (id, organisation_id, name, description, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, '{}'::jsonb, now(), now())
                    ON CONFLICT (organisation_id, name) DO UPDATE SET description = EXCLUDED.description
                    RETURNING id;
                """, (uuid.uuid4(), tid, gname, gdesc))
                gid = cur.fetchone()[0]
                tenant_group_map.setdefault(tid, {})[gname] = gid

        # Add realistic users
        user_specs = [
            ("alex.chen@genworx.ai", "Alex Chen", genworx_org_id, "FinOps & Cost Governance"),
            ("sarah.miller@genworx.ai", "Sarah Miller", genworx_org_id, "Accounts Payable & Finance"),
            ("david.ross@genworx.ai", "David Ross", genworx_org_id, "IT Procurement & Vendor Management"),
            ("elena.rostova@genworx.ai", "Elena Rostova", genworx_org_id, "Cloud Architecture & SRE"),
            ("marcus.vance@genworx.ai", "Marcus Vance", genworx_org_id, "Executive Leadership"),
            ("priya.sharma@genworx.ai", "Priya Sharma", genworx_org_id, "Accounts Payable & Finance"),
            ("robert.kim@acmeglobal.com", "Robert Kim", acme_org_id, "FinOps & Cost Governance"),
            ("amanda.hughes@acmeglobal.com", "Amanda Hughes", acme_org_id, "IT Procurement & Vendor Management"),
            ("vikram.patel@cloudscale.ai", "Vikram Patel", cloudscale_org_id, "Cloud Architecture & SRE"),
            ("jessica.walsh@fintechapex.com", "Jessica Walsh", fintech_org_id, "Accounts Payable & Finance")
        ]

        user_id_map = {}
        for uemail, ufullname, uorg, ugrp in user_specs:
            cur.execute("""
                INSERT INTO auth.users (id, organisation_id, email, full_name, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'active', now() - interval '120 days', now())
                ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name, organisation_id = EXCLUDED.organisation_id
                RETURNING id;
            """, (uuid.uuid4(), uorg, uemail, ufullname))
            uid = cur.fetchone()[0]
            user_id_map[uemail] = (uid, uorg, ugrp)

            # Link group member
            gid = tenant_group_map[uorg][ugrp]
            cur.execute("""
                INSERT INTO auth.org_group_members (id, group_id, user_id, created_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (group_id, user_id) DO NOTHING;
            """, (uuid.uuid4(), gid, uid))

        # Assign user roles
        cur.execute("SELECT id, name FROM auth.roles;")
        roles_dict = {r[1]: r[0] for r in cur.fetchall()}
        admin_role_id = roles_dict.get('admin') or list(roles_dict.values())[0]
        member_role_id = roles_dict.get('member') or admin_role_id

        for uemail, (uid, uorg, ugrp) in user_id_map.items():
            role_to_assign = admin_role_id if "Executive" in ugrp or "FinOps" in ugrp else member_role_id
            cur.execute("""
                INSERT INTO auth.user_roles (id, user_id, role_id, created_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (user_id, role_id) DO NOTHING;
            """, (uuid.uuid4(), uid, role_to_assign))

        print("  auth.org_groups, auth.users, and group members populated.")

        # ---------------------------------------------------------
        # 5. CONNECTOR SERVICE: CONNECTORS & SYNC JOBS
        # ---------------------------------------------------------
        print("\n--- Populating connector_service ---")
        extra_connectors = [
            (2, "Xero", "xero", "client_xero_id_01", "sec_xero_01", "accounting.transactions", True),
            (5, "NetSuite ERP", "netsuite", "client_ns_id_02", "sec_ns_02", "restlets.accounting", True),
            (6, "SAP S/4HANA", "sap", "client_sap_id_03", "sec_sap_03", "api.finance.invoices", True)
        ]
        for cid, cname, ckey, clid, csec, cscope, cactive in extra_connectors:
            cur.execute("""
                INSERT INTO connector_service.connectors (id, name, key, client_id, client_secret, scope, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, key = EXCLUDED.key;
            """, (cid, cname, ckey, clid, csec, cscope, cactive))

        for tid, _, _, _, _ in tenants:
            for cid in [1, 4, 5]:
                cur.execute("""
                    INSERT INTO connector_service.tenant_connector_auth (
                        tenant_id, connector_id, external_realm_id, access_token, refresh_token,
                        token_type, is_active, access_expires_at, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, 'enc_token_fake_abc123', 'enc_refresh_fake_xyz789', 'Bearer', true, now() + interval '30 days', now(), now())
                    ON CONFLICT DO NOTHING;
                """, (str(tid), cid, random.randint(100000, 999999)))

        sync_entities = ["INVOICE", "PURCHASE_ORDER", "VENDOR", "CUSTOMER", "PAYMENT"]
        sync_statuses = ["COMPLETED", "COMPLETED", "COMPLETED", "FAILED", "QUEUED"]
        for tid, _, _, _, _ in tenants[:3]:
            for cid in [1, 4]:
                for ent in sync_entities:
                    st = random.choice(sync_statuses)
                    tot = random.randint(20, 150)
                    proc = tot if st == "COMPLETED" else random.randint(0, tot - 5)
                    err = "Socket timeout during batch fetch from ERP API endpoint" if st == "FAILED" else None
                    cur.execute("""
                        INSERT INTO connector_service.sync_jobs (
                            tenant_id, connector_id, external_realm_id, user_email, entity_type,
                            sync_type, status, total_records, processed_records, new_records,
                            updated_records, failed_records, error_message, triggered_by,
                            created_at, updated_at
                        )
                        VALUES (%s, %s, %s, 'sync-daemon@finopsiq.com', %s, 'SCHEDULED', %s, %s, %s, %s, %s, %s, %s, 'CRON_SCHEDULER', now() - interval '2 days', now() - interval '2 days');
                    """, (str(tid), cid, str(random.randint(100000, 999999)), ent, st, tot, proc, int(proc * 0.7), int(proc * 0.3), (tot - proc) if st == "FAILED" else 0, err))

        print("  connector_service tables populated.")

        # ---------------------------------------------------------
        # 6. FINOPSIQ: PRODUCTS (CATALOG OF CLOUD, SAAS & SERVICES)
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.products ---")
        product_catalog = [
            ("AWS-EC2-C5-XLARGE", "AWS Compute Optimized EC2 Instance c5.xlarge (4 vCPU, 8 GiB RAM)"),
            ("AWS-EC2-M6I-4XLARGE", "AWS General Purpose EC2 Instance m6i.4xlarge (16 vCPU, 64 GiB RAM)"),
            ("AWS-EC2-R6G-GRAVITON", "AWS Memory Optimized Graviton2 r6g.2xlarge"),
            ("AWS-S3-STANDARD-STORAGE", "AWS S3 Standard Storage (per TB-month)"),
            ("AWS-RDS-AURORA-POSTGRES", "AWS Aurora Serverless v2 PostgreSQL ACU/hour"),
            ("AWS-LAMBDA-INVOCATIONS", "AWS Lambda Compute Duration and Invocations"),
            ("AWS-CLOUDFRONT-CDN", "AWS CloudFront Global CDN Data Transfer Out (GB)"),
            ("AWS-SAVINGS-PLAN-COMPUTE", "AWS 3-Year All-Upfront Compute Savings Plan Commitment"),
            ("GCP-COMPUTE-N2-STANDARD-8", "Google Cloud Compute Engine N2 Standard 8-core Instance"),
            ("GCP-BIGQUERY-ANALYSIS", "Google BigQuery On-Demand Query Analysis (per TB scanned)"),
            ("GCP-GKE-MANAGED-CLUSTER", "Google Kubernetes Engine (GKE) Enterprise Control Plane"),
            ("AZURE-VM-STANDARD-D8S-V5", "Microsoft Azure Virtual Machines D8s v5 (8 vCPUs, 32 GiB RAM)"),
            ("AZURE-BLOB-HOT-TIER", "Azure Blob Storage Hot Tier Capacity (TB/Month)"),
            ("AZURE-COSMOS-DB-RU", "Azure Cosmos DB Request Units (RU/s standard provisioned)"),
            ("SNOWFLAKE-COMPUTE-CREDIT", "Snowflake Enterprise Data Cloud Compute Credit (Warehouse Usage)"),
            ("DATADOG-APM-PRO-HOST", "Datadog APM & Distributed Tracing Pro Host License (Monthly)"),
            ("DATADOG-LOGS-INGESTION", "Datadog Log Management Ingestion & 15-Day Retention (Million Events)"),
            ("MONGODB-ATLAS-M30-CLUSTER", "MongoDB Atlas Managed M30 Dedicated Cluster (Dual Region High Availability)"),
            ("OPENAI-GPT4O-API-TOKENS", "OpenAI GPT-4o Enterprise API Ingestion & Completion Tokens (1M Tokens)"),
            ("ANTHROPIC-CLAUDE-35-SONNET", "Anthropic Claude 3.5 Sonnet Enterprise API Ingestion & Output Tokens"),
            ("GITHUB-ENTERPRISE-CLOUD", "GitHub Enterprise Cloud Developer Seat License (Annual Subscription)"),
            ("ATLASSIAN-JIRA-ENTERPRISE", "Atlassian Jira Software Cloud Enterprise Tier (Per Active User)"),
            ("ATLASSIAN-CONFLUENCE-ENT", "Atlassian Confluence Cloud Knowledge Base Seat License"),
            ("SLACK-ENTERPRISE-GRID", "Slack Enterprise Grid Collaboration Suite (Per Seat License)"),
            ("SALESFORCE-REVENUE-CLOUD", "Salesforce Revenue Cloud & CPQ Enterprise Platform License"),
            ("CLOUDFLARE-MAGIC-WAN", "Cloudflare Magic WAN Enterprise Edge Network Routing & DDoS"),
            ("DOCKER-BUSINESS-SEAT", "Docker Desktop & Business Registry Annual User License"),
            ("TWILIO-SENDGRID-PRO", "Twilio SendGrid Pro Marketing & Transactional Email Tier"),
            ("FINOPS-AUDIT-CONSULTING", "FinOps Foundation Certified Cloud Cost Optimization Consulting (Hourly)"),
            ("CLOUD-MIGRATION-SERVICES", "Cloud Architecture & Modernization Professional Implementation Services"),
            ("SRE-24X7-MANAGED-RETAINER", "24/7 Managed Site Reliability Engineering Retainer Support"),
            ("DELL-POWEREDGE-R750-SERVER", "Dell PowerEdge R750 Rack Server Dual Xeon Gold (Hardware Asset)"),
            ("CISCO-NEXUS-9300-SWITCH", "Cisco Nexus 9300 Series 48-Port 100G Data Center Switch"),
            ("APPLE-MACBOOK-PRO-M3-MAX", "Apple MacBook Pro 16-inch M3 Max 64GB Unified Memory Engineering Unit"),
            ("ZOOM-ENTERPRISE-BUNDLE", "Zoom Enterprise Video Conferencing & Phone System License")
        ]

        product_id_map = {}
        for pcode, pdesc in product_catalog:
            cur.execute("""
                INSERT INTO finopsiq.products (product_code, default_description, created_at, updated_at)
                VALUES (%s, %s, now(), now())
                ON CONFLICT (product_code) DO UPDATE SET default_description = EXCLUDED.default_description
                RETURNING product_id;
            """, (pcode, pdesc))
            pid = cur.fetchone()[0]
            product_id_map[pcode] = pid
        print(f"  finopsiq.products populated: {len(product_id_map)} products.")

        # ---------------------------------------------------------
        # 7. FINOPSIQ: VENDORS
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.vendors ---")
        vendor_definitions = [
            ("Amazon Web Services, Inc.", "AWS", "41-1994444", "aws-invoicing@amazon.com", "+1-206-266-1000", "Seattle, WA, USA", "USD"),
            ("Microsoft Corporation", "Microsoft Azure & 365", "91-1144442", "msft-billing@microsoft.com", "+1-425-882-8080", "Redmond, WA, USA", "USD"),
            ("Google LLC", "Google Cloud Platform", "77-0493581", "cloud-billing@google.com", "+1-650-253-0000", "Mountain View, CA, USA", "USD"),
            ("Snowflake Inc.", "Snowflake Data Cloud", "47-1939281", "ar@snowflake.com", "+1-844-766-9355", "Bozeman, MT, USA", "USD"),
            ("Datadog, Inc.", "Datadog Monitoring", "27-2483921", "billing@datadoghq.com", "+1-866-328-2364", "New York, NY, USA", "USD"),
            ("MongoDB, Inc.", "MongoDB Atlas", "26-1483920", "billing@mongodb.com", "+1-646-727-4092", "New York, NY, USA", "USD"),
            ("OpenAI OpCo, LLC", "OpenAI Platform", "82-3849182", "invoices@openai.com", "+1-415-555-0199", "San Francisco, CA, USA", "USD"),
            ("Anthropic PBC", "Anthropic AI", "87-1928374", "billing@anthropic.com", "+1-415-555-0188", "San Francisco, CA, USA", "USD"),
            ("Atlassian Pty Ltd", "Atlassian Cloud", "98-0482910", "accounts-receivable@atlassian.com", "+61-2-9262-0777", "Sydney, Australia", "USD"),
            ("GitHub, Inc.", "GitHub Enterprise", "26-2582918", "billing@github.com", "+1-877-448-4820", "San Francisco, CA, USA", "USD"),
            ("Salesforce, Inc.", "Salesforce CRM", "94-3352917", "ar@salesforce.com", "+1-800-667-6389", "San Francisco, CA, USA", "USD"),
            ("Slack Technologies, LLC", "Slack Enterprise", "46-2481928", "billing@slack.com", "+1-415-555-0144", "San Francisco, CA, USA", "USD"),
            ("Cloudflare, Inc.", "Cloudflare Infrastructure", "27-0105345", "billing@cloudflare.com", "+1-888-993-5273", "San Francisco, CA, USA", "USD"),
            ("Cisco Systems, Inc.", "Cisco Enterprise Hardware", "77-0059951", "invoices@cisco.com", "+1-408-526-4000", "San Jose, CA, USA", "USD"),
            ("Dell Technologies Inc.", "Dell Hardware & Servers", "04-2616805", "accounts@dell.com", "+1-800-456-3355", "Round Rock, TX, USA", "USD"),
            ("Twilio Inc.", "Twilio & SendGrid", "26-2578091", "ar@twilio.com", "+1-877-889-4546", "San Francisco, CA, USA", "USD"),
            ("Stripe, Inc.", "Stripe Payments", "27-0192834", "ar@stripe.com", "+1-888-926-2289", "South San Francisco, CA, USA", "USD"),
            ("Docker, Inc.", "Docker Business", "46-2391028", "billing@docker.com", "+1-415-555-0177", "Palo Alto, CA, USA", "USD"),
            ("Zoom Video Communications", "Zoom Video", "61-1648780", "billing@zoom.us", "+1-888-799-9666", "San Jose, CA, USA", "USD"),
            ("ThoughtWorks Technologies", "ThoughtWorks Consulting", "36-3918274", "invoicing@thoughtworks.com", "+1-312-373-1000", "Chicago, IL, USA", "USD"),
            ("Infosys Limited", "Infosys Cloud Ops", "98-1928374", "billing@infosys.com", "+91-80-2852-0261", "Bengaluru, Karnataka, India", "INR"),
            ("Tata Consultancy Services", "TCS Enterprise Systems", "98-2938475", "invoicing@tcs.com", "+91-22-6778-9999", "Mumbai, Maharashtra, India", "INR"),
            ("OVHcloud SAS", "OVH European Cloud", "FR-482910293", "billing@ovhcloud.com", "+33-9-72-10-10-07", "Roubaix, France", "EUR"),
            ("Hetzner Online GmbH", "Hetzner Dedicated Servers", "DE-182938471", "accounting@hetzner.com", "+49-9831-505-0", "Gunzenhausen, Germany", "EUR")
        ]

        tenant_vendor_map = {}
        for tid, _, _, _, _ in tenants:
            tenant_vendor_map[tid] = []
            for vname, vdisp, vtin, vemail, vphone, vaddr, vcurr in vendor_definitions:
                tenant_tin = f"{vtin}-{str(tid)[:4]}"
                cur.execute("""
                    INSERT INTO finopsiq.vendors (
                        vendor_id, tenant_id, company_name, display_name, tin, email, phone,
                        is_active, is_taxable, balance, vendor_address, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, true, true, %s, %s, now() - interval '300 days', now())
                    ON CONFLICT (tenant_id, tin) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        display_name = EXCLUDED.display_name,
                        email = EXCLUDED.email
                    RETURNING vendor_id;
                """, (
                    uuid.uuid4(), tid, vname, vdisp, tenant_tin, vemail, vphone,
                    Decimal(str(random.randint(0, 150000))),
                    json.dumps({"city": vaddr.split(",")[0], "raw": vaddr, "country": vaddr.split(",")[-1].strip(), "currency": vcurr})
                ))
                vid = cur.fetchone()[0]
                tenant_vendor_map[tid].append((vid, vname, vdisp, vcurr, vemail))

        print(f"  finopsiq.vendors populated across {len(tenants)} tenants.")

        # ---------------------------------------------------------
        # 8. FINOPSIQ: CUSTOMERS
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.customers ---")
        customer_definitions = [
            ("Core Engineering Business Unit", "Eng-BU", "CUST-TIN-001", "eng-finance@genworx.ai"),
            ("Data Platform & Analytics Division", "Data-Division", "CUST-TIN-002", "data-lead@genworx.ai"),
            ("GenAI Applications Group", "GenAI-Group", "CUST-TIN-003", "genai-billing@genworx.ai"),
            ("Global Enterprise Sales & Go-To-Market", "GTM-Sales", "CUST-TIN-004", "salesops@genworx.ai"),
            ("Corporate Operations & HR Tech", "Corp-Ops", "CUST-TIN-005", "corp-finance@genworx.ai"),
            ("Customer Success & Professional Services", "CS-Services", "CUST-TIN-006", "cs-billing@genworx.ai")
        ]

        tenant_customer_map = {}
        for tid, _, _, _, _ in tenants:
            tenant_customer_map[tid] = []
            for cname, cdisp, ctin, cemail in customer_definitions:
                tenant_ctin = f"{ctin}-{str(tid)[:4]}"
                cur.execute("""
                    INSERT INTO finopsiq.customers (
                        customer_id, tenant_id, company_name, display_name, name, tin, email,
                        is_active, is_taxable, balance, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, true, false, 0, now() - interval '300 days', now())
                    ON CONFLICT (tenant_id, tin) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        display_name = EXCLUDED.display_name
                    RETURNING customer_id;
                """, (uuid.uuid4(), tid, cname, cdisp, cname, tenant_ctin, cemail))
                cid = cur.fetchone()[0]
                tenant_customer_map[tid].append((cid, cname, cemail))
        print("  finopsiq.customers populated.")

        # ---------------------------------------------------------
        # 9. FINOPSIQ: VENDOR INSTRUCTIONS, PARSER HEADERS, ALIASES & DRA MAILBOX
        # ---------------------------------------------------------
        print("\n--- Populating vendor_instructions, parser_headers, aliases & mailbox ---")
        rule_specs = [
            ("Auto-Approve AWS Reserved Instances", "If invoice matches active 3-year RI commitment and variance is under $5, auto-approve for payment.", "auto_approve", "low", True),
            ("Mandatory PO 3-Way Match for Hardware", "Require strict PO line matching for hardware purchases from Cisco and Dell before AP release.", "enforce_po_match", "high", True),
            ("Tax Withholding Validation TDS Section 194J", "Verify 10% TDS deduction for technical and professional consulting invoices from India-based entities.", "tax_check", "medium", True),
            ("Flag Over-Usage on LLM APIs > 20%", "If OpenAI or Anthropic monthly billing exceeds historical baseline by >20%, require VP sign-off.", "flag_anomaly", "high", True),
            ("Net-60 Payment Terms Validation", "Ensure payment terms match agreed Net-60 MSA schedule and prompt payment discounts are deducted.", "validate_terms", "medium", False)
        ]

        for tid, _, _, _, _ in tenants:
            # Mailbox config
            cur.execute("""
                INSERT INTO finopsiq.dra_mailbox_config (
                    tenant_id, mailbox_address, status, validated_at, created_at, updated_at
                )
                VALUES (%s, %s, 'connected', now(), now(), now())
                ON CONFLICT (tenant_id) DO NOTHING;
            """, (tid, f"invoices@{str(tid)[:4]}.finopsiq.ai"))

            for rname, rprompt, ract, rsev, rglob in rule_specs:
                cur.execute("""
                    INSERT INTO finopsiq.vendor_instructions (
                        id, tenant_id, rule_name, instruction_prompt, action_type, severity,
                        is_global, is_active, created_by, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, true, 'finops-admin@finopsiq.com', now(), now());
                """, (uuid.uuid4(), tid, rname, rprompt, ract, rsev, rglob))

            for vendor_idx, vitem in enumerate(tenant_vendor_map[tid][:5]):
                vid, vname, _, _, _ = vitem
                fp = f"FP-{vname[:4].upper()}-{str(tid)[:4]}-V1"
                cur.execute("""
                    INSERT INTO finopsiq.parser_headers (
                        parser_header_id, tenant_id, fingerprint, doc_type, vendor_id,
                        field_rules, table_config, match_count, status, learned_by, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, 'INVOICE', %s, '{\"invoice_num_regex\": \"INV-[0-9]+\", \"total_due_anchor\": \"Total Amount Due\"}'::jsonb, '{\"header_row\": 1, \"qty_col\": 3, \"total_col\": 5}'::jsonb, %s, 'trusted', 'llm', now(), now())
                    ON CONFLICT (tenant_id, fingerprint) DO NOTHING;
                """, (uuid.uuid4(), tid, fp, str(vid), random.randint(15, 80)))

            alias_pairs = [
                ("Amazon Web Services LLC", "VENDOR", tenant_vendor_map[tid][0][0]),
                ("AWS Cloud Services", "VENDOR", tenant_vendor_map[tid][0][0]),
                ("Microsoft Azure Ireland", "VENDOR", tenant_vendor_map[tid][1][0]),
                ("Google Cloud Inc EMEA", "VENDOR", tenant_vendor_map[tid][2][0]),
                ("Snowflake Computing Inc", "VENDOR", tenant_vendor_map[tid][3][0])
            ]
            for raw_n, etype, eid in alias_pairs:
                cur.execute("""
                    INSERT INTO finopsiq.genbooks_entity_aliases (
                        id, tenant_id, raw_name, entity_type, entity_id, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now(), now())
                    ON CONFLICT (tenant_id, entity_type, raw_name) DO NOTHING;
                """, (uuid.uuid4(), tid, raw_n, etype, eid))

        print("  vendor_instructions, parser_headers, aliases, and mailbox populated.")

        # ---------------------------------------------------------
        # 10. FINOPSIQ: PURCHASE ORDERS, PO LINE ITEMS & PO REVISIONS
        # ---------------------------------------------------------
        print("\n--- Populating purchase_orders, po_line_items & po_revisions ---")
        po_statuses = ["issued", "approved", "partially_billed", "fully_billed", "closed", "cancelled"]
        created_pos = []
        all_product_keys = list(product_id_map.keys())

        base_date = date(2025, 1, 10)
        po_counter = 1000

        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            vendors = tenant_vendor_map[tid]
            customers = tenant_customer_map[tid]

            for i in range(25):
                po_counter += 1
                vid, vname, vdisp, vcurr, vemail = random.choice(vendors)
                cid, cname, _ = random.choice(customers)
                po_num = f"PO-{base_date.year + (i % 2)}-{po_counter}"
                po_date = base_date + timedelta(days=i * 15 + random.randint(1, 10))
                exp_delivery = po_date + timedelta(days=30)
                status = random.choice(po_statuses)

                num_lines = random.randint(2, 5)
                selected_prods = random.sample(all_product_keys, num_lines)
                
                subtotal = Decimal('0.00')
                lines_info = []
                for l_idx, pcode in enumerate(selected_prods, start=1):
                    pid = product_id_map[pcode]
                    qty = Decimal(str(random.choice([1, 2, 5, 10, 24, 50, 100])))
                    unit_p = Decimal(str(random.choice([45.0, 120.0, 250.0, 850.0, 1500.0, 4200.0, 12000.0])))
                    ltot = (qty * unit_p).quantize(Decimal('0.01'))
                    subtotal += ltot
                    lines_info.append((pid, pcode, l_idx, qty, unit_p, ltot))

                discount = Decimal(str(random.choice([0, 0, 500, 1200]))) if subtotal > 5000 else Decimal('0.00')
                tax_amt = ((subtotal - discount) * Decimal('0.10')).quantize(Decimal('0.01'))
                total_amt = subtotal - discount + tax_amt

                po_id = uuid.uuid4()
                cur.execute("""
                    INSERT INTO finopsiq.purchase_orders (
                        po_id, tenant_id, vendor_id, customer_id, po_number, order_status,
                        po_date, expected_delivery_date, delivery_date, currency,
                        subtotal, discount_amount, total_amount, vendor_name, email,
                        terms_and_conditions, notes, show_status, is_deleted, apply_tax_after_discount,
                        created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        'Net-30. Subject to MSA terms.', 'Approved by Finance Committee.',
                        true, false, true, %s, %s
                    );
                """, (
                    po_id, tid, vid, cid, po_num, status,
                    po_date, exp_delivery, exp_delivery if status in ["fully_billed", "closed"] else None, vcurr,
                    subtotal, discount, total_amt, vname, vemail,
                    datetime.combine(po_date, datetime.min.time(), tzinfo=timezone.utc),
                    datetime.combine(po_date, datetime.min.time(), tzinfo=timezone.utc)
                ))

                po_lines = []
                for pid, pcode, l_idx, qty, unit_p, ltot in lines_info:
                    po_line_id = uuid.uuid4()
                    cur.execute("""
                        INSERT INTO finopsiq.po_line_items (
                            po_line_item_id, po_id, tenant_id, product_id, line_number,
                            item_name, description, quantity, unit_price, line_total,
                            apply_tax_after_discount, customer_id, customer_name,
                            created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s, %s, now(), now());
                    """, (
                        po_line_id, po_id, tid, pid, l_idx,
                        pcode, f"Procurement line item for {pcode}", qty, unit_p, ltot,
                        cid, cname
                    ))
                    po_lines.append((po_line_id, pid, pcode, qty, unit_p, ltot))

                if status in ["partially_billed", "fully_billed", "closed"] and i % 2 == 0:
                    rev_id = uuid.uuid4()
                    cur.execute("""
                        INSERT INTO finopsiq.po_revisions (
                            revision_id, po_id, revision_number, snapshot, edit_source,
                            edited_by, user_acknowledged_risk, created_at
                        )
                        VALUES (%s, %s, 1, %s, 'erp_sync', 'procurement-bot@finopsiq.com', true, now() - interval '15 days');
                    """, (rev_id, po_id, json.dumps({"subtotal": str(subtotal), "lines_count": len(po_lines), "po_number": po_num})))

                created_pos.append({
                    "po_id": po_id,
                    "po_number": po_num,
                    "tenant_id": tid,
                    "vendor_id": vid,
                    "customer_id": cid,
                    "currency": vcurr,
                    "subtotal": subtotal,
                    "total_amount": total_amt,
                    "po_date": po_date,
                    "status": status,
                    "lines": po_lines
                })

        print(f"  finopsiq.purchase_orders populated: {len(created_pos)} POs.")

        # ---------------------------------------------------------
        # 11. FINOPSIQ: CONTRACTS
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.contracts ---")
        contract_titles = [
            ("Master Cloud Infrastructure & Compute Agreement", "AWS Cloud Enterprise", 36, 120000.0, 360000.0),
            ("Enterprise Software License & Support Agreement (SLA)", "Microsoft Corp", 24, 75000.0, 150000.0),
            ("Managed Data Cloud & Warehousing Agreement", "Snowflake Inc", 12, 48000.0, 48000.0),
            ("Application Performance Monitoring & Observability MSA", "Datadog Inc", 24, 32000.0, 64000.0),
            ("Frontier Artificial Intelligence Model Access Contract", "Anthropic PBC", 12, 100000.0, 100000.0),
            ("Enterprise Source Code Management & CI/CD Master Contract", "GitHub Inc", 36, 25000.0, 75000.0),
            ("Data Center Core Switching Hardware Supply & Warranty Agreement", "Cisco Systems", 60, 22000.0, 110000.0),
            ("24x7 Cloud Site Reliability Engineering Staffing Agreement", "ThoughtWorks", 12, 95000.0, 95000.0)
        ]

        created_contracts = []
        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            vendors = tenant_vendor_map[tid]
            customers = tenant_customer_map[tid]

            for c_idx, (title, s_desc, term_m, ann_fee, tot_val) in enumerate(contract_titles):
                cid_val = uuid.uuid4()
                vid, _, _, _, _ = vendors[c_idx % len(vendors)]
                cust_id, _, _ = random.choice(customers)
                st_date = date(2024, 6, 1) + timedelta(days=c_idx * 45)
                exp_date = st_date + timedelta(days=term_m * 30)

                cur.execute("""
                    INSERT INTO finopsiq.contracts (
                        contract_id, tenant_id, vendor_id, customer_id, official_document_title,
                        assigned_contract_number, agreement_signed_date, start_date, expiry_date,
                        term_in_months, initial_annual_fee_value, total_contract_amount_value,
                        overall_contract_term_description, payment_terms_description, contract_type,
                        is_deleted, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, 'Net 45 payment terms upon invoice presentation.', 'MSA',
                        false, now() - interval '250 days', now()
                    );
                """, (
                    cid_val, tid, vid, cust_id, title,
                    f"CNT-2024-MSA-{c_idx + 101}", st_date - timedelta(days=10), st_date, exp_date,
                    term_m, Decimal(str(ann_fee)), Decimal(str(tot_val)),
                    f"Fixed commitment for {term_m} months with automatic annual indexation."
                ))
                created_contracts.append({
                    "contract_id": cid_val,
                    "tenant_id": tid,
                    "vendor_id": vid,
                    "total_value": Decimal(str(tot_val)),
                    "start_date": st_date,
                    "expiry_date": exp_date
                })
        print(f"  finopsiq.contracts populated: {len(created_contracts)} contracts.")

        # ---------------------------------------------------------
        # 12. FINOPSIQ: INVOICES, LINE ITEMS & INVOICE REVISIONS
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.invoices, invoice_line_items & invoice_revisions ---")
        created_invoices = []
        inv_counter = 5000

        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            vendors = tenant_vendor_map[tid]
            customers = tenant_customer_map[tid]
            tenant_pos = [p for p in created_pos if p["tenant_id"] == tid]

            for j in range(35):
                inv_counter += 1
                vid, vname, vdisp, vcurr, vemail = random.choice(vendors)
                cid, cname, cemail = random.choice(customers)
                
                matching_po = None
                if random.random() > 0.35 and tenant_pos:
                    matching_po = random.choice(tenant_pos)
                    vid = matching_po["vendor_id"]
                    cid = matching_po["customer_id"]
                    vcurr = matching_po["currency"]

                inv_num = f"INV-{2025 + (j % 2)}-{inv_counter}"
                inv_date = date(2025, 2, 1) + timedelta(days=j * 15 + random.randint(1, 5))
                due_date = inv_date + timedelta(days=random.choice([15, 30, 45, 60]))

                is_past_due = due_date < date(2026, 10, 1)
                
                # Assign status and balance_due
                status_choices = ["paid", "paid", "partially_paid", "overdue", "void"] if is_past_due else ["pending", "ready_to_pay", "needs_review", "paid"]
                inv_status = random.choice(status_choices)
                inv_type = random.choice(["AP", "AP", "AP", "AR"])

                num_lines = random.randint(1, 4)
                subtotal = Decimal('0.00')
                inv_lines = []

                if matching_po and matching_po["lines"]:
                    for l_idx, p_line in enumerate(matching_po["lines"][:num_lines], start=1):
                        po_line_id, pid, pcode, p_qty, p_up, p_ltot = p_line
                        has_var = random.random() < 0.25
                        inv_qty = p_qty if not has_var else (p_qty + Decimal('2.00'))
                        inv_up = p_up if not has_var else (p_up * Decimal('1.15')).quantize(Decimal('0.01'))
                        ltot = (inv_qty * inv_up).quantize(Decimal('0.01'))
                        subtotal += ltot
                        inv_lines.append((po_line_id, pid, pcode, l_idx, inv_qty, inv_up, ltot, has_var))
                else:
                    selected_prods = random.sample(all_product_keys, num_lines)
                    for l_idx, pcode in enumerate(selected_prods, start=1):
                        pid = product_id_map[pcode]
                        qty = Decimal(str(random.choice([1, 4, 12, 30, 80])))
                        up = Decimal(str(random.choice([85.0, 320.0, 750.0, 2200.0, 8500.0])))
                        ltot = (qty * up).quantize(Decimal('0.01'))
                        subtotal += ltot
                        inv_lines.append((None, pid, pcode, l_idx, qty, up, ltot, False))

                discount = Decimal(str(random.choice([0, 0, 250, 800]))) if subtotal > 3000 else Decimal('0.00')
                tax_amt = ((subtotal - discount) * Decimal('0.0825')).quantize(Decimal('0.01'))
                total_amt = subtotal - discount + tax_amt

                if inv_status == "paid":
                    balance_due = Decimal('0.00')
                elif inv_status == "partially_paid":
                    balance_due = (total_amt * Decimal('0.40')).quantize(Decimal('0.01'))
                else:
                    balance_due = total_amt

                fx_rates = {"USD": Decimal('1.000'), "EUR": Decimal('1.085'), "GBP": Decimal('1.295'), "INR": Decimal('0.0119')}
                fx_rate = fx_rates.get(vcurr, Decimal('1.000'))
                amount_usd = (total_amt * fx_rate).quantize(Decimal('0.01'))

                inv_id = uuid.uuid4()
                cur.execute("""
                    INSERT INTO finopsiq.invoices (
                        invoice_id, tenant_id, vendor_id, customer_id, purchase_order_number,
                        invoice_number, invoice_type, invoice_received_date, invoice_date, due_date, currency,
                        subtotal, discount_amount, total_amount, balance_due, status,
                        vendor_email, customer_email, fx_rate_used, amount_usd,
                        fx_rate_date, show_status, is_deleted, apply_tax_after_discount,
                        created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, true, false, true,
                        %s, %s
                    );
                """, (
                    inv_id, tid, vid, cid, matching_po["po_number"] if matching_po else None,
                    inv_num, inv_type, datetime.combine(inv_date, datetime.min.time(), tzinfo=timezone.utc), inv_date, due_date, vcurr,
                    subtotal, discount, total_amt, balance_due, inv_status,
                    vemail, cemail, fx_rate, amount_usd,
                    inv_date,
                    datetime.combine(inv_date, datetime.min.time(), tzinfo=timezone.utc),
                    datetime.combine(inv_date, datetime.min.time(), tzinfo=timezone.utc)
                ))

                for po_lid, pid, pcode, l_idx, qty, up, ltot, _ in inv_lines:
                    cur.execute("""
                        INSERT INTO finopsiq.invoice_line_items (
                            invoice_line_item_id, invoice_id, tenant_id, po_line_item_id, product_id,
                            line_number, item_name, description, quantity, unit_price, line_total,
                            apply_tax_after_discount, customer_id, customer_name, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s, %s, now(), now());
                    """, (
                        uuid.uuid4(), inv_id, tid, po_lid, pid,
                        l_idx, pcode, f"Invoice billing line for {pcode}", qty, up, ltot,
                        cid, cname
                    ))

                if j % 3 == 0:
                    cur.execute("""
                        INSERT INTO finopsiq.invoice_revisions (
                            revision_id, invoice_id, revision_number, snapshot, edit_source,
                            edited_by, user_acknowledged_risk, created_at
                        )
                        VALUES (%s, %s, 1, %s, 'ocr_correction', 'ap-specialist@finopsiq.com', true, now() - interval '10 days');
                    """, (uuid.uuid4(), inv_id, json.dumps({"invoice_number": inv_num, "total_amount": str(total_amt), "status": inv_status})))

                created_invoices.append({
                    "invoice_id": inv_id,
                    "invoice_number": inv_num,
                    "tenant_id": tid,
                    "vendor_id": vid,
                    "customer_id": cid,
                    "po_number": matching_po["po_number"] if matching_po else None,
                    "total_amount": total_amt,
                    "balance_due": balance_due,
                    "status": inv_status,
                    "invoice_date": inv_date,
                    "due_date": due_date,
                    "currency": vcurr,
                    "lines": inv_lines,
                    "matching_po": matching_po
                })

        print(f"  finopsiq.invoices populated: {len(created_invoices)} invoices.")

        # ---------------------------------------------------------
        # 13. FINOPSIQ: PAYMENT HISTORY
        # ---------------------------------------------------------
        print("\n--- Populating finopsiq.payment_history ---")
        payment_count = 0
        for inv in created_invoices:
            if inv["status"] in ["paid", "partially_paid"]:
                paid_amt = inv["total_amount"] if inv["status"] == "paid" else (inv["total_amount"] - inv["balance_due"])
                cur.execute("""
                    INSERT INTO finopsiq.payment_history (
                        payment_id, tenant_id, invoice_id, amount, payment_date,
                        action, currency, balance_after, recorded_by, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'treasury-daemon@finopsiq.com', now(), now());
                """, (
                    uuid.uuid4(), inv["tenant_id"], inv["invoice_id"], paid_amt,
                    inv["invoice_date"] + timedelta(days=14),
                    "full_settlement" if inv["status"] == "paid" else "partial_payment",
                    inv["currency"], inv["balance_due"]
                ))
                payment_count += 1
        print(f"  finopsiq.payment_history populated: {payment_count} payments.")

        # ---------------------------------------------------------
        # 14. FINOPSIQ: RELEVANT DOCUMENTS, DISCREPANCIES & IGNORED DISCREPANCIES
        # ---------------------------------------------------------
        print("\n--- Populating relevant_document_details, discrepancies_details & resolutions ---")
        discrepancy_count = 0
        for inv in created_invoices:
            if inv["matching_po"]:
                match_id = uuid.uuid4()
                has_any_var = any(line[7] for line in inv["lines"])
                m_conf = Decimal('0.98') if not has_any_var else Decimal('0.74')
                m_stat = "completed" if not has_any_var else "partial"

                cur.execute("""
                    INSERT INTO finopsiq.relevant_document_details (
                        match_id, tenant_id, invoice_id, match_status, source_type,
                        match_type, match_confidence, po_details, processing_metadata,
                        processed_at, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, 'ERP_CONNECTOR', 'EXACT_PO_MATCH', %s, %s, '{}'::jsonb, now(), now(), now());
                """, (
                    match_id, inv["tenant_id"], inv["invoice_id"], m_stat, m_conf,
                    json.dumps({"matched_po_number": inv["matching_po"]["po_number"], "po_amount": str(inv["matching_po"]["total_amount"])})
                ))

                if has_any_var or inv["status"] == "needs_review":
                    disc_id = uuid.uuid4()
                    res_id = uuid.uuid4()
                    conf_breakdown = json.dumps({"price_score": 0.85, "qty_score": 0.92})
                    comp_details = json.dumps({"variance_type": "unit_price_exceeded", "po_rate": 100.0, "inv_rate": 115.0})
                    disc_summary = json.dumps({"summary": "Invoice unit price exceeds purchase order contract price by 15 percent.", "severity": "HIGH"})
                    cur.execute("""
                        INSERT INTO finopsiq.discrepancies_details (
                            discrepancies_id, tenant_id, match_id, match_type, has_discrepancy,
                            confidence_score, confidence_breakdown, comparison_discrepancies_details,
                            basic_validation_details, missing_relevant_documents, matched_details,
                            discrepancy_summary, processing_metadata, dra_trigger_status, dra_attempts,
                            created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, 'PO_VARIANCE', true,
                            0.88, %s::jsonb,
                            %s::jsonb,
                            '{"vendor_tin_matched": true}'::jsonb, false, '{"po_num": "PO"}'::jsonb,
                            %s::jsonb,
                            '{}'::jsonb, 'triggered', 1, now(), now()
                        );
                    """, (disc_id, inv["tenant_id"], match_id, conf_breakdown, comp_details, disc_summary))

                    res_status = random.choice(["Awaiting_Review", "Email_Sent", "Closed", "Manual_Review"])
                    cur.execute("""
                        INSERT INTO finopsiq.discrepancy_resolution (
                            discrepancy_id, resolution_id, match_id, invoice_type, discrepancy_type,
                            vendor_email, finance_admin_id, status, retry_count, trigger_time,
                            resolution_notes, created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, 'STANDARD', 'PRICE_MISMATCH',
                            'vendor-ap@cloudvendor.com', 'admin-sarah@genworx.ai', %s, 0, now() - interval '5 days',
                            'Discrepancy flagged: contacted vendor account executive for credit memo.',
                            now(), now()
                        );
                    """, (disc_id, res_id, match_id, res_status))

                    cur.execute("""
                        INSERT INTO finopsiq.resolution_communication (
                            communication_id, resolution_id, tenant_id, vendor_name, vendor_email,
                            direction, type, subject, body_content, status, retry_count,
                            created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, 'Cloud Vendor AP', 'vendor-ap@cloudvendor.com',
                            'OUTBOUND', 'EMAIL', %s,
                            '{\"text\": \"Dispute notification: unit price on invoice does not match PO agreed rate.\"}'::jsonb,
                            'sent', 0, now(), now()
                        );
                    """, (uuid.uuid4(), res_id, inv["tenant_id"], f"Price Discrepancy on {inv['invoice_number']}"))

                    cur.execute("""
                        INSERT INTO finopsiq.resolution_audit_log (
                            audit_id, resolution_id, action_type, performed_by, details, created_at
                        )
                        VALUES (%s, %s, 'STATUS_UPDATE', 'sarah.miller@genworx.ai', '{\"status\": \"dispute_opened\"}'::jsonb, now());
                    """, (uuid.uuid4(), res_id))

                    # Ignored discrepancies (minor tolerances)
                    if random.random() < 0.35:
                        cur.execute("""
                            INSERT INTO finopsiq.ignored_discrepancies (
                                ignored_discrepancy_id, tenant_id, vendor_id, invoice_id,
                                discrepancy_type, is_future_rule, ignored_by_user_id, is_active,
                                message, severity, created_at
                            )
                            VALUES (%s, %s, %s, %s, 'ROUNDING_TOLERANCE', false, %s, true, 'Acceptable FX rounding error < $1.00', 'LOW', now());
                        """, (uuid.uuid4(), inv["tenant_id"], inv["vendor_id"], inv["invoice_id"], list(user_id_map.values())[0][0]))

                    discrepancy_count += 1

        print(f"  finopsiq.discrepancies & resolutions populated: {discrepancy_count} discrepancy records.")

        # ---------------------------------------------------------
        # 15. FINOPSIQ: REMINDERS & REMINDER STATE
        # ---------------------------------------------------------
        print("\n--- Populating reminder_configuration, alert_config, invoice_reminder_state & reminders ---")
        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            rem_levels = [
                (1, 1, 5, "Upcoming Due Date", "Gentle Reminder", "polite"),
                (2, 0, 0, "Due Today", "Payment Due Notice", "firm"),
                (3, 7, 14, "Past Due Notice", "Urgent Overdue Alert", "assertive"),
                (4, 30, 60, "Severe Overdue Demand", "Final Notice Before Service Hold", "legal")
            ]
            for lvl, fdays, tdays, dlbl, rlbl, rtone in rem_levels:
                cur.execute("""
                    INSERT INTO finopsiq.reminder_configuration (
                        config_id, tenant_id, level, from_days, to_days, display_label,
                        reminder_label, reminder_tone, is_enabled, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true, now(), now())
                    ON CONFLICT (tenant_id, level) DO UPDATE SET display_label = EXCLUDED.display_label;
                """, (uuid.uuid4(), tid, lvl, fdays, tdays, dlbl, rlbl, rtone))

            cur.execute("""
                INSERT INTO finopsiq.reminder_alert_config (
                    config_id, tenant_id, is_enabled, recipient_emails, delay_minutes, created_at, updated_at
                )
                VALUES (%s, %s, true, '[\"ap-alerts@finopsiq.com\"]'::jsonb, 15, now(), now())
                ON CONFLICT (tenant_id) DO NOTHING;
            """, (uuid.uuid4(), tid))

        for inv in created_invoices:
            if inv["balance_due"] > 0:
                cur.execute("""
                    INSERT INTO finopsiq.invoice_reminder_state (
                        invoice_id, tenant_id, status, next_level, invoice_number,
                        total_amount, amount_paid, amount_due, currency, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, 1, %s,
                        %s, %s, %s, %s, now(), now()
                    )
                    ON CONFLICT (invoice_id) DO UPDATE SET amount_due = EXCLUDED.amount_due;
                """, (
                    inv["invoice_id"], inv["tenant_id"],
                    "ACTIVE" if inv["status"] != "void" else "VOID",
                    inv["invoice_number"], inv["total_amount"],
                    inv["total_amount"] - inv["balance_due"], inv["balance_due"], inv["currency"]
                ))

                if inv["status"] == "overdue":
                    cur.execute("""
                        INSERT INTO finopsiq.reminders (
                            reminder_id, invoice_id, tenant_id, level, reminder_label,
                            reminder_tone, status, recipient_email, sent_at, attempt, created_at
                        )
                        VALUES (%s, %s, %s, 2, 'Past Due Notice', 'firm', 'SENT', 'ap@enterprise.com', now() - interval '2 days', 1, now());
                    """, (uuid.uuid4(), inv["invoice_id"], inv["tenant_id"]))

        print("  reminders and invoice_reminder_state populated.")

        # ---------------------------------------------------------
        # 16. FINOPSIQ: GENBOOKS REQUESTS, INBOUND EMAILS & AUDIT LOGS
        # ---------------------------------------------------------
        print("\n--- Populating genbooks_requests, email_context, entity_audit_log & system_audit_log ---")
        for inv in created_invoices[:30]:
            cur.execute("""
                INSERT INTO finopsiq.genbooks_requests (
                    request_id, invoice_id, tenant_id, status, channel,
                    invoice_type, ar_status, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, 'EMAIL_INGESTION', 'STANDARD', %s, now() - interval '10 days', now())
                ON CONFLICT (invoice_id, tenant_id) DO NOTHING;
            """, (
                uuid.uuid4(), inv["invoice_id"], inv["tenant_id"],
                "ready_to_pay" if inv["status"] == "ready_to_pay" else "needs_review",
                "overdue" if inv["status"] == "overdue" else "sent"
            ))

            msg_id = f"<msg-{inv['invoice_number'].lower()}@cloudvendor.com>"
            email_ctx_id = uuid.uuid4()
            cur.execute("""
                INSERT INTO finopsiq.email_context (
                    id, internet_message_id, tenant_id, conversation_id, subject,
                    sender, recipients, intent_category, received_at, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, ARRAY['invoices@genworx.ai']::text[], 'INVOICE_SUBMISSION', now() - interval '12 days', now(), now())
                ON CONFLICT (tenant_id, internet_message_id) DO NOTHING;
            """, (
                email_ctx_id, msg_id, inv["tenant_id"], f"conv-{inv['invoice_number']}",
                f"New Invoice Issued: {inv['invoice_number']}",
                "billing-service@cloudvendor.com"
            ))

            cur.execute("""
                INSERT INTO finopsiq.email_context_link (
                    id, tenant_id, context_id, entity_type, entity_id, linked_by,
                    link_type, created_at, updated_at
                )
                VALUES (%s, %s, %s, 'INVOICE', %s, %s, 'LINKED', now(), now())
                ON CONFLICT (context_id, entity_type, entity_id) DO NOTHING;
            """, (uuid.uuid4(), inv["tenant_id"], email_ctx_id, inv["invoice_id"], list(user_id_map.values())[0][0]))

            cur.execute("""
                INSERT INTO finopsiq.entity_audit_log (
                    audit_id, invoice_id, tenant_id, action_type, performed_by,
                    performed_by_name, summary, changes, metadata, created_at
                )
                VALUES (%s, %s, %s, 'status_change', 'ap-engine', 'Automated AP Ingestion Engine', 'Invoice processed and status updated', '{\"old_status\": \"pending\", \"new_status\": \"needs_review\"}'::jsonb, '{}'::jsonb, now());
            """, (uuid.uuid4(), inv["invoice_id"], inv["tenant_id"]))

        unmatched_reasons = ["no_invoice_candidate", "ambiguous_invoice_match", "wrong_invoice_type", "sender_unauthorized"]
        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            for r in unmatched_reasons:
                cur.execute("""
                    INSERT INTO finopsiq.unmatched_inbound_email (
                        id, tenant_id, sender_email, sender_name, subject, reason, received_at, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, now() - interval '3 days', now());
                """, (
                    uuid.uuid4(), tid, "unknown-vendor@unregistered.io", "Unknown Vendor AP",
                    "Billing Inquiry regarding past payment", r
                ))

        sys_actions = ["synced", "created", "approved", "paid", "edited", "linked"]
        for tid in [genworx_org_id, test_org_id, acme_org_id]:
            for act in sys_actions:
                cur.execute("""
                    INSERT INTO finopsiq.system_audit_log (
                        audit_id, tenant_id, actor_id, actor_name, action, outcome,
                        entity_group, entity_type, summary, created_at
                    )
                    VALUES (%s, %s, 'system-worker', 'Background FinOps Processor', %s, 'success', 'payables', 'INVOICE', 'Scheduled batch reconciliation pass completed successfully', now());
                """, (uuid.uuid4(), tid, act))

        print("  genbooks_requests, email_context, and audit logs populated.")

        conn.commit()
        print("\n>>> ALL SYNTHETIC DATA SUCCESSFULLY COMMITTED TO DATABASE! <<<")

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] Transaction failed and was rolled back: {e}", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    generate_synthetic_data()
