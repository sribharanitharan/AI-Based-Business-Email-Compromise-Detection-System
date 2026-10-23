"""
generate_100_emails.py — Generates 120 realistic RFC 822 .eml files
Targeted to sribharani101@gmail.com, with a realistic mix of:
  - 40 BEC attack scenarios (CEO fraud, Payroll diversion, Invoice fraud, Gift card scams, Credential traps)
  - 80 Legitimate corporate emails (Meeting agendas, Reports, IT notices, HR updates, Vendor communications)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import random

TARGET_EMAIL = "sribharani101@gmail.com"
OUTPUT_DIR = Path("data/raw/emails")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Templates ─────────────────────────────────────────────────────────────────

BEC_SCENARIOS = [
    # CEO Impersonation
    {
        "subject": "URGENT: Confidential Acquisition Transfer Required TODAY",
        "sender": "CEO Richard Vance <ceo@comp4ny-exec.xyz>",
        "reply_to": "exec-desk@gmail.com",
        "body": "Hi Sri,\n\nI am currently in an executive board meeting regarding a confidential acquisition and cannot take calls. I need you to immediately execute a priority wire transfer of $85,000 to our acquiring partner's escrow account.\n\nBeneficiary: Apex Legal Escrow\nBank: Chase Federal NY\nRouting: 021000021\nAccount: 4892019342\n\nDo NOT discuss this with anyone on the floor until the closing announcement. Send me the wire confirmation receipt ASAP.\n\nBest,\nRichard Vance\nCEO",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": ["http://bit.ly/wire-confirm-88"]
    },
    {
        "subject": "Executive Directive: Immediate Wire Payment Needed",
        "sender": "Mark Sterling <director@corp-executive.live>",
        "reply_to": "mark.sterling77@outlook.com",
        "body": "Hello Sri,\n\nAre you at your desk? We have a critical overdue retainer with our European counsel that must be settled before 2 PM GMT to avoid litigation.\n\nPlease process an international SWIFT wire of €42,000:\nSWIFT: DEUTDEDBM\nIBAN: DE89370400440532013000\n\nI have pre-authorized this expense. Confirm once transferred.",
        "headers": {"spf": "softfail", "dkim": "none"},
        "urls": []
    },
    # Payroll Redirection
    {
        "subject": "Request to Update Direct Deposit Banking Details",
        "sender": "David Miller <david.miller@company-hr.co>",
        "reply_to": "david.m.updates@gmail.com",
        "body": "Hi Payroll Team,\n\nI recently switched my primary checking account. Could you please update my direct deposit allocation starting with the next pay period on the 1st?\n\nNew Account Information:\nBank: Wells Fargo NA\nRouting Number: 121000248\nAccount Number: 8872619041\nAccount Type: Checking\n\nPlease let me know if you need a voided check. I would appreciate an immediate confirmation when completed.\n\nThanks,\nDavid Miller\nVP Engineering",
        "headers": {"spf": "fail", "dkim": "neutral"},
        "urls": []
    },
    {
        "subject": "Direct Deposit Change - Effective Immediately",
        "sender": "Sarah Jenkins <s.jenkins@mycompany-portal.xyz>",
        "reply_to": "sarah.j.deposit@protonmail.com",
        "body": "Hello Sri,\n\nPlease update my payroll direct deposit details to my new account before this week's processing deadline:\nRouting: 071000013\nAccount: 7492810394\n\nConfirm once updated. Thank you!\nSarah",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": []
    },
    # Invoice Fraud
    {
        "subject": "Overdue Invoice #INV-2026-9042 - Updated Remittance Account",
        "sender": "Accounts Receivable <billing@cloud-sys-vendor.com>",
        "reply_to": "payments-desk@billing-cloud.xyz",
        "body": "Dear Valued Customer,\n\nPlease be advised that our banking partner has changed due to annual financial audits. All pending remittances for Invoice #INV-2026-9042 ($38,750.00) must now be wired to our new clearing account:\n\nBank: Citibank NA\nRouting: 021000089\nAccount: 9948271034\nBeneficiary: Cloud Systems International\n\nPlease do NOT send funds to our previous coordinates.\n\nSincerely,\nBilling Dept.",
        "headers": {"spf": "softfail", "dkim": "fail"},
        "urls": ["http://invoice-view.xyz/download/9042"]
    },
    {
        "subject": "Urgent: Payment Discrepancy on Outstanding Balance",
        "sender": "Global Logistics <finance@gl0bal-logistics.net>",
        "reply_to": "gl-finance@outlook.com",
        "body": "Sri,\n\nOur systems indicate that payment of $19,400 for shipment #GL-882 is 14 days overdue. Please process this ACH payment immediately using our updated account:\nRouting: 322271627\nAccount: 5548291032\n\nFailure to remit today will result in shipment hold.",
        "headers": {"spf": "fail", "dkim": "none"},
        "urls": ["http://203.0.113.19/pay"]
    },
    # Gift Card Scams
    {
        "subject": "Quick task needed right now",
        "sender": "Chief Executive Officer <ceo.internal@exec-company.vip>",
        "reply_to": "ceo.urgent.tasks@gmail.com",
        "body": "Hi Sri,\n\nAre you available? I am in a conference meeting with prospective investors and need your help with an urgent favor. I need to reward our top speakers with client incentives.\n\nCould you please run to the nearest store and purchase 6 Apple Gift Cards ($100 each)? Scratch the back coating and email me clear photos of the redemption codes right away. I will reimburse you by tomorrow morning with an expense voucher.\n\nPlease keep this confidential as it is a surprise.\n\nRegards,\nCEO",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": []
    },
    {
        "subject": "Store request - Client appreciation vouchers",
        "sender": "Amanda Cole <amanda.cole@corp-vip.org>",
        "reply_to": "amanda.c.desk@gmail.com",
        "body": "Sri, can you handle a confidential assignment? I need 4 Google Play or Steam gift cards ($100 denomination each) for our audit team today. Email me the codes as soon as purchased. Do not mention this to the team.",
        "headers": {"spf": "softfail", "dkim": "none"},
        "urls": []
    },
    # Credential Harvesting / Threat Traps
    {
        "subject": "CRITICAL: Corporate Email Password Expiring in 2 Hours",
        "sender": "IT Helpdesk <support@microsoft-security-alert.xyz>",
        "reply_to": "",
        "body": "Dear Sri,\n\nYour corporate email password will expire today at 5:00 PM. To prevent disruption in receiving emails, you must retain your existing password by verifying your credentials at the link below:\n\nhttp://192.168.1.105/auth/login?user=sribharani101@gmail.com\n\nIT Security Operations",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": ["http://192.168.1.105/auth/login"]
    },
    {
        "subject": "Action Required: Suspicious Login Detected on Your Account",
        "sender": "Security Operations <security@verify-portal.top>",
        "reply_to": "admin@login-support.info",
        "body": "We detected an unauthorized sign-in attempt from IP 185.220.101.5 (Kyiv, Ukraine). If this was not you, confirm your identity immediately: http://verify-corp-session.icu/reset-password",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": ["http://verify-corp-session.icu/reset-password"]
    }
]

LEGIT_TOPICS = [
    ("Q3 Financial Performance & Revenue Summary", "Hi Sri,\n\nAttached is the quarterly financial deck for your review. Revenues increased 14% year-over-year with strong margins in software consulting. Let's discuss at Thursday's standup.\n\nBest,\nFinance Team", "finance@company.com"),
    ("Sprint 42 Retrospective & Backlog Refinement", "Hi team,\n\nTomorrow at 10 AM we will hold our Sprint 42 retrospective. Please update Jira tickets and log your remaining story points beforehand.\n\nLink to meeting: https://zoom.us/j/9482710492\n\nThanks,\nScrum Master", "scrum@company.com"),
    ("Scheduled Data Center Maintenance Notice", "Dear Employees,\n\nPlease note our primary database cluster will undergo routine patch maintenance this Saturday between 02:00 UTC and 06:00 UTC. System services may be intermittently unavailable.\n\nIT Operations", "it-ops@company.com"),
    ("Welcome Aboard: New Engineering Cohort", "Please join us in welcoming our three new software engineers starting this Monday! They will be joining the Cloud Platforms team.\n\nHR Department", "hr@company.com"),
    ("Weekly Team Lunch at Riverside Cafe", "Hi all,\n\nTeam lunch is scheduled for Friday at 12:30 PM. Please RSVP by Thursday evening so we can finalize the reservation headcount.\n\nCheers,\nEvents Committee", "events@company.com"),
    ("Updated Holiday & PTO Guidelines 2026", "Team,\n\nPlease review the updated paid time off and public holiday calendar for the remaining quarters of 2026. Accessible on the internal wiki portal.\n\nPeople Operations", "people@company.com"),
    ("Customer Success Feedback — Acme Corp Contract", "Sri,\n\nGreat news! Acme Corp just signed their multi-year renewal agreement. The client explicitly commended our response times and reliability.\n\nKudos to the entire support team!\n\nSales Director", "sales@company.com"),
    ("Weekly Code Review & Security Best Practices", "Engineering team,\n\nHere is our weekly digest of security review notes and dependency updates. All CVE patches for Python libraries have been merged into develop.\n\nLead Architect", "eng-lead@company.com"),
    ("Health Insurance Enrollment Window Open", "This is a reminder that open enrollment for medical, dental, and vision insurance ends on the 25th of this month. Visit the benefits portal to review your plans.\n\nBenefits Team", "benefits@company.com"),
    ("Quarterly Townhall Meeting Agenda", "All Hands,\n\nJoin us this Thursday at 3 PM for our Q3 Townhall where leadership will discuss product roadmaps and answer live Q&A.\n\nExecutive Office", "allhands@company.com")
]


def generate_emails(count=120):
    start_date = datetime.now() - timedelta(days=20)
    created_files = []

    for i in range(1, count + 1):
        # ~35% BEC, 65% Legit
        is_bec = (random.random() < 0.35)
        email_date = start_date + timedelta(hours=i * 3.5 + random.randint(-60, 60))
        date_str = email_date.strftime("%a, %d %b %Y %H:%M:%S +0530")

        if is_bec:
            template = random.choice(BEC_SCENARIOS)
            subject = template["subject"]
            sender = template["sender"]
            reply_to = template["reply_to"]
            body = template["body"]
            spf = template["headers"]["spf"]
            dkim = template["headers"]["dkim"]
            dmarc = "fail"
            category = "BEC"
        else:
            subj_title, body_text, sndr_addr = random.choice(LEGIT_TOPICS)
            subject = f"{subj_title} #{i}"
            sender = sndr_addr
            reply_to = ""
            body = body_text
            spf = "pass"
            dkim = "pass"
            dmarc = "pass"
            category = "LEGIT"

        msg_id = f"<{category.lower()}-{i:03d}-{random.randint(10000,99999)}@corp-mail.net>"

        # Construct standard RFC 822 .eml format
        eml_content = f"""From: {sender}
To: {TARGET_EMAIL}
Date: {date_str}
Subject: {subject}
Message-ID: {msg_id}
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"
Content-Transfer-Encoding: 8bit
Authentication-Results: mx.google.com; spf={spf} smtp.mailfrom={sender}; dkim={dkim}; dmarc={dmarc}
Received-SPF: {spf} (google.com: domain designates sender IP)
"""
        if reply_to:
            eml_content += f"Reply-To: {reply_to}\n"

        eml_content += f"\n{body}\n"

        filename = f"email_{i:03d}_{category}.eml"
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(eml_content)
        created_files.append(filename)

    return len(created_files)


if __name__ == "__main__":
    total = generate_emails(120)
    print(f"[OK] Generated {total} realistic .eml files in {OUTPUT_DIR}")
