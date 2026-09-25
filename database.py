"""
============================================================
BorderShield AI - PostgreSQL Database Layer
Neon PostgreSQL + FastAPI
============================================================
"""

import os
import json
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor


# ============================================================
# DATABASE CONNECTION
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """
    Create a PostgreSQL connection using the DATABASE_URL
    stored in the environment.
    """

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set."
        )

    return psycopg2.connect(DATABASE_URL)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        # ====================================================
        # IDENTITIES TABLE
        # ====================================================

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS identities (

            id SERIAL PRIMARY KEY,

            full_name TEXT,

            document_number TEXT UNIQUE,

            document_type TEXT,

            nationality TEXT,

            risk_level TEXT DEFAULT 'LOW',

            watchlist_flag INTEGER DEFAULT 0,

            fraud_flag INTEGER DEFAULT 0,

            created_at TEXT

        )
        """)

        # ====================================================
        # SCREENINGS TABLE
        # ====================================================

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenings (

            id SERIAL PRIMARY KEY,

            identity_id INTEGER,

            full_name TEXT,

            document_number TEXT,

            document_type TEXT,

            risk_score INTEGER,

            status TEXT,

            issues TEXT,

            checkpoint TEXT,

            officer_name TEXT,

            created_at TEXT,

            FOREIGN KEY(identity_id)
            REFERENCES identities(id)

        )
        """)

        # ====================================================
        # FRAUD CASES TABLE
        # ====================================================

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_cases (

            id SERIAL PRIMARY KEY,

            identity_id INTEGER,

            document_number TEXT,

            case_type TEXT,

            severity TEXT,

            description TEXT,

            status TEXT DEFAULT 'OPEN',

            created_at TEXT,

            FOREIGN KEY(identity_id)
            REFERENCES identities(id)

        )
        """)

        # ====================================================
        # CHECKPOINTS TABLE
        # ====================================================

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (

            id SERIAL PRIMARY KEY,

            checkpoint_name TEXT,

            location TEXT,

            status TEXT DEFAULT 'ACTIVE',

            last_activity TEXT

        )
        """)

        # ====================================================
        # AUDIT LOGS TABLE
        # ====================================================

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (

            id SERIAL PRIMARY KEY,

            officer_name TEXT,

            action TEXT,

            details TEXT,

            created_at TEXT

        )
        """)

        conn.commit()

        print(
            "BorderShield PostgreSQL database initialized successfully."
        )

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# CREATE OR GET IDENTITY
# ============================================================

def create_or_get_identity(
    full_name=None,
    document_number=None,
    document_type="Unknown",
    name=None
):

    # Allow main.py to use either name or full_name

    if name and not full_name:
        full_name = name

    if not document_number:
        return None

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # ====================================================
        # CHECK EXISTING IDENTITY
        # ====================================================

        cursor.execute("""
            SELECT id
            FROM identities
            WHERE document_number = %s
        """, (
            document_number,
        ))

        existing = cursor.fetchone()

        if existing:

            return existing["id"]

        # ====================================================
        # CREATE NEW IDENTITY
        # ====================================================

        created_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO identities (
                full_name,
                document_number,
                document_type,
                created_at
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (
            full_name or "Unknown",
            document_number,
            document_type or "Unknown",
            created_at
        ))

        identity_id = cursor.fetchone()["id"]

        conn.commit()

        return identity_id

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# SAVE SCREENING
# ============================================================

def save_screening(
    identity_id=None,
    full_name="Unknown",
    document_number="Not detected",
    document_type="Unknown",
    risk_score=0,
    status="LOW RISK",
    issues=None,
    checkpoint="Main Border Checkpoint",
    officer_name="BorderShield AI"
):

    if issues is None:
        issues = []

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        created_at = datetime.now().isoformat()

        issues_json = json.dumps(
            issues,
            ensure_ascii=False
        )

        cursor.execute("""
            INSERT INTO screenings (

                identity_id,
                full_name,
                document_number,
                document_type,
                risk_score,
                status,
                issues,
                checkpoint,
                officer_name,
                created_at

            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )

            RETURNING id

        """, (

            identity_id,

            full_name or "Unknown",

            document_number or "Not detected",

            document_type or "Unknown",

            risk_score,

            status,

            issues_json,

            checkpoint,

            officer_name,

            created_at

        ))

        screening_id = cursor.fetchone()["id"]

        conn.commit()

        return screening_id

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# GET IDENTITY INTELLIGENCE
# ============================================================

def get_identity_intelligence(identity_id):

    if not identity_id:
        return None

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # ====================================================
        # GET IDENTITY
        # ====================================================

        cursor.execute("""
            SELECT *
            FROM identities
            WHERE id = %s
        """, (
            identity_id,
        ))

        identity = cursor.fetchone()

        if not identity:
            return None

        identity = dict(identity)

        document_number = identity.get(
            "document_number"
        )

        # ====================================================
        # GET SCREENING HISTORY
        # ====================================================

        cursor.execute("""
            SELECT *
            FROM screenings
            WHERE identity_id = %s
            ORDER BY created_at DESC
        """, (
            identity_id,
        ))

        screenings = [
            dict(row)
            for row in cursor.fetchall()
        ]

        # ====================================================
        # GET FRAUD CASES
        # ====================================================

        cursor.execute("""
            SELECT *
            FROM fraud_cases
            WHERE identity_id = %s
            ORDER BY created_at DESC
        """, (
            identity_id,
        ))

        fraud_cases = [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:

        conn.close()

    # ========================================================
    # CALCULATE ALERTS
    # ========================================================

    alerts = []

    if identity.get("watchlist_flag"):

        alerts.append(
            "Identity matched a watchlist record"
        )

    if identity.get("fraud_flag"):

        alerts.append(
            "Identity has previous fraud activity"
        )

    suspicious_screenings = [

        screening

        for screening in screenings

        if (screening.get("risk_score") or 0) >= 50

    ]

    if suspicious_screenings:

        alerts.append(
            f"{len(suspicious_screenings)} "
            f"suspicious screening(s) found"
        )

    if fraud_cases:

        alerts.append(
            f"{len(fraud_cases)} "
            f"fraud case(s) linked to this identity"
        )

    # ========================================================
    # CALCULATE RISK LEVEL
    # ========================================================

    if identity.get("watchlist_flag"):

        intelligence_risk = "HIGH"

    elif identity.get("fraud_flag"):

        intelligence_risk = "SUSPICIOUS"

    elif suspicious_screenings:

        intelligence_risk = "SUSPICIOUS"

    elif len(screenings) > 1:

        intelligence_risk = "REVIEW"

    else:

        intelligence_risk = "LOW"

    return {

        "identity": identity,

        "screenings": screenings,

        "fraud_cases": fraud_cases,

        "alerts": alerts,

        "risk_level": intelligence_risk,

        "total_screenings": len(screenings)

    }


# ============================================================
# ADD AUDIT LOG
# ============================================================

def add_audit_log(
    action,
    details,
    officer_name="BorderShield AI"
):

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("""
            INSERT INTO audit_logs (

                officer_name,
                action,
                details,
                created_at

            )

            VALUES (%s, %s, %s, %s)

            RETURNING id

        """, (

            officer_name,

            action,

            details,

            datetime.now().isoformat()

        ))

        audit_id = cursor.fetchone()["id"]

        conn.commit()

        return audit_id

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# GET DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # ====================================================
        # TOTAL SCREENINGS
        # ====================================================

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM screenings
        """)

        total_screenings = cursor.fetchone()["total"]

        # ====================================================
        # AVERAGE RISK
        # ====================================================

        cursor.execute("""
            SELECT AVG(risk_score) AS average
            FROM screenings
        """)

        average = cursor.fetchone()["average"]

        # ====================================================
        # CLEARED
        # ====================================================

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'LOW RISK'
        """)

        cleared = cursor.fetchone()["total"]

        # ====================================================
        # FLAGGED
        # ====================================================

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status != 'LOW RISK'
        """)

        flagged = cursor.fetchone()["total"]

        # ====================================================
        # RECENT SCREENINGS
        # ====================================================

        cursor.execute("""
            SELECT *
            FROM screenings
            ORDER BY id DESC
            LIMIT 10
        """)

        recent_screenings = [

            dict(row)

            for row in cursor.fetchall()

        ]

    finally:

        conn.close()

    return {

        "success": True,

        "statistics": {

            "screenings_today":
                total_screenings,

            "average_risk_score":
                round(float(average), 1)
                if average is not None
                else 0,

            "cleared_documents":
                cleared,

            "flagged_for_review":
                flagged

        },

        "recent_screenings":
            recent_screenings

    }


# ============================================================
# GET AUDIT LOGS
# ============================================================

def get_audit_logs(limit=100):

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("""
            SELECT *
            FROM audit_logs
            ORDER BY id DESC
            LIMIT %s
        """, (
            limit,
        ))

        logs = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return logs

    finally:

        conn.close()


# ============================================================
# GET ALL SCREENINGS
# ============================================================

def get_all_screenings(limit=100):

    conn = get_connection()

    try:

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("""
            SELECT *
            FROM screenings
            ORDER BY id DESC
            LIMIT %s
        """, (
            limit,
        ))

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:

        conn.close()


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database_connection():

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
            SELECT 1
        """)

        result = cursor.fetchone()

        return result[0] == 1

    except Exception as error:

        print(
            "DATABASE CONNECTION ERROR:",
            error
        )

        return False

    finally:

        if conn:
            conn.close()
