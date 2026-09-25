"""
============================================================
BorderShield AI / TeamTrace
PostgreSQL Database Layer
============================================================

This file replaces the old SQLite database.py.

Required Vercel Environment Variable:
DATABASE_URL = your Neon PostgreSQL connection string

Do NOT put DATABASE_URL inside this file.
Do NOT put DATABASE_URL on GitHub.
============================================================
"""

import os
import json
from datetime import datetime

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set."
        )

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    # ========================================================
    # IDENTITIES TABLE
    # ========================================================

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

    # ========================================================
    # SCREENINGS TABLE
    # ========================================================

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

    # ========================================================
    # FRAUD CASES TABLE
    # ========================================================

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

    # ========================================================
    # CHECKPOINTS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (

            id SERIAL PRIMARY KEY,

            checkpoint_name TEXT,

            location TEXT,

            status TEXT DEFAULT 'ACTIVE',

            last_activity TEXT

        )
    """)

    # ========================================================
    # AUDIT LOGS TABLE
    # ========================================================

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

    cursor.close()
    conn.close()

    print("BorderShield PostgreSQL database initialized successfully.")


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
    cursor = conn.cursor()

    try:

        # ----------------------------------------------------
        # CHECK EXISTING IDENTITY
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM identities
            WHERE document_number = %s
            """,
            (document_number,)
        )

        existing = cursor.fetchone()

        if existing:

            identity_id = existing["id"]

            return identity_id

        # ----------------------------------------------------
        # CREATE NEW IDENTITY
        # ----------------------------------------------------

        created_at = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO identities (
                full_name,
                document_number,
                document_type,
                created_at
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                full_name or "Unknown",
                document_number,
                document_type or "Unknown",
                created_at
            )
        )

        result = cursor.fetchone()

        identity_id = result["id"]

        conn.commit()

        return identity_id

    except Exception:

        conn.rollback()
        raise

    finally:

        cursor.close()
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
    cursor = conn.cursor()

    try:

        created_at = datetime.now().isoformat()

        issues_json = json.dumps(issues)

        cursor.execute(
            """
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
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                identity_id,
                full_name or "Unknown",
                document_number or "Not detected",
                document_type or "Unknown",
                int(risk_score or 0),
                status or "LOW RISK",
                issues_json,
                checkpoint,
                officer_name,
                created_at
            )
        )

        result = cursor.fetchone()

        screening_id = result["id"]

        conn.commit()

        return screening_id

    except Exception:

        conn.rollback()
        raise

    finally:

        cursor.close()
        conn.close()


# ============================================================
# GET IDENTITY INTELLIGENCE
# ============================================================

def get_identity_intelligence(identity_id):

    if not identity_id:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # ----------------------------------------------------
        # GET IDENTITY
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT *
            FROM identities
            WHERE id = %s
            """,
            (identity_id,)
        )

        identity = cursor.fetchone()

        if not identity:
            return None

        identity = dict(identity)

        # ----------------------------------------------------
        # GET SCREENING HISTORY
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT *
            FROM screenings
            WHERE identity_id = %s
            ORDER BY created_at DESC
            """,
            (identity_id,)
        )

        screenings = [
            dict(row)
            for row in cursor.fetchall()
        ]

        # ----------------------------------------------------
        # GET FRAUD CASES
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT *
            FROM fraud_cases
            WHERE identity_id = %s
            ORDER BY created_at DESC
            """,
            (identity_id,)
        )

        fraud_cases = [
            dict(row)
            for row in cursor.fetchall()
        ]

        # ----------------------------------------------------
        # CALCULATE ALERTS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DETERMINE RISK LEVEL
        # ----------------------------------------------------

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

    finally:

        cursor.close()
        conn.close()


# ============================================================
# ADD AUDIT LOG
# ============================================================

def add_audit_log(
    action,
    details,
    officer_name="BorderShield AI"
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO audit_logs (
                officer_name,
                action,
                details,
                created_at
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                officer_name,
                action,
                details,
                datetime.now().isoformat()
            )
        )

        result = cursor.fetchone()

        audit_id = result["id"]

        conn.commit()

        return audit_id

    except Exception:

        conn.rollback()
        raise

    finally:

        cursor.close()
        conn.close()


# ============================================================
# GET AUDIT LOGS
# ============================================================

def get_audit_logs(limit=100):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM audit_logs
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )

        logs = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return logs

    finally:

        cursor.close()
        conn.close()


# ============================================================
# GET ALL SCREENINGS
# ============================================================

def get_all_screenings(limit=100):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM screenings
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )

        screenings = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return screenings

    finally:

        cursor.close()
        conn.close()


# ============================================================
# GET DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # ----------------------------------------------------
        # TOTAL SCREENINGS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            """
        )

        row = cursor.fetchone()

        total_screenings = int(row["total"] or 0)

        # ----------------------------------------------------
        # AVERAGE RISK
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT AVG(risk_score) AS average
            FROM screenings
            """
        )

        row = cursor.fetchone()

        average = row["average"]

        if average is not None:
            average = float(average)
        else:
            average = 0.0

        # ----------------------------------------------------
        # CLEARED
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'LOW RISK'
            """
        )

        row = cursor.fetchone()

        cleared = int(row["total"] or 0)

        # ----------------------------------------------------
        # FLAGGED
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status != 'LOW RISK'
            """
        )

        row = cursor.fetchone()

        flagged = int(row["total"] or 0)

        # ----------------------------------------------------
        # RECENT SCREENINGS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT *
            FROM screenings
            ORDER BY id DESC
            LIMIT 10
            """
        )

        recent_screenings = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return {

            "success": True,

            "statistics": {

                "screenings_today": total_screenings,

                "average_risk_score": round(
                    average,
                    1
                ),

                "cleared_documents": cleared,

                "flagged_for_review": flagged

            },

            "recent_screenings": recent_screenings

        }

    finally:

        cursor.close()
        conn.close()


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database_connection():

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 AS test"
        )

        result = cursor.fetchone()

        if result and result["test"] == 1:

            return {

                "success": True,

                "database": "PostgreSQL",

                "connected": True

            }

        return {

            "success": False,

            "database": "PostgreSQL",

            "connected": False,

            "error": "Database test query failed"

        }

    except Exception as e:

        return {

            "success": False,

            "database": "PostgreSQL",

            "connected": False,

            "error": str(e)

        }

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    try:

        initialize_database()

        result = check_database_connection()

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:

        print(
            json.dumps(
                {
                    "success": False,
                    "database": "PostgreSQL",
                    "connected": False,
                    "error": str(e)
                },
                indent=2
            )
        )
