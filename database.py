import sqlite3
import json
from datetime import datetime


DB_NAME = "bordershield.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    conn = sqlite3.connect(DB_NAME)

    conn.row_factory = sqlite3.Row

    return conn


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

        id INTEGER PRIMARY KEY AUTOINCREMENT,

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

        id INTEGER PRIMARY KEY AUTOINCREMENT,

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

        id INTEGER PRIMARY KEY AUTOINCREMENT,

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

        id INTEGER PRIMARY KEY AUTOINCREMENT,

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

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        officer_name TEXT,

        action TEXT,

        details TEXT,

        created_at TEXT

    )
    """)


    conn.commit()

    conn.close()

    print("BorderShield database initialized successfully.")


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


    # Check existing identity

    cursor.execute("""

        SELECT id

        FROM identities

        WHERE document_number = ?

    """, (document_number,))


    existing = cursor.fetchone()


    if existing:

        identity_id = existing["id"]

        conn.close()

        return identity_id


    # Create new identity

    created_at = datetime.now().isoformat()


    cursor.execute("""

        INSERT INTO identities (

            full_name,

            document_number,

            document_type,

            created_at

        )

        VALUES (?, ?, ?, ?)

    """, (

        full_name or "Unknown",

        document_number,

        document_type or "Unknown",

        created_at

    ))


    identity_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return identity_id


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


    created_at = datetime.now().isoformat()


    issues_json = json.dumps(issues)


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

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        identity_id,

        full_name or "Unknown",

        document_number or "Not detected",

        document_type,

        risk_score,

        status,

        issues_json,

        checkpoint,

        officer_name,

        created_at

    ))


    screening_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return screening_id


# ============================================================
# GET IDENTITY INTELLIGENCE
# ============================================================

def get_identity_intelligence(identity_id):


    if not identity_id:

        return None


    conn = get_connection()

    cursor = conn.cursor()


    # Get identity using ID

    cursor.execute("""

        SELECT *

        FROM identities

        WHERE id = ?

    """, (identity_id,))


    identity = cursor.fetchone()


    if not identity:

        conn.close()

        return None


    identity = dict(identity)


    document_number = identity["document_number"]


    # ========================================================
    # GET SCREENING HISTORY
    # ========================================================

    cursor.execute("""

        SELECT *

        FROM screenings

        WHERE identity_id = ?

        ORDER BY created_at DESC

    """, (identity_id,))


    screenings = [

        dict(row)

        for row in cursor.fetchall()

    ]


    # ========================================================
    # GET FRAUD CASES
    # ========================================================

    cursor.execute("""

        SELECT *

        FROM fraud_cases

        WHERE identity_id = ?

        ORDER BY created_at DESC

    """, (identity_id,))


    fraud_cases = [

        dict(row)

        for row in cursor.fetchall()

    ]


    conn.close()


    # ========================================================
    # CALCULATE ALERTS
    # ========================================================

    alerts = []


    if identity["watchlist_flag"]:

        alerts.append(
            "Identity matched a watchlist record"
        )


    if identity["fraud_flag"]:

        alerts.append(
            "Identity has previous fraud activity"
        )


    suspicious_screenings = [

        screening

        for screening in screenings

        if screening["risk_score"] >= 50

    ]


    if suspicious_screenings:

        alerts.append(

            f"{len(suspicious_screenings)} suspicious screening(s) found"

        )


    if fraud_cases:

        alerts.append(

            f"{len(fraud_cases)} fraud case(s) linked to this identity"

        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if identity["watchlist_flag"]:

        intelligence_risk = "HIGH"


    elif identity["fraud_flag"]:

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

    cursor = conn.cursor()


    cursor.execute("""

        INSERT INTO audit_logs (

            officer_name,

            action,

            details,

            created_at

        )

        VALUES (?, ?, ?, ?)

    """, (

        officer_name,

        action,

        details,

        datetime.now().isoformat()

    ))


    conn.commit()

    conn.close()


# ============================================================
# GET DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    conn = get_connection()

    cursor = conn.cursor()


    # Total screenings

    cursor.execute("""

        SELECT COUNT(*) AS total

        FROM screenings

    """)


    total_screenings = cursor.fetchone()["total"]


    # Average risk

    cursor.execute("""

        SELECT AVG(risk_score) AS average

        FROM screenings

    """)


    average = cursor.fetchone()["average"]


    # Cleared

    cursor.execute("""

        SELECT COUNT(*) AS total

        FROM screenings

        WHERE status = 'LOW RISK'

    """)


    cleared = cursor.fetchone()["total"]


    # Flagged

    cursor.execute("""

        SELECT COUNT(*) AS total

        FROM screenings

        WHERE status != 'LOW RISK'

    """)


    flagged = cursor.fetchone()["total"]


    # Recent screenings

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


    conn.close()


    return {

        "success": True,

        "statistics": {

            "screenings_today": total_screenings,

            "average_risk_score": round(average, 1) if average else 0,

            "cleared_documents": cleared,

            "flagged_for_review": flagged

        },

        "recent_screenings": recent_screenings

    }