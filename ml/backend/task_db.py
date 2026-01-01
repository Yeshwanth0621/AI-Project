"""
Task Database Layer
SQLite database for storing task assignments and worker contact info.
Works fully offline on LAN.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json

# Database path
DB_PATH = Path(__file__).parent / "tasks.db"


def get_connection():
    """Get database connection with row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL,
            worker_name TEXT NOT NULL,
            job_title TEXT NOT NULL,
            job_description TEXT,
            location TEXT,
            start_date TEXT,
            duration TEXT,
            status TEXT DEFAULT 'pending',
            assigned_by TEXT,
            requester_name TEXT,
            requester_email TEXT,
            requester_phone TEXT,
            special_requirements TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Worker contact info table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS worker_contacts (
            worker_id INTEGER PRIMARY KEY,
            email TEXT,
            phone TEXT,
            push_subscription TEXT,
            notification_enabled INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Notification log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            worker_id INTEGER,
            notification_type TEXT,
            status TEXT,
            message TEXT,
            sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")


def create_task(
    worker_id: int,
    worker_name: str,
    job_title: str,
    job_description: str = None,
    location: str = None,
    start_date: str = None,
    duration: str = None,
    assigned_by: str = "admin",
    requester_name: str = None,
    requester_email: str = None,
    requester_phone: str = None,
    special_requirements: str = None
) -> Dict:
    """Create a new task assignment."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tasks (
            worker_id, worker_name, job_title, job_description,
            location, start_date, duration, assigned_by,
            requester_name, requester_email, requester_phone, special_requirements
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        worker_id, worker_name, job_title, job_description,
        location, start_date, duration, assigned_by,
        requester_name, requester_email, requester_phone, special_requirements
    ))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": task_id,
        "worker_id": worker_id,
        "worker_name": worker_name,
        "job_title": job_title,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }


def get_task(task_id: int) -> Optional[Dict]:
    """Get a single task by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_worker_tasks(worker_id: int, status: str = None) -> List[Dict]:
    """Get all tasks for a worker, optionally filtered by status."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if status:
        cursor.execute(
            'SELECT * FROM tasks WHERE worker_id = ? AND status = ? ORDER BY created_at DESC',
            (worker_id, status)
        )
    else:
        cursor.execute(
            'SELECT * FROM tasks WHERE worker_id = ? ORDER BY created_at DESC',
            (worker_id,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_all_tasks(status: str = None, limit: int = 50) -> List[Dict]:
    """Get all tasks, optionally filtered by status."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if status:
        cursor.execute(
            'SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?',
            (status, limit)
        )
    else:
        cursor.execute(
            'SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def update_task_status(task_id: int, status: str) -> bool:
    """Update task status (pending, accepted, in_progress, completed, declined)."""
    valid_statuses = ['pending', 'accepted', 'in_progress', 'completed', 'declined']
    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE tasks 
        SET status = ?, updated_at = ? 
        WHERE id = ?
    ''', (status, datetime.now().isoformat(), task_id))
    
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    
    return affected > 0


def get_worker_contact(worker_id: int) -> Optional[Dict]:
    """Get worker contact info."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM worker_contacts WHERE worker_id = ?', (worker_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        # Parse push subscription JSON if present
        if result.get('push_subscription'):
            try:
                result['push_subscription'] = json.loads(result['push_subscription'])
            except:
                pass
        return result
    return None


def upsert_worker_contact(
    worker_id: int,
    email: str = None,
    phone: str = None,
    push_subscription: dict = None,
    notification_enabled: bool = True
) -> Dict:
    """Create or update worker contact info."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Serialize push subscription
    push_sub_json = json.dumps(push_subscription) if push_subscription else None
    
    cursor.execute('''
        INSERT INTO worker_contacts (worker_id, email, phone, push_subscription, notification_enabled, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(worker_id) DO UPDATE SET
            email = COALESCE(excluded.email, email),
            phone = COALESCE(excluded.phone, phone),
            push_subscription = COALESCE(excluded.push_subscription, push_subscription),
            notification_enabled = excluded.notification_enabled,
            updated_at = excluded.updated_at
    ''', (worker_id, email, phone, push_sub_json, 1 if notification_enabled else 0, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    
    return get_worker_contact(worker_id)


def log_notification(
    task_id: int,
    worker_id: int,
    notification_type: str,
    status: str,
    message: str = None
):
    """Log notification attempt for auditing."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO notification_log (task_id, worker_id, notification_type, status, message)
        VALUES (?, ?, ?, ?, ?)
    ''', (task_id, worker_id, notification_type, status, message))
    
    conn.commit()
    conn.close()


def get_task_stats() -> Dict:
    """Get task statistics for dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            status,
            COUNT(*) as count
        FROM tasks
        GROUP BY status
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    stats = {
        'total': 0,
        'pending': 0,
        'accepted': 0,
        'in_progress': 0,
        'completed': 0,
        'declined': 0
    }
    
    for row in rows:
        stats[row['status']] = row['count']
        stats['total'] += row['count']
    
    return stats


# Initialize database on import
if __name__ == "__main__":
    init_db()
    print("Database tables created!")
    
    # Test creating a task
    task = create_task(
        worker_id=15,
        worker_name="Alexandria Smith",
        job_title="Electrical Maintenance",
        job_description="High voltage panel inspection",
        location="Site A, Building 3",
        start_date="2026-01-05",
        duration="2 days",
        assigned_by="admin",
        requester_name="John Doe",
        requester_email="john@company.com"
    )
    print(f"Created test task: {task}")
