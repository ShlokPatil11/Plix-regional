import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import fields
from models.job import Job, STATUSES

class Database:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY, job_id TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL, original_path TEXT NOT NULL,
                    processing_path TEXT NOT NULL DEFAULT '', output_path TEXT,
                    file_hash TEXT NOT NULL, source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL, status TEXT NOT NULL,
                    current_stage TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT, completed_at TEXT, retry_count INTEGER DEFAULT 0,
                    error_message TEXT, metadata_json TEXT DEFAULT '{}');
                CREATE UNIQUE INDEX IF NOT EXISTS unique_live_hash ON jobs(file_hash, target_language)
                    WHERE status != 'FAILED';
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def create(self, path, digest, source, target, force=False):
        job_id = uuid.uuid4().hex
        try:
            with self.connect() as db:
                if force:
                    db.execute("UPDATE jobs SET status='FAILED' WHERE file_hash=? AND target_language=? AND status!='FAILED'", (digest, target))
                db.execute('INSERT INTO jobs(job_id,filename,original_path,file_hash,source_language,target_language,status,current_stage) VALUES(?,?,?,?,?,?,?,?)',
                           (job_id,path.name,str(path),digest,source,target,'DETECTED','DETECTED'))
        except sqlite3.IntegrityError:
            return None
        return job_id

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE job_id=?',(job_id,)).fetchone()
        if row is None: raise ValueError(f'Unknown job: {job_id}')
        return Job(**{f.name:row[f.name] for f in fields(Job)})

    def update(self, job_id, status=None, **values):
        allowed = {'processing_path','output_path','retry_count','error_message','metadata_json','current_stage','source_language'}
        if not values.keys() <= allowed: raise ValueError('Invalid update fields')
        if status:
            if status not in STATUSES: raise ValueError('Invalid job status')
            values.update(status=status,current_stage=status)
        if not values: return
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(f'{key}=?' for key in values)+' WHERE job_id=?',(*values.values(),job_id))
            if status in ('COMPLETED','FAILED'):
                db.execute('UPDATE jobs SET completed_at=CURRENT_TIMESTAMP WHERE job_id=?',(job_id,))

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT job_id FROM jobs WHERE status='QUEUED' ORDER BY id LIMIT 1").fetchone()
            if row is None: return None
            db.execute("UPDATE jobs SET status='ANALYZING',current_stage='ANALYZING',started_at=CURRENT_TIMESTAMP WHERE job_id=?",(row[0],))
        return self.get(row[0])

    def counts(self):
        with self.connect() as db:
            return dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())

    def unfinished(self):
        with self.connect() as db:
            return [r[0] for r in db.execute("SELECT job_id FROM jobs WHERE status NOT IN ('COMPLETED','FAILED','QUEUED')")]

    def reserve_retry(self, job_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if not row or row[0] != 'FAILED': raise ValueError('Only FAILED jobs can be retried')
            db.execute("UPDATE jobs SET status='DETECTED',current_stage='RETRY',retry_count=0,error_message=NULL,completed_at=NULL WHERE job_id=?",(job_id,))
