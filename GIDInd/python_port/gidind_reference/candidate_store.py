"""Lossless coordinator-owned candidate storage; RAM use is independent of row count.

SQLite's page cache and disk sorting replace growing Python object collections.
Insertion slots preserve traversal/tie order; replacements retain their slot.
Only committed transactions are promised recoverable after forced process death.
"""
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np


def score(candidate):
    return candidate.native_scores.get('seed_linear_sumerr',
                                       candidate.native_scores.get('seed_qxyz_rmse', 0.))


def assignment_key(candidate):
    h = np.asarray(candidate.hkl, dtype='<i4').tobytes()
    raw = (b'\0'+h if candidate.plane is None else
           b'\1'+np.asarray(candidate.plane, dtype='<i4').tobytes()+h)
    # HKL arrays contain many small repeated integers. Compress the exact key
    # as well as its unique index; never quantize metrics or native scores.
    return zlib.compress(raw, 1)


def decode(row):
    from .state import Candidate
    key, metric, projection, scores, timing = row
    key = zlib.decompress(key)
    plane = np.frombuffer(key[1:13], dtype='<i4').astype(int) if key[0] else None
    h = np.frombuffer(key[13:] if key[0] else key[1:], dtype='<i4').reshape(-1, 3).astype(int)
    return Candidate(np.frombuffer(metric, dtype='<f8').reshape(3, 3).copy(), h,
                     plane, None if projection is None else
                     np.frombuffer(projection, dtype='<f8').copy(),
                     json.loads(scores), json.loads(timing))


class CandidateView:
    def __init__(self, store, ceiling=None):
        self.store, self.ceiling = store, ceiling

    def __len__(self):
        if self.ceiling is None:
            return len(self.store)
        return self.store.db.execute('SELECT count(*) FROM native WHERE score<=?',
                                     (self.ceiling,)).fetchone()[0]

    def __iter__(self):
        yield from self.store.iter_rows(ordered=True, ceiling=self.ceiling)


class CandidateStore:
    columns = 'assignment,metric,projection,scores,timing'

    def __init__(self, path, *, create=True):
        self.path = Path(path)
        if create and self.path.exists():
            raise FileExistsError(f'Refusing to reuse candidate database: {self.path}')
        if create:
            self.db = sqlite3.connect(self.path)
        else:
            # Read/write opening permits SQLite to roll back a crash's hot
            # journal. mode=rw refuses missing files; it never creates one.
            self.db = sqlite3.connect(self.path.resolve().as_uri()+'?mode=rw', uri=True)
        self.db.execute('PRAGMA cache_size=-8192')  # 8 MiB; not a row-count limit.
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA temp_store=FILE')
        if create:
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE native(slot INTEGER PRIMARY KEY, assignment BLOB UNIQUE NOT NULL,
                  score REAL NOT NULL, metric BLOB NOT NULL, projection BLOB,
                  scores TEXT NOT NULL, timing TEXT NOT NULL);
                CREATE INDEX native_score ON native(score,slot);
                CREATE TABLE abg(slot INTEGER PRIMARY KEY, a REAL,b REAL,g REAL,u INTEGER,v INTEGER,
                  area REAL,error REAL,choices BLOB,n INTEGER,r INTEGER);
                CREATE INDEX abg_area ON abg(area,error,a,b,g,u,v,slot);
                CREATE INDEX abg_error ON abg(error,a,b,g,u,v,slot);
                PRAGMA user_version=1;
            ''')
        self.size = self.db.execute('SELECT count(*) FROM native').fetchone()[0]
        self.writes = 0
        self.final_order = False
        self.abgs = ABGStore(self)

    def __len__(self):
        return self.size

    def lookup(self, key):
        return self.db.execute('SELECT slot,score,timing FROM native WHERE assignment=?',
                               (key,)).fetchone()

    def put(self, slot, key, candidate):
        self.db.execute('INSERT OR REPLACE INTO native VALUES(?,?,?,?,?,?,?)',
            (slot, key, float(score(candidate)),
             np.asarray(candidate.metric, dtype='<f8').tobytes(),
             None if candidate.projection is None else
             np.asarray(candidate.projection, dtype='<f8').tobytes(),
             json.dumps(candidate.native_scores, allow_nan=False),
             json.dumps(candidate.timing, allow_nan=False)))
        if slot == self.size:
            self.size += 1
        self.changed()

    def changed(self):
        self.writes += 1
        if self.writes >= 2048:
            self.flush()

    def flush(self):
        self.db.commit()
        self.writes = 0

    def close(self):
        try:
            self.flush()
        finally:
            self.db.close()

    def iter_rows(self, *, ordered=False, ceiling=None):
        query = f'SELECT {self.columns} FROM native'
        params = ()
        if ceiling is not None:
            query += ' WHERE score<=?'; params = (ceiling,)
        query += ' ORDER BY score,slot' if ordered else ' ORDER BY slot'
        cursor = self.db.execute(query, params)
        try:
            for row in cursor:
                yield decode(row)
        finally:
            cursor.close()

    def __iter__(self):
        yield from self.iter_rows(ordered=self.final_order)

    def __getitem__(self, slot):
        if slot < 0:
            slot += self.size
        row = self.db.execute(f'SELECT {self.columns} FROM native WHERE slot=?', (slot,)).fetchone()
        if row is None:
            raise IndexError(slot)
        return decode(row)

    def view(self, ceiling=None):
        return CandidateView(self, ceiling)

    def minimum(self):
        return self.db.execute('SELECT score FROM native ORDER BY score,slot LIMIT 1').fetchone()[0]

    def metadata(self):
        return dict(backend='sqlite',path=self.path.name,schema_version=1,
                    sqlite_version=sqlite3.sqlite_version,
                    native_candidates=self.size,abg_candidates=len(self.abgs),
                    page_cache_mib=8,commit_every_writes=2048,
                    database_bytes=self.path.stat().st_size,
                    policy='Lossless storage; no top-N eviction. Stable score ordering with insertion-slot ties.')


class ABGStore:
    def __init__(self, store):
        self.store = store
        self.size = store.db.execute('SELECT count(*) FROM abg').fetchone()[0]

    def __len__(self):
        return self.size

    def append(self, row):
        abg, uv, choices, error = row
        a,b,g = (float(x) for x in abg)
        self.store.db.execute('INSERT INTO abg VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (self.size,a,b,g,int(uv[0]),int(uv[1]),float(a*b*np.sin(g)),float(error),
             zlib.compress(np.asarray(choices,dtype='<i8').tobytes(),1),choices.shape[0],choices.shape[1]))
        self.size += 1
        self.store.changed()

    def ordered(self, ordering):
        primary = 'area' if ordering == 'area' else 'error'
        cursor = self.store.db.execute(
            f'SELECT a,b,g,u,v,choices,n,r,error FROM abg ORDER BY {primary},error,a,b,g,u,v,slot')
        try:
            for a,b,g,u,v,choices,n,r,error in cursor:
                yield (np.array([a,b,g]),(u,v),
                       np.frombuffer(zlib.decompress(choices),dtype='<i8').reshape(n,r,2).copy(),error)
        finally:
            cursor.close()
