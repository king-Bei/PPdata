"""Validation and atomic, process-safe storage for local API credentials."""

from contextlib import contextmanager
from datetime import date, datetime
import fcntl
import json
import os
from pathlib import Path
import secrets
import tempfile
from zoneinfo import ZoneInfo


class KeyStorageError(OSError):
    """Storage is unavailable or malformed; never silently discard saved keys."""


def validate_key_record(payload):
    """Validate editable fields and normalize ISO dates; credentials are server-owned."""
    result = {}
    for field in ('source_name', 'app_name'):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} 為必填文字欄位')
        result[field] = value.strip()
    result['key_type'] = payload.get('key_type', 'permanent')
    if result['key_type'] not in ('permanent', 'periodic'):
        raise ValueError('key_type 必須為 permanent 或 periodic')
    result['status'] = payload.get('status', 'active')
    if result['status'] not in ('active', 'disabled'):
        raise ValueError('status 僅能為 active 或 disabled')
    for field in ('start_date', 'end_date'):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{field} 必須為 YYYY-MM-DD 日期')
        value = value.strip() if value else None
        if value:
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError:
                raise ValueError(f'{field} 必須為有效的 YYYY-MM-DD 日期') from None
        result[field] = value or None
    if result['key_type'] == 'periodic':
        if not result['start_date'] or not result['end_date']:
            raise ValueError('週期金鑰需填寫開始與結束日期')
        if result['start_date'] > result['end_date']:
            raise ValueError('結束日期不可早於開始日期')
    else:
        result.update(start_date=None, end_date=None)
    return result


def key_is_usable(record, today=None):
    """A periodic key includes both boundary days, using the Asia/Taipei calendar."""
    if record.get('status') != 'active':
        return False
    if record.get('key_type') == 'permanent':
        return True
    if record.get('key_type') != 'periodic':
        return False
    try:
        start = date.fromisoformat(record['start_date'])
        end = date.fromisoformat(record['end_date'])
    except (ValueError, TypeError, KeyError):
        return False
    today = today or datetime.now(ZoneInfo('Asia/Taipei')).date()
    return start <= today <= end


class KeyStore:
    """Serialize mutations via a lock file and replace JSON atomically on the same disk."""

    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def locked(self, exclusive=False):
        """Use one stable lock inode across atomic data-file replacements."""
        with open(str(self.path) + '.lock', 'a', encoding='utf-8') as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self):
        """Reject invalid JSON/credential records without replacing existing data."""
        if not self.path.exists():
            return []
        try:
            records = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(records, list):
                raise ValueError
            ids, credentials = set(), set()
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError
                for field in ('id', 'api_key'):
                    if not isinstance(record.get(field), str) or not record[field]:
                        raise ValueError
                if record['id'] in ids or record['api_key'] in credentials:
                    raise ValueError
                ids.add(record['id'])
                credentials.add(record['api_key'])
            return records
        except (ValueError, UnicodeError) as exc:
            raise KeyStorageError('Invalid API key storage') from exc

    def read(self):
        """Read a consistent snapshot while cooperating with writers."""
        with self.locked():
            return self._read()

    @staticmethod
    def new_key(records):
        """Generate a 256-bit credential that differs from all existing keys."""
        existing = {record['api_key'] for record in records}
        while True:
            candidate = secrets.token_urlsafe(32)
            if candidate not in existing:
                return candidate

    def mutate(self, operation):
        """Run a mutation under lock; a failed write leaves the original file intact."""
        with self.locked(exclusive=True):
            records = self._read()
            result = operation(records)
            if result is None:
                return None
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                 dir=self.path.parent, delete=False) as file:
                    temporary = file.name
                    json.dump(records, file, ensure_ascii=False, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            return result
