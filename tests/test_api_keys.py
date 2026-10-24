"""Exercise key lifecycle, access enforcement, validation and durable writes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app
from api_keys import KeyStore, key_is_usable


class KeyManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'keys.json'
        self.storage = patch.object(app, 'APP_KEYS_FILE', str(self.path))
        self.storage.start()
        self.addCleanup(self.storage.stop)
        self.admin = self.admin_client()
        self.public = app.app.test_client()

    def admin_client(self):
        """Use a separate signed session for each management client."""
        client = app.app.test_client()
        with client.session_transaction() as session:
            session['user'] = 'admin'
        return client

    def create(self, **values):
        """Create a key through the same API used by the management page."""
        payload = dict(source_name='測試來源', app_name='測試 App', key_type='permanent')
        payload.update(values)
        return self.admin.post('/api/app-keys', json=payload)

    def test_create_auto_generates_and_persists(self):
        response = self.create(api_key='client-chosen-credential')
        self.assertEqual(response.status_code, 201)
        key = response.json['data']
        self.assertNotEqual(key['api_key'], 'client-chosen-credential')
        self.assertGreaterEqual(len(key['api_key']), 43)
        self.assertEqual(KeyStore(self.path).read(), [key])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_management_requires_login_even_with_api_key(self):
        key = self.create().json['data']
        for path, method in [('/api/app-keys', 'get'), ('/api/app-keys', 'post'),
                             (f"/api/app-keys/{key['id']}", 'put'),
                             (f"/api/app-keys/{key['id']}/regenerate", 'post')]:
            with self.subTest(path=path, method=method):
                response = getattr(self.public, method)(path, json={}, headers={'X-API-Key': key['api_key']})
                self.assertEqual(response.status_code, 401)

    def test_ocr_requires_key_for_both_endpoints(self):
        for path in ['/api/ocr', '/api/ocr-csv']:
            for headers in [{}, {'X-API-Key': 'invalid'}]:
                self.assertEqual(self.public.post(path, headers=headers).status_code, 401)

    def test_valid_key_reaches_image_validation(self):
        key = self.create().json['data']
        for path in ['/api/ocr', '/api/ocr-csv']:
            response = self.public.post(path, headers={'X-API-Key': key['api_key']})
            self.assertEqual(response.status_code, 400)

    def test_disable_enable_and_rotate_invalidate_old_key(self):
        key = self.create().json['data']
        url = f"/api/app-keys/{key['id']}"
        self.assertEqual(self.admin.put(url, json={'status': 'disabled'}).status_code, 200)
        self.assertEqual(self.public.post('/api/ocr', headers={'X-API-Key': key['api_key']}).status_code, 403)
        rotated = self.admin.post(url + '/regenerate').json['data']
        self.assertNotEqual(key['api_key'], rotated['api_key'])
        self.assertEqual(rotated['status'], 'disabled')
        self.assertEqual(self.public.post('/api/ocr', headers={'X-API-Key': key['api_key']}).status_code, 401)
        self.assertEqual(self.public.post('/api/ocr', headers={'X-API-Key': rotated['api_key']}).status_code, 403)
        self.admin.put(url, json={'status': 'active'})
        self.assertEqual(self.public.post('/api/ocr', headers={'X-API-Key': rotated['api_key']}).status_code, 400)

    def test_expired_and_future_keys_are_forbidden(self):
        for start, end in [('2000-01-01','2000-01-02'), ('2999-01-01','2999-01-02')]:
            key = self.create(key_type='periodic', start_date=start, end_date=end).json['data']
            self.assertEqual(self.public.post('/api/ocr', headers={'X-API-Key': key['api_key']}).status_code, 403)

    def test_period_includes_both_dates_and_uses_taipei_calendar(self):
        key = dict(status='active', key_type='periodic', start_date='2026-10-25', end_date='2026-10-26')
        self.assertTrue(key_is_usable(key, date(2026,10,25)))
        self.assertTrue(key_is_usable(key, date(2026,10,26)))
        self.assertFalse(key_is_usable(key, date(2026,10,24)))
        with patch('api_keys.datetime') as clock:
            clock.now.return_value.date.return_value = date(2026,10,25)
            self.assertTrue(key_is_usable(key))
            self.assertEqual(str(clock.now.call_args.args[0]), 'Asia/Taipei')

    def test_invalid_metadata_is_rejected_without_writing(self):
        for fields in [dict(source_name='  '), dict(app_name=123), dict(key_type='x'),
                       dict(status='x'), dict(start_date=[]),
                       dict(key_type='periodic'),
                       dict(key_type='periodic',start_date='2026-02-30',end_date='2026-03-01'),
                       dict(key_type='periodic',start_date='2026-10-26',end_date='2026-10-25')]:
            with self.subTest(fields=fields):
                self.assertEqual(self.create(**fields).status_code, 400)
        self.assertFalse(self.path.exists())

    def test_invalid_json_and_unknown_ids(self):
        self.assertEqual(self.admin.post('/api/app-keys',json=[]).status_code,400)
        self.assertEqual(self.admin.put('/api/app-keys/missing',json={}).status_code,404)
        self.assertEqual(self.admin.post('/api/app-keys/missing/regenerate').status_code,404)

    def test_date_only_update_is_validated_and_does_not_change_saved_record(self):
        key = self.create(key_type='periodic', start_date='2026-10-25', end_date='2026-10-26').json['data']
        before = self.path.read_bytes()
        response = self.admin.put(f"/api/app-keys/{key['id']}",json={'end_date':'2026-10-24'})
        self.assertEqual(response.status_code,400)
        self.assertEqual(before,self.path.read_bytes())

    def test_cannot_replace_key_via_metadata_update(self):
        key = self.create().json['data']
        response = self.admin.put(f"/api/app-keys/{key['id']}",json={'api_key':'chosen'})
        self.assertEqual(response.status_code,400)
        self.assertEqual(KeyStore(self.path).read()[0]['api_key'],key['api_key'])

    def test_cross_origin_mutation_is_forbidden(self):
        self.assertEqual(self.admin.post('/api/app-keys',json={},headers={'Origin':'https://other.example'}).status_code,403)

    def test_same_host_https_origin_behind_tls_proxy(self):
        response = self.admin.post('/api/app-keys',json=dict(source_name='Test',app_name='App'),
                                   headers={'Origin':'https://localhost'})
        self.assertEqual(response.status_code,201)

    def test_explicit_invalid_key_cannot_fall_back_to_admin_session(self):
        self.assertEqual(self.admin.post('/api/ocr',headers={'X-API-Key':'invalid'}).status_code,401)
        self.assertEqual(self.admin.post('/api/ocr').status_code,400)

    def test_preflight_does_not_require_credential(self):
        self.assertEqual(self.public.options('/api/ocr').status_code,200)

    def test_corrupt_store_fails_closed_without_overwrite(self):
        for contents in ['bad-json', '{}', '[{}]']:
            self.path.write_text(contents)
            self.assertEqual(self.create().status_code,503)
            self.assertEqual(self.admin.get('/api/app-keys').status_code,503)
            self.assertEqual(self.public.post('/api/ocr',headers={'X-API-Key':'key'}).status_code,503)
            self.assertEqual(self.path.read_text(),contents)

    def test_failed_atomic_replace_retains_old_key(self):
        key = self.create().json['data']
        before = self.path.read_bytes()
        with patch('api_keys.os.replace',side_effect=OSError('disk error')):
            self.assertEqual(self.admin.post(f"/api/app-keys/{key['id']}/regenerate").status_code,503)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(len(list(self.path.parent.iterdir())),2)  # Store and lock; no leaked temporary file.

    def test_concurrent_creates_do_not_lose_keys(self):
        def create(index):
            response = self.admin_client().post('/api/app-keys',json=dict(source_name=str(index),app_name='App'))
            return response.status_code
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(create,range(12))),[201]*12)
        records = KeyStore(self.path).read()
        self.assertEqual(len(records),12)
        self.assertEqual(len({key['api_key'] for key in records}),12)

    def test_guide_download_requires_login(self):
        self.assertEqual(self.public.get('/api-guide').status_code,302)
        response = self.admin.get('/api-guide')
        self.assertEqual(response.status_code,200)
        self.assertIn('attachment',response.headers['Content-Disposition'])
        response.close()


if __name__ == '__main__':
    unittest.main()
