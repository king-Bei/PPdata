"""Full-photo evaluation keeps failed scans and independent groups in its metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import app
from mrz import parse_mrz
from tools.passport_dataset import expected_mrz
from tools.evaluate_passports import evaluate


class FullPhotoEvaluationTests(unittest.TestCase):
    def test_failed_scans_stay_in_accuracy_denominator(self):
        passport = next(str(n).zfill(9) for n in range(1,1000)
                        if int(hashlib.sha256(str(n).zfill(9).encode()).hexdigest()[:8],16) % 10 == 0)
        expected = dict(passport_number=passport, last_name='TEST', first_name='ANNA',
                        birth_date='1990/01/01', expiry_date='2030/01/01', gender='F', ID_number='A123456789')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'details').mkdir()
            for i in range(2):
                source = root / f'測試_護照_{i}.png'
                Image.new('RGB',(200,100),'white').save(source)
                (root / 'details' / f'{i}.json').write_text(json.dumps({'id':str(i),'source':str(source)}))
            model = root / 'ocrb.traineddata'
            model.write_bytes(b'model')
            report = root / 'comparison.json'
            args = argparse.Namespace(labels=root/'labels.xlsx', source_review=root,
                                      tesseract=root/'tesseract', models=[model], report=report)
            parsed = parse_mrz(*expected_mrz(expected))
            with patch('tools.evaluate_passports.load_labels', return_value=({'測試':expected},0)), \
                 patch.object(app, 'OCR_LANGUAGE', app.OCR_LANGUAGE), \
                 patch.object(app, 'OCR_CONFIG', app.OCR_CONFIG), \
                 patch.object(app, 'recognize_passport', side_effect=[(None,parsed,''),ValueError('unreadable')]):
                evaluate(args)
            result = json.loads(report.read_text())
            self.assertEqual(result['test_images'],2)
            self.assertEqual(result['test_passport_groups'],1)
            self.assertEqual(result['models'][0]['all_fields_accuracy'],.5)
            self.assertEqual(result['models'][0]['fields']['passport_number'],{'correct':1,'total':2})


if __name__ == '__main__':
    unittest.main()
