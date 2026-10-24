"""Verified spreadsheet conversion and passport grouping regression tests."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.passport_dataset import (expected_mrz, field_matches, load_labels,
                                    normalized_date, numeric_identifier)
from mrz import parse_mrz


class DatasetTests(unittest.TestCase):
    def test_excel_scientific_notation_keeps_identifier(self):
        self.assertEqual(numeric_identifier('3.12345678E8'), '312345678')
        self.assertEqual(numeric_identifier('12345678'), '012345678')
        self.assertEqual(numeric_identifier('A12345678'), 'A12345678')
        self.assertEqual(numeric_identifier('新辦/待補'), '新辦/待補')

    def test_excel_dates_and_iso_dates_agree(self):
        self.assertEqual(normalized_date('43831.0'), '2020/01/01')
        self.assertEqual(normalized_date('2020-01-01'), '2020/01/01')
        self.assertEqual(normalized_date('新辦/待補'), '')

    def test_independent_labels_generate_valid_td3(self):
        record = dict(last_name='TEST', first_name='ANNA MARIA', passport_number='123456789',
                      birth_date='1990/01/01', expiry_date='2030/01/01', ID_number='A123456789', gender='F')
        lines = expected_mrz(record)
        self.assertEqual([len(line) for line in lines], [44,44])
        result = parse_mrz(*lines)
        self.assertTrue(result['verification']['overall'])
        self.assertTrue(all(field_matches(result, record).values()))

    def test_missing_fields_cannot_become_ground_truth(self):
        self.assertIsNone(expected_mrz({'passport_number':'123456789'}))

    def test_conflicting_workbook_rows_are_excluded(self):
        rows = [[{'中文姓':'測', '中文名':'試', '英文姓':'TEST', '英文名':'ANNA',
                  '護照號碼':'123456789'},
                 {'中文姓':'測', '中文名':'試', '英文姓':'TEST', '英文名':'ANNA',
                  '護照號碼':'987654321'}]]
        with patch('tools.passport_dataset.workbook_tables', return_value=rows):
            labels, conflicts = load_labels(Path('unused.xlsx'))
        self.assertEqual(labels, {})
        self.assertEqual(conflicts, 1)

    def test_missing_cells_do_not_overwrite_verified_values(self):
        rows = [[{'中文姓':'測', '中文名':'試', '英文姓':'TEST', '英文名':'ANNA',
                  '護照號碼':'123456789', '出生日期':'1990-01-01'},
                 {'中文姓':'測', '中文名':'試', '英文姓':'TEST', '英文名':'ANNA',
                  '護照號碼':'123456789', '出生日期':''}]]
        with patch('tools.passport_dataset.workbook_tables', return_value=rows):
            labels, conflicts = load_labels(Path('unused.xlsx'))
        self.assertEqual(conflicts, 0)
        self.assertEqual(next(iter(labels.values()))['birth_date'], '1990/01/01')

    def test_name_spacing_does_not_inflate_field_errors(self):
        self.assertTrue(field_matches({'first_name':'ANNA MARIA'}, {'first_name':'ANNAMARIA'})['first_name'])


if __name__ == '__main__':
    unittest.main()
