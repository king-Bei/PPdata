"""Regression coverage for passport parsing and OCR failure recovery."""

import base64
import io
import unittest
from datetime import date
from unittest.mock import patch

from PIL import Image

import app
from mrz import extract_mrz_data, format_date, mrz_checksum, parse_mrz


LINE1 = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<'
LINE2 = 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'
TEXT = LINE1 + '\n' + LINE2


def make_line2(birth='900101', expiry='300101', gender='M', optional='<' * 14):
    """Build synthetic MRZ fields with independently recomputed check digits."""
    line = ('A12345678' + str(mrz_checksum('A12345678')) + 'TWN'
            + birth + str(mrz_checksum(birth)) + gender
            + expiry + str(mrz_checksum(expiry)) + optional + str(mrz_checksum(optional)))
    return line + str(mrz_checksum(line[:10] + line[13:20] + line[21:43]))


class ParserTests(unittest.TestCase):
    def test_icao_example(self):
        data = parse_mrz(LINE1, LINE2)
        self.assertTrue(data['verification']['overall'])
        self.assertEqual(data['passport_number'], 'L898902C3')
        self.assertEqual(data['last_name'], 'ERIKSSON')
        self.assertEqual(data['first_name'], 'ANNA MARIA')
        self.assertEqual(data['birth_date'], '1974/08/12')

    def test_letters_after_name_padding_are_not_given_names(self):
        line = LINE1[:-2] + 'X<'
        self.assertEqual(parse_mrz(line, LINE2)['first_name'], 'ANNA MARIA')

    def test_outer_ocr_noise_requires_complete_checks(self):
        data = extract_mrz_data(LINE1 + '<\nXX' + LINE2)
        self.assertTrue(data['verification']['overall'])
        self.assertEqual(data['passport_number'], 'L898902C3')
        self.assertTrue(data['corrections'])
        with self.assertRaises(ValueError):
            extract_mrz_data(LINE1 + '<\nXX' + LINE2[:-1] + '1')

    def test_unique_passport_correction_is_explicit(self):
        text = LINE1 + '\n' + LINE2.replace('L898902C3','L8989O2C3')
        data = extract_mrz_data(text)
        self.assertTrue(data['verification']['overall'])
        self.assertEqual(data['passport_number'],'L898902C3')
        self.assertEqual(data['corrections'][0]['field'],'passport_number')

    def test_missing_final_check_digits_are_never_reconstructed(self):
        with self.assertRaises(ValueError):
            extract_mrz_data(LINE1 + '\n' + LINE2[:-2])

    def test_composite_must_pass(self):
        data = parse_mrz(LINE1, LINE2[:-1] + '1')
        self.assertTrue(data['verification']['passport'])
        self.assertFalse(data['verification']['composite'])
        self.assertFalse(data['verification']['overall'])

    def test_optional_data_must_pass(self):
        data = parse_mrz(LINE1, LINE2[:28] + 'X' + LINE2[29:])
        self.assertFalse(data['verification']['optional_data'])
        self.assertFalse(data['verification']['overall'])

    def test_unknown_gender_is_not_female(self):
        data = parse_mrz(LINE1, make_line2(gender='X'))
        self.assertEqual(data['gender'], '')
        self.assertFalse(data['verification']['overall'])
        self.assertEqual(parse_mrz(LINE1, make_line2(gender='<'))['gender'], '<')

    def test_empty_optional_check_can_be_filler(self):
        line = make_line2()
        line = line[:42] + '<' + line[43:]
        self.assertTrue(parse_mrz(LINE1, line)['verification']['overall'])

    def test_numeric_confusion_is_corrected_before_composite_check(self):
        line = make_line2().replace('900101', '9OOIOI').replace('300101', '3OOIOI')
        self.assertTrue(parse_mrz(LINE1, line)['verification']['overall'])

    def test_passport_is_not_guessed(self):
        line = LINE2.replace('L898902C3', 'L8989O2C3')
        self.assertEqual(parse_mrz(LINE1, line)['passport_number'], 'L8989O2C3')
        self.assertFalse(parse_mrz(LINE1, line)['verification']['passport'])

    def test_invalid_calendar_date_fails_even_with_matching_checksum(self):
        data = parse_mrz(LINE1, make_line2(birth='900230'))
        self.assertEqual(data['birth_date'], 'Invalid Date')
        self.assertFalse(data['verification']['birth_date'])
        self.assertFalse(data['verification']['overall'])

    def test_century_depends_on_field_and_today(self):
        today = date(2026, 10, 25)
        self.assertEqual(format_date('300101', 'birth', today), '1930/01/01')
        self.assertEqual(format_date('300101', 'expiry', today), '2030/01/01')
        self.assertEqual(format_date('500101', 'expiry', today), '2050/01/01')
        self.assertEqual(format_date('270101', 'birth', today), '1927/01/01')
        self.assertEqual(format_date('000229', 'birth', today), '2000/02/29')

    def test_joined_and_wrapped_lines(self):
        for text in (LINE1 + LINE2, LINE1[:20] + '\n' + LINE1[20:] + '\n' + LINE2,
                     LINE1 + '\n' + LINE2[:20] + '\n' + LINE2[20:]):
            with self.subTest(text=text):
                self.assertTrue(extract_mrz_data(text)['verification']['overall'])

    def test_prefers_valid_pair_over_later_noise(self):
        text = TEXT + '\n' + LINE1 + '\n' + LINE2[:-1] + '1'
        self.assertTrue(extract_mrz_data(text)['verification']['overall'])

    def test_rejects_incomplete_or_oversized_lines(self):
        for text in (LINE1[:-1] + '\n' + LINE2, LINE1 + '\n' + LINE2 + '0',
                     'A' * 44 + '\n' + LINE2, 'not a passport'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_mrz_data(text)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (600, 300), 'white')

    def test_blank_image_has_no_divide_by_zero(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(app.detect_mrz_region(self.image).size, self.image.size)

    def test_transparency_is_white_not_black(self):
        image = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        page, _ = app.preprocess_image_pil(image)
        self.assertEqual(page.getpixel((0, 0)), 255)

    def test_face_crop_keeps_original_for_fallback_scans(self):
        with patch.object(app, 'crop_to_data_page_via_face', return_value=self.image.crop((0,150,600,300))):
            page, region, original = app.preprocess_image_pil(self.image, include_original=True)
        self.assertEqual(page.height, 150)
        self.assertEqual(original.height, 300)
        variants = list(app.mrz_image_variants(page,region,original))
        self.assertEqual(len(variants),7)
        self.assertEqual(variants[3][1].height,300)

    def test_valid_result_stops_retrying(self):
        with patch.object(app.pytesseract, 'image_to_string', return_value=TEXT) as ocr:
            _, data, _ = app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count, 1)
        self.assertFalse(data['recognition']['needs_review'])

    def test_offline_training_can_retrieve_selected_crop(self):
        with patch.object(app.pytesseract, 'image_to_string', return_value=TEXT):
            page, data, text, crop = app.recognize_passport(self.image, include_crop=True)
        self.assertIsInstance(crop, Image.Image)
        self.assertEqual(text, TEXT)
        self.assertTrue(data['verification']['overall'])

    def test_retry_recovers_from_bad_crop(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=['noise', TEXT]) as ocr:
            _, data, _ = app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count, 2)
        self.assertTrue(data['verification']['overall'])

    def test_split_line_modes_reuse_the_already_read_name(self):
        lines=[Image.new('L',(600,40),'white') for _ in range(2)]
        with patch.object(app,'split_mrz_lines',return_value=lines), \
             patch.object(app.pytesseract,'image_to_string',side_effect=['noise',LINE1,LINE2[:-1]+'1',LINE2]) as ocr:
            _,data,_=app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count,4)
        self.assertEqual(data['recognition']['method'],'mixed_lines')
        self.assertTrue(data['verification']['overall'])

    def test_deadline_stops_ocr_before_starting_another_process(self):
        with patch.object(app.time,'monotonic',side_effect=[0,46,46]), \
             patch.object(app.pytesseract,'image_to_string') as ocr:
            with self.assertRaises(RuntimeError):
                app.recognize_passport(self.image)
        ocr.assert_not_called()

    def test_corrected_passport_always_requires_human_review(self):
        text=LINE1+'\n'+LINE2.replace('L898902C3','L8989O2C3')
        with patch.object(app.pytesseract,'image_to_string',return_value=text):
            _,data,_=app.recognize_passport(self.image)
        self.assertTrue(data['verification']['overall'])
        self.assertTrue(data['recognition']['needs_review'])

    def test_clean_read_takes_precedence_over_corrected_candidate(self):
        corrected=LINE1+'<\nXX'+LINE2
        with patch.object(app.pytesseract,'image_to_string',side_effect=[corrected,TEXT]) as ocr:
            _,data,_=app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count,2)
        self.assertTrue(data['verification']['overall'])
        self.assertFalse(data['recognition']['needs_review'])
        self.assertNotIn('corrections',data)

    def test_request_never_exceeds_fourteen_ocr_processes(self):
        attempts=[(self.image,Image.new('L',(80,40),'white'),'fallback',[]) for _ in range(30)]
        attempts=[(page,crop,method,[(crop,6)]) for page,crop,method,_ in attempts]
        with patch.object(app,'recognition_attempts',return_value=iter(attempts)), \
             patch.object(app.pytesseract,'image_to_string',return_value='noise') as ocr:
            with self.assertRaises(RuntimeError):
                app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count,14)
        self.assertTrue(all(call.kwargs['timeout']<=10 for call in ocr.call_args_list))

    def test_best_partial_result_requires_review(self):
        partial = LINE1 + '\n' + LINE2[:-1] + '1'
        with patch.object(app.pytesseract, 'image_to_string', side_effect=[partial] + ['noise'] * 6):
            _, data, _ = app.recognize_passport(self.image)
        self.assertTrue(data['recognition']['needs_review'])
        self.assertEqual(data['recognition']['attempts'], 7)

    def test_all_timeouts_are_bounded(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=RuntimeError('timeout')) as ocr:
            with self.assertRaises(RuntimeError):
                app.recognize_passport(self.image)
        self.assertEqual(ocr.call_count, 7)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        buffer = io.BytesIO()
        Image.new('RGB', (200, 100), 'white').save(buffer, format='PNG')
        self.image = buffer.getvalue()

    def test_malformed_json_and_base64_are_client_errors(self):
        for payload in (['image'], {'image': 123}, {'image': '!invalid'}, {}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/ocr', json=payload).status_code, 400)

    def test_non_image_is_client_error(self):
        response = self.client.post('/api/ocr', data={'image': (io.BytesIO(b'bad'), 'bad.png')})
        self.assertEqual(response.status_code, 400)

    def test_base64_data_url_and_multipart(self):
        with patch.object(app.pytesseract, 'image_to_string', return_value=TEXT), patch.object(app, 'extract_chinese_name', return_value=''):
            encoded = base64.b64encode(self.image).decode()
            responses = [self.client.post('/api/ocr', json={'image': 'data:image/png;base64,' + encoded}),
                         self.client.post('/api/ocr', data={'image': (io.BytesIO(self.image), 'sample.png')})]
        for response in responses:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json['data']['verification']['composite'])

    def test_missing_engine_is_server_error(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=app.pytesseract.TesseractNotFoundError()):
            response = self.client.post('/api/ocr', data={'image': (io.BytesIO(self.image), 'sample.png')})
        self.assertEqual(response.status_code, 500)

    def test_timeout_is_gateway_timeout(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=RuntimeError('timeout')):
            response = self.client.post('/api/ocr', data={'image': (io.BytesIO(self.image), 'sample.png')})
        self.assertEqual(response.status_code, 504)

    def test_engine_error_is_not_reported_as_timeout(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=app.pytesseract.TesseractError(1, 'bad model')):
            response = self.client.post('/api/ocr', data={'image': (io.BytesIO(self.image), 'sample.png')})
        self.assertEqual(response.status_code, 500)

    def test_csv_uses_same_recovery_pipeline(self):
        with patch.object(app.pytesseract, 'image_to_string', side_effect=['noise', TEXT]), patch('builtins.open', unittest.mock.mock_open()), patch.object(app.os, 'makedirs'):
            response = self.client.post('/api/ocr-csv', data={'image': (io.BytesIO(self.image), 'sample.png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['detected_text'], TEXT)


if __name__ == '__main__':
    unittest.main()
