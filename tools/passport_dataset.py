"""Inventory and benchmark local passport photos against manually verified XLSX data."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import pytesseract
import app
from mrz import mrz_checksum
from tools.ocr_training import edit_distance

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook_tables(path):
    """Read stored values only; never execute formulas, macros or document instructions."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        for name in archive.namelist():
            if not re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name):
                continue
            rows = []
            for row in ET.fromstring(archive.read(name)).findall('m:sheetData/m:row', NS):
                values = {}
                for cell in row.findall('m:c', NS):
                    value = cell.find('m:v', NS)
                    text = value.text if value is not None else ''
                    if cell.get('t') == 's' and text:
                        text = strings[int(text)]
                    elif cell.get('t') == 'inlineStr':
                        text = ''.join(cell.find('m:is', NS).itertext())
                    values[re.sub(r'\d', '', cell.get('r'))] = text or ''
                rows.append(values)
            if rows:
                headers = rows[0]
                yield [{headers.get(col, ''): value for col, value in row.items() if headers.get(col)} for row in rows[1:]]


def numeric_identifier(value):
    """Recover numeric passport identifiers stored in scientific notation by Excel."""
    try:
        number = Decimal(value)
        if number == number.to_integral_value() and 0 <= number < 10 ** 9:
            return str(int(number)).zfill(9)
    except (InvalidOperation, ValueError):
        pass
    return value.strip().upper()


def normalized_date(value):
    """Convert Excel serials and explicit dates to the API's calendar format."""
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).strftime('%Y/%m/%d')
        except ValueError:
            pass
    try:
        number = float(value)
        if 1 < number < 100000:
            return (datetime(1899, 12, 30) + timedelta(days=number)).strftime('%Y/%m/%d')
    except (ValueError, OverflowError):
        pass
    return ''


def normalized_name(value):
    """Normalize name spacing and punctuation for field-level comparisons."""
    return ' '.join(re.sub(r'[^A-Z]', ' ', value.upper()).split())


def load_labels(path, conflicts_path=None):
    """Match verified rows by Chinese name, excluding conflicting passport identities."""
    candidates = defaultdict(list)
    for table in workbook_tables(path):
        for row in table:
            if '英文姓' in row:
                name = row.get('中文姓', '') + row.get('中文名', '')
                surname, given = row.get('英文姓', ''), row.get('英文名', '')
                birth, expiry = row.get('出生日期', ''), row.get('截止日期', '')
                identity, gender = row.get('身分證號碼', ''), row.get('性別', '')
            elif '護照英文姓名' in row:
                name = row.get('姓名', '')
                parts = normalized_name(row.get('護照英文姓名', '')).split(' ', 1)
                surname, given = (parts + [''])[:2]
                birth, expiry = row.get('出生年月日', ''), row.get('護照效期', '')
                identity, gender = row.get('身分證字號', ''), ''
            else:
                continue
            passport = numeric_identifier(row.get('護照號碼', ''))
            if not name or not re.fullmatch(r'[A-Z0-9]{9}', passport) or passport in {'000000000'}:
                continue
            record = dict(passport_number=passport, last_name=normalized_name(surname),
                          first_name=normalized_name(given), birth_date=normalized_date(birth),
                          expiry_date=normalized_date(expiry), ID_number=identity.strip().upper(),
                          gender=gender.upper() if gender.upper() in {'M', 'F'} else '')
            candidates[name].append(record)
    labels = {}
    conflicts = 0
    conflict_rows = []
    for name, records in candidates.items():
        merged = {}
        inconsistent = False
        for record in records:
            for field, value in record.items():
                if value and merged.get(field) and value != merged[field]:
                    inconsistent = True
                if value:
                    merged[field] = value
        if inconsistent:
            conflicts += 1
            conflict_rows.append({'name': name, 'records': records})
        else:
            labels[name] = merged
    if conflicts_path is not None:
        conflicts_path.write_text(json.dumps(conflict_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return labels, conflicts


def expected_mrz(record):
    """Build Taiwan TD3 labels only when all manually verified source fields exist."""
    required = ('last_name', 'first_name', 'passport_number', 'birth_date', 'expiry_date', 'ID_number', 'gender')
    if not all(record.get(k) for k in required) or not re.fullmatch(r'[A-Z][0-9]{9}', record['ID_number']):
        return None
    names = record['last_name'].replace(' ', '<') + '<<' + record['first_name'].replace(' ', '<')
    line1 = ('P<TWN' + names).ljust(44, '<')[:44]
    passport = record['passport_number']
    birth = datetime.strptime(record['birth_date'], '%Y/%m/%d').strftime('%y%m%d')
    expiry = datetime.strptime(record['expiry_date'], '%Y/%m/%d').strftime('%y%m%d')
    optional = record['ID_number'].ljust(14, '<')
    line2 = (passport + str(mrz_checksum(passport)) + 'TWN' + birth + str(mrz_checksum(birth))
             + record['gender'] + expiry + str(mrz_checksum(expiry)) + optional + str(mrz_checksum(optional)))
    return line1, line2 + str(mrz_checksum(line2[:10] + line2[13:20] + line2[21:43]))


def field_matches(data, expected):
    """Compare each available independently verified field, without logging identities."""
    result = {}
    for field, value in expected.items():
        if not value:
            continue
        actual = data.get(field, '')
        if field in {'first_name', 'last_name'}:
            actual, value = normalized_name(actual).replace(' ', ''), normalized_name(value).replace(' ', '')
        result[field] = actual == value
    return result


def export_lines(crop, expected, directory, sample_id):
    """Locate adjacent MRZ lines by TSV boxes and attach verified, independent text."""
    ground_truth = expected_mrz(expected)
    if not ground_truth:
        return False
    info = pytesseract.image_to_data(crop, lang=app.OCR_LANGUAGE, config=app.OCR_CONFIG,
                                     timeout=10, output_type=pytesseract.Output.DICT)
    groups = defaultdict(list)
    for i, text in enumerate(info['text']):
        if text.strip():
            groups[(info['block_num'][i], info['par_num'][i], info['line_num'][i])].append(i)
    lines = []
    for indexes in groups.values():
        text = re.sub(r'[^A-Z0-9<]', '', ''.join(info['text'][i] for i in indexes).upper())
        if not 30 <= len(text) <= 50:
            continue
        x = min(info['left'][i] for i in indexes)
        y = min(info['top'][i] for i in indexes)
        right = max(info['left'][i] + info['width'][i] for i in indexes)
        bottom = max(info['top'][i] + info['height'][i] for i in indexes)
        lines.append((text, (x, y, right, bottom)))
    for text1, box1 in lines:
        if not text1.startswith('P<TWN') or edit_distance(text1, ground_truth[0]) > 8:
            continue
        for text2, box2 in lines:
            height = max(box1[3] - box1[1], box2[3] - box2[1])
            if not (box1[3] <= box2[1] < box1[3] + 3 * height):
                continue
            if edit_distance(text2, ground_truth[1]) > 8:
                continue
            if expected['passport_number'] not in text2 and expected['ID_number'] not in text2:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            midpoint = (box1[3] + box2[1]) // 2
            for number, (text, box) in enumerate(zip(ground_truth, (box1, box2)), 1):
                x1, y1, x2, y2 = box
                padding = max(4, int((y2-y1)*.25))
                top = max(0, y1-padding, midpoint if number == 2 else 0)
                bottom = min(crop.height, y2+padding, midpoint if number == 1 else crop.height)
                region = crop.crop((max(0,x1-padding), top, min(crop.width,x2+padding), bottom))
                prefix = directory / f'{sample_id}_line{number}'
                region.save(prefix.with_suffix('.png'))
                prefix.with_suffix('.gt.txt').write_text(text + '\n', encoding='utf-8')
            return True
    return False


def benchmark(args):
    """Benchmark photos locally, export grouped verified line pairs and an aggregate report."""
    pytesseract.pytesseract.tesseract_cmd = str(args.tesseract.resolve())
    logging.getLogger().setLevel(logging.CRITICAL)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = Path(app.tessdata_dir) / (app.OCR_LANGUAGE + '.traineddata')
    context = {
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        'app_sha256': hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
        'mrz_sha256': hashlib.sha256((Path(app.__file__).parent / 'mrz.py').read_bytes()).hexdigest(),
        'dataset_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tesseract_version': str(pytesseract.get_tesseract_version()),
        'source': str(args.source.resolve()),
        'labels_manually_verified_by_user': True,
    }
    context_path = args.output / 'context.json'
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError('Model, labels or pipeline changed; use a new output directory')
    context_path.write_text(json.dumps(context, indent=2), encoding='utf-8')
    labels, conflicts = load_labels(args.labels, args.output / 'label-conflicts.json')
    images = sorted(p for p in args.source.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if args.limit:
        images = images[:args.limit]
    details = args.output / 'details'
    details.mkdir(exist_ok=True)
    counts = Counter(images=len(images), labels=len(labels), conflicting_labels=conflicts)
    fields = Counter()
    totals = Counter()
    unique_results = {}

    def process(path):
        """Cache one photo result; source filenames remain in ignored local artifacts."""
        sample_id = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
        cache = details / (sample_id + '.json')
        if cache.exists():
            return json.loads(cache.read_text())
        expected = labels.get(path.name.split('_護照_')[0])
        result = {'id': sample_id, 'source': str(path), 'labeled': bool(expected)}
        started = time.monotonic()
        try:
            with Image.open(path) as image:
                _, data, text, crop = app.recognize_passport(image, include_crop=True)
            result.update(parsed=True, verified=data['verification']['overall'],
                          attempts=data['recognition']['attempts'])
            if expected:
                result['fields'] = field_matches(data, expected)
                group = hashlib.sha256(expected['passport_number'].encode()).hexdigest()
                bucket = int(group[:8], 16) % 10
                split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
                result['split'] = split
                # Guard the document type/country before deriving country-specific labels.
                if data['nationality'] == 'TWN':
                    result['training_pair'] = export_lines(crop, expected, args.output / 'dataset' / split, sample_id)
        except Exception as exc:
            result.setdefault('parsed', False)
            result['error'] = type(exc).__name__
        result['seconds'] = round(time.monotonic() - started, 3)
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(process, images), 1):
            unique_results[result['id']] = result
            counts['labeled_images'] += result['labeled']
            counts['parsed'] += result.get('parsed', False)
            counts['checksum_passed'] += result.get('verified', False)
            counts['training_images'] += result.get('training_pair', False)
            counts['failed'] += not result.get('parsed', False)
            for field, matched in result.get('fields', {}).items():
                totals[field] += 1
                fields[field] += matched
            if result.get('fields'):
                counts['all_labeled_fields_correct'] += all(result['fields'].values())
                counts['compared_images'] += 1
            if index % 20 == 0 or index == len(images):
                print(f'{index}/{len(images)} processed; parsed={counts["parsed"]}; checked={counts["checksum_passed"]}; training_images={counts["training_images"]}', flush=True)
    report = dict(counts)
    report['unique_images'] = len(unique_results)
    report['duplicate_images'] = len(images) - len(unique_results)
    report['unique_images_parsed'] = sum(r.get('parsed', False) for r in unique_results.values())
    report['unique_images_checksum_passed'] = sum(r.get('verified', False) for r in unique_results.values())
    groups = defaultdict(set)
    for result in unique_results.values():
        if result.get('training_pair'):
            expected = labels.get(Path(result['source']).name.split('_護照_')[0])
            if expected:
                groups[result['split']].add(hashlib.sha256(expected['passport_number'].encode()).hexdigest())
    report['training_passport_groups'] = {split: len(group) for split, group in groups.items()}
    report['training_passports'] = sum(len(group) for group in groups.values())
    report['dataset_line_images'] = {split: len(list((args.output / 'dataset' / split).glob('*.png')))
                                     for split in ('train', 'validation', 'test')}
    report['all_labeled_fields_accuracy'] = counts['all_labeled_fields_correct'] / max(1, counts['labeled_images'])
    report['field_matches_among_parsed'] = {key: {'correct': fields[key], 'compared': value} for key, value in totals.items()}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    """Use explicit source and output paths; never modify original photos/workbooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tesseract', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=range(1,5))
    parser.add_argument('--limit', type=int)
    benchmark(parser.parse_args())


if __name__ == '__main__':
    main()
