"""Compare models on hash-assigned full passport photos, with an explicit evaluation role."""

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import pytesseract
import app
from tools.passport_dataset import load_labels, field_matches


def evaluate(args):
    """Keep failed scans in the denominator, comparing only independently held-out identities."""
    labels, _ = load_labels(args.labels)
    records = []
    seen = set()
    for cache in sorted((args.source_review / 'details').glob('*.json')):
        record = json.loads(cache.read_text())
        expected = labels.get(Path(record['source']).name.split('_護照_')[0])
        if not expected:
            continue
        group = hashlib.sha256(expected['passport_number'].encode()).hexdigest()
        if int(group[:8], 16) % 10 != 0 or record['id'] in seen:
            continue
        seen.add(record['id'])
        records.append((record, expected, group))
    if not records:
        raise ValueError('No held-out full passport images found')
    pytesseract.pytesseract.tesseract_cmd = str(args.tesseract.resolve())
    logging.getLogger().setLevel(logging.CRITICAL)
    report = {'test_images': len(records), 'test_passport_groups': len({g for _,_,g in records}),
              'evaluation_role': getattr(args, 'evaluation_role', 'held_out'),
              'selection': 'sha256(passport_number) first 8 hex digits modulo 10 equals 0', 'models': []}
    for model in args.models:
        model = model.resolve()
        if not model.is_file():
            raise ValueError(f'Missing model: {model}')
        app.OCR_LANGUAGE = model.stem
        app.OCR_CONFIG = (f'--tessdata-dir "{model.parent}" --oem 1 --psm 6 '
                          '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
        counts, correct, totals = Counter(), Counter(), Counter()
        results = []
        started = time.monotonic()
        for index, (source, expected, group) in enumerate(records, 1):
            result = {'id': source['id'], 'passport_group': group}
            matches = {field: False for field,value in expected.items() if value}
            try:
                with Image.open(source['source']) as image:
                    _, data, _ = app.recognize_passport(image)
                matches = field_matches(data, expected)
                counts['parsed'] += 1
                counts['checksum_passed'] += data['verification']['overall']
                counts['all_fields_correct'] += all(matches.values())
                counts['incorrect_checksum_pass'] += data['verification']['overall'] and not all(matches.values())
                result['verification'] = data['verification']
            except (ValueError, RuntimeError) as exc:
                result['error'] = type(exc).__name__
            for field, matched in matches.items():
                totals[field] += 1
                correct[field] += matched
            result['field_matches'] = matches
            results.append(result)
            print(f'{model.stem}: {index}/{len(records)} held-out photos evaluated', flush=True)
        report['models'].append({
            'model': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'counts': dict(counts), 'all_fields_accuracy': counts['all_fields_correct'] / len(records),
            'seconds': round(time.monotonic() - started, 2),
            'fields': {key: {'correct':correct[key], 'total':value} for key,value in totals.items()},
            'results': results,
        })
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved held-out full-photo comparison:', args.report)


def main():
    """Use pre-existing source inventory without writing labels into evaluation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--tesseract', type=Path, required=True)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--evaluation-role', choices=('held_out','development'), default='held_out',
                        help='Use development once these photos have informed pipeline changes')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
