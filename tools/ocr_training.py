"""Fine-tune OCR-B on corrected line images and compare models on held-out lines."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

os.environ.setdefault('OMP_THREAD_LIMIT', '1')


EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}


def samples(directory):
    """Read single-line image/ground-truth pairs, rejecting silent omissions."""
    directory = Path(directory).resolve()
    pairs = []
    for label in sorted(directory.rglob('*.gt.txt')):
        stem = label.name[:-7]
        images = [p for p in label.parent.iterdir() if p.stem == stem and p.suffix.lower() in EXTENSIONS]
        if len(images) != 1:
            raise ValueError(f'{label}: expected exactly one matching image')
        text = label.read_text(encoding='utf-8').strip()
        if not re.fullmatch(r'[A-Z0-9<]{44}', text):
            raise ValueError(f'{label}: expected one corrected 44-character MRZ line')
        pairs.append((images[0], text))
    if not pairs:
        raise ValueError(f'No .gt.txt pairs found in {directory}')
    return pairs


def image_digest(path):
    """Hash decoded pixels so image format changes cannot bypass overlap checks."""
    with Image.open(path) as image:
        image = image.convert('L')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def validate_split(train, evaluation):
    """Reject pixel-identical samples shared between training and validation."""
    training_hashes = {image_digest(p) for p, _ in train}
    if any(image_digest(p) in training_hashes for p, _ in evaluation):
        raise ValueError('Training and evaluation images overlap; use different passports')


def run(command):
    """Run a Tesseract tool without invoking a shell, preserving its diagnostics."""
    subprocess.run([str(arg) for arg in command], check=True)


def prepare(pairs, output, base):
    """Generate full-line character boxes and serialized LSTM training samples."""
    output.mkdir(parents=True)
    files = []
    for index, (image_path, text) in enumerate(pairs):
        prefix = output / f'{index:06d}'
        with Image.open(image_path) as image:
            image = image.convert('L')
            width, height = image.size
            image.save(prefix.with_suffix('.png'))
        box = ''.join(f'{char} 0 0 {width} {height} 0\n' for char in text)
        box += f'\t 0 0 {width} {height} 0\n'
        prefix.with_suffix('.box').write_text(box, encoding='utf-8')
        run(['tesseract', prefix.with_suffix('.png'), prefix,
             '--tessdata-dir', base.parent, '-l', base.stem,
             '--psm', '13', '-c', 'tessedit_train_line_recognizer=1',
             '-c', 'tessedit_init_config_only=1'])
        lstmf = prefix.with_suffix('.lstmf')
        if not lstmf.is_file():
            raise ValueError(f'Tesseract did not generate {lstmf}')
        files.append(str(lstmf))
    listing = output / 'files.txt'
    listing.write_text('\n'.join(files) + '\n', encoding='utf-8')
    return listing


def train(args):
    """Build a separate candidate; never overwrite the serving model."""
    for command in ('tesseract', 'combine_tessdata', 'lstmtraining'):
        if not shutil.which(command):
            raise ValueError(f'Missing executable: {command}; install Tesseract training tools')
    base = args.base.resolve()
    if not base.is_file() or base.suffix != '.traineddata':
        raise ValueError('Base must be an existing floating-point .traineddata model')
    if args.iterations < 1:
        raise ValueError('iterations must be positive')
    training = samples(args.train_dir)
    evaluation = samples(args.eval_dir)
    validate_split(training, evaluation)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output directory already exists; use a new directory for each run')
    output.mkdir(parents=True)
    extracted = output / 'base.lstm'
    run(['combine_tessdata', '-e', base, extracted])
    train_list = prepare(training, output / 'train', base)
    eval_list = prepare(evaluation, output / 'eval', base)
    prefix = output / 'candidate'
    run(['lstmtraining', '--continue_from', extracted, '--traineddata', base,
         '--train_listfile', train_list, '--eval_listfile', eval_list,
         '--model_output', prefix, '--max_iterations', args.iterations,
         '--learning_rate', '0.0001', '--debug_interval', '0'])
    checkpoint = output / 'candidate_checkpoint'
    if not checkpoint.is_file():
        raise ValueError('Training did not produce candidate_checkpoint')
    model = output / 'ocrb_candidate.traineddata'
    run(['lstmtraining', '--stop_training', '--continue_from', checkpoint,
         '--traineddata', base, '--model_output', model])
    manifest = {
        'base_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
        'model': str(model), 'iterations': args.iterations,
        'train_count': len(training), 'eval_count': len(evaluation),
        'train_image_sha256': [image_digest(p) for p, _ in training],
        'eval_image_sha256': [image_digest(p) for p, _ in evaluation],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Candidate saved: {model}. Compare against the base on an independent test set before activation.')


def edit_distance(left, right):
    """Return Levenshtein distance, including insertions and deletions in CER."""
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def evaluate(args):
    """Compare baseline and candidate on identical held-out corrected line crops."""
    import pytesseract
    pairs = samples(args.data)
    report = {'samples': len(pairs), 'models': []}
    for model_path in args.models:
        model = model_path.resolve()
        if not model.is_file() or not re.fullmatch(r'[A-Za-z0-9_]+', model.stem):
            raise ValueError('Model must exist and have an alphanumeric/underscore name')
        errors = exact = characters = 0
        for path, expected in pairs:
            config = (f'--tessdata-dir "{model.parent}" --oem 1 --psm 13 '
                      '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
            with Image.open(path) as image:
                actual = pytesseract.image_to_string(image, lang=model.stem, config=config, timeout=15)
            actual = ''.join(actual.split())
            errors += edit_distance(expected, actual)
            exact += actual == expected
            characters += len(expected)
        report['models'].append({'model': str(model), 'character_error_rate': errors / characters,
                                 'exact_line_accuracy': exact / len(pairs)})
    # Deliberately exclude passport text and filenames from the summary report.
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    """Expose reproducible training and baseline/candidate comparison commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--train-dir', type=Path, required=True)
    training.add_argument('--eval-dir', type=Path, required=True)
    training.add_argument('--base', type=Path, default=Path('tessdata/ocrb.traineddata'))
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--iterations', type=int, default=800)
    training.set_defaults(action=train)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--data', type=Path, required=True)
    evaluation.add_argument('--models', type=Path, nargs='+', required=True)
    evaluation.add_argument('--report', type=Path, required=True)
    evaluation.set_defaults(action=evaluate)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
