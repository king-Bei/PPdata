"""Dataset validation and model evaluation regression tests."""

import tempfile
import unittest
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.ocr_training import edit_distance, evaluate, prepare, samples, train, validate_split


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def pair(self, name='sample', text='A' * 44, color='white'):
        """Create synthetic corrected line data without passport information."""
        image = self.root / (name + '.png')
        Image.new('L', (100, 40), color).save(image)
        (self.root / (name + '.gt.txt')).write_text(text, encoding='utf-8')
        return image, text

    def test_reads_corrected_pairs(self):
        pair = self.pair()
        self.assertEqual(samples(self.root), [pair])

    def test_rejects_missing_or_multiple_images(self):
        (self.root / 'sample.gt.txt').write_text('A' * 44)
        with self.assertRaises(ValueError):
            samples(self.root)
        self.pair()
        Image.new('L', (100, 40), 'white').save(self.root / 'sample.tif')
        with self.assertRaises(ValueError):
            samples(self.root)

    def test_rejects_uncorrected_or_multiline_labels(self):
        for text in ('A' * 43, 'A' * 44 + '\n' + 'B' * 44, 'a' * 44, 'A' * 43 + '!'):
            self.pair(text=text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                samples(self.root)

    def test_empty_dataset_fails(self):
        with self.assertRaises(ValueError):
            samples(self.root)

    def test_split_overlap_detects_same_pixels_in_different_formats(self):
        training = self.pair()
        copied = self.root / 'copy.tif'
        with Image.open(training[0]) as image:
            image.save(copied)
        with self.assertRaises(ValueError):
            validate_split([training], [(copied, training[1])])

    def test_distinct_split_is_accepted(self):
        validate_split([self.pair()], [self.pair('other', color='black')])

    def test_box_files_preserve_filler_and_terminal_tab(self):
        pair = self.pair(text='P' + '<' * 43)
        output = self.root / 'prepared'

        def create_lstmf(command):
            """Stand in for training binary, validating artifact generation inputs."""
            Path(str(command[2]) + '.lstmf').write_bytes(b'sample')

        with patch('tools.ocr_training.run', side_effect=create_lstmf):
            listing = prepare([pair], output, self.root / 'ocrb.traineddata')
        lines = (output / '000000.box').read_text().splitlines()
        self.assertEqual(len(lines), 45)
        self.assertTrue(lines[1].startswith('< 0 0 100 40'))
        self.assertTrue(lines[-1].startswith('\t 0 0'))
        self.assertEqual(Path(listing.read_text().strip()), output / '000000.lstmf')

    def test_cer_counts_insertion_deletion_and_substitution(self):
        self.assertEqual(edit_distance('ABC', 'ABC'), 0)
        self.assertEqual(edit_distance('ABC', 'ABXC'), 1)
        self.assertEqual(edit_distance('ABC', 'AC'), 1)
        self.assertEqual(edit_distance('ABC', 'ADC'), 1)
        self.assertEqual(edit_distance('ABC', ''), 3)

    def test_training_exports_candidate_without_modifying_base(self):
        training = self.pair()
        validation = self.pair('validation', color='black')
        base = self.root / 'ocrb.traineddata'
        base.write_bytes(b'original model')
        output = self.root / 'run'
        args = argparse.Namespace(base=base, train_dir=self.root, eval_dir=self.root,
                                  output=output, iterations=5)

        def tool(command):
            """Simulate file artifacts at each external tool boundary."""
            if command[0] == 'combine_tessdata':
                Path(command[-1]).write_bytes(b'network')
            elif command[0] == 'tesseract':
                Path(str(command[2]) + '.lstmf').write_bytes(b'lstm sample')
            elif '--stop_training' in command:
                Path(command[-1]).write_bytes(b'candidate model')
            else:
                prefix = command[command.index('--model_output') + 1]
                Path(str(prefix) + '_checkpoint').write_bytes(b'checkpoint')

        with patch('tools.ocr_training.shutil.which', return_value='/tool'), \
                patch('tools.ocr_training.samples', side_effect=[[training], [validation]]), \
                patch('tools.ocr_training.run', side_effect=tool):
            train(args)
        self.assertEqual(base.read_bytes(), b'original model')
        self.assertEqual((output / 'ocrb_candidate.traineddata').read_bytes(), b'candidate model')
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['train_count'], 1)
        self.assertEqual(manifest['eval_count'], 1)

    def test_evaluation_compares_models_on_identical_samples(self):
        self.pair()
        baseline = self.root / 'ocrb.traineddata'
        candidate = self.root / 'candidate.traineddata'
        baseline.touch()
        candidate.touch()
        report = self.root / 'report.json'
        args = argparse.Namespace(data=self.root, models=[baseline, candidate], report=report)
        with patch('pytesseract.image_to_string', side_effect=['B' + 'A' * 43, 'A' * 44]):
            evaluate(args)
        results = json.loads(report.read_text())['models']
        self.assertAlmostEqual(results[0]['character_error_rate'], 1 / 44)
        self.assertEqual(results[0]['exact_line_accuracy'], 0)
        self.assertEqual(results[1]['character_error_rate'], 0)
        self.assertEqual(results[1]['exact_line_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
