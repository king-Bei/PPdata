"""TD3 passport parsing and check digits, independent of the OCR engine."""

import re
from datetime import date, datetime


DIGITS = str.maketrans({'O': '0', 'Q': '0', 'D': '0', 'I': '1',
                      'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'})
LETTERS = str.maketrans({'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'})


def mrz_checksum(data):
    """Return the ICAO 9303 weighted check digit for MRZ characters."""
    if not re.fullmatch(r'[A-Z0-9<]*', data):
        raise ValueError('MRZ 包含非法字元')
    values = [0 if c == '<' else int(c) if c.isdigit() else ord(c) - 55 for c in data]
    return sum(v * (7, 3, 1)[i % 3] for i, v in enumerate(values)) % 10


def format_date(value, kind='birth', today=None):
    """Resolve YYMMDD by field: birth cannot be future; expiry is nearest century."""
    if not re.fullmatch(r'[0-9]{6}', value):
        return 'Invalid Date'
    today = today or date.today()
    dates = []
    century = today.year // 100 * 100
    for base in (century - 100, century, century + 100):
        try:
            candidate = datetime.strptime(str(base + int(value[:2])) + value[2:], '%Y%m%d').date()
        except ValueError:
            continue
        if kind != 'birth' or candidate <= today:
            dates.append(candidate)
    if not dates:
        return 'Invalid Date'
    chosen = max(dates) if kind == 'birth' else min(dates, key=lambda d: abs((d - today).days))
    return chosen.strftime('%Y/%m/%d')


def parse_mrz(line1, line2):
    """Parse exactly two TD3 lines; correct only characters with known field types."""
    line1 = re.sub(r'\s+', '', line1.upper())
    line2 = re.sub(r'\s+', '', line2.upper())
    if len(line1) != 44 or len(line2) != 44:
        raise ValueError('MRZ 必須為兩行，每行 44 字元')
    if not re.fullmatch(r'P[A-Z<][A-Z0-9<]{42}', line1) or not re.fullmatch(r'[A-Z0-9<]{44}', line2):
        raise ValueError('MRZ 護照格式不正確')
    names = line1[5:].translate(LETTERS)
    if '<<' not in names or not names.split('<<', 1)[0].strip('<'):
        raise ValueError('MRZ 姓名分隔符號或姓氏缺失')
    surname, given = names.split('<<', 1)
    # Repeated fillers terminate the used name; later padding noise is not a name.
    given = given.split('<<', 1)[0]
    normalized = list(line2)
    for start, end in ((9, 10), (13, 20), (21, 28), (42, 44)):
        normalized[start:end] = line2[start:end].translate(DIGITS)
    normalized[10:13] = line2[10:13].translate(LETTERS)
    normalized = ''.join(normalized)

    def matches(value, digit):
        """Compare a numeric check digit without inventing absent digits."""
        return digit in '0123456789' and str(mrz_checksum(value)) == digit

    birth = format_date(normalized[13:19], 'birth')
    expiry = format_date(normalized[21:27], 'expiry')
    optional = normalized[28:42]
    optional_valid = matches(optional, normalized[42]) or (optional == '<' * 14 and normalized[42] == '<')
    verification = {
        'passport': matches(normalized[:9], normalized[9]),
        'birth_date': matches(normalized[13:19], normalized[19]) and birth != 'Invalid Date',
        'expiry_date': matches(normalized[21:27], normalized[27]) and expiry != 'Invalid Date',
        'optional_data': optional_valid,
        'composite': matches(normalized[:10] + normalized[13:20] + normalized[21:43], normalized[43]),
    }
    gender = normalized[20]
    structure_valid = (gender in {'M', 'F', '<'}
                       and bool(re.fullmatch(r'[A-Z]{3}|<{3}', normalized[10:13]))
                       and bool(re.fullmatch(r'[A-Z]{3}|<{3}', line1[2:5].translate(LETTERS))))
    verification['overall'] = all(verification.values()) and structure_valid
    return {
        'last_name': surname.replace('<', ' ').strip(),
        'first_name': ' '.join(given.replace('<', ' ').split()),
        'passport_number': normalized[:9].rstrip('<'),
        'nationality': normalized[10:13].replace('<', ''),
        'birth_date': birth,
        'expiry_date': expiry,
        'gender': gender if gender in {'M', 'F', '<'} else '',
        'ID_number': optional.replace('<', ''),
        'verification': verification,
    }


def verified_passport_repair(line1, line2, parsed):
    """Accept one ambiguous passport character only if one alternative passes all checks."""
    verification = parsed['verification']
    if verification['passport'] or not all(verification[k] for k in ('birth_date','expiry_date','optional_data')):
        return None
    solutions = []
    for index, character in enumerate(line2[:9]):
        replacement = character.translate(DIGITS)
        if replacement == character:
            continue
        corrected = line2[:index] + replacement + line2[index+1:]
        candidate = parse_mrz(line1, corrected)
        if candidate['verification']['overall']:
            candidate['corrections'] = [{'field':'passport_number', 'reason':'unique_checksummed_character'}]
            solutions.append(candidate)
    return solutions[0] if len(solutions) == 1 else None


def extract_mrz_data(text):
    """Rank adjacent TD3 pairs; also recover wrapped or joined 88-character blocks."""
    lines = [re.sub(r'[^A-Z0-9<]', '', line.upper()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    candidates = []
    for i, line in enumerate(lines):
        if not re.match(r'^P[A-Z<]', line):
            continue
        # Do not pad/truncate missing characters: offsets would become untrustworthy.
        block = ''
        for fragment in lines[i:i + 6]:
            block += fragment
            if len(block) == 88:
                try:
                    parsed = parse_mrz(block[:44], block[44:])
                    candidates.append(parsed)
                    repaired = verified_passport_repair(block[:44], block[44:], parsed)
                    if repaired:
                        candidates.append(repaired)
                except ValueError:
                    pass
                break
            if len(block) > 88:
                break
        # Raw-line OCR can add trailing name fillers or a few pixels of leading noise.
        # Only exact 44-character data windows with every check passing are accepted.
        if (44 <= len(line) <= 48 and set(line[44:]) <= {'<'}
                and line.endswith('<<<<') and i+1 < len(lines)):
            following = lines[i+1]
            if not 44 <= len(following) <= 48:
                continue
            for offset in [len(following)-44]:
                try:
                    parsed = parse_mrz(line[:44], following[offset:offset+44])
                except ValueError:
                    continue
                if parsed['verification']['overall']:
                    if len(line) > 44 or len(following) > 44:
                        parsed['corrections'] = [{'field':'MRZ', 'reason':'discarded_outer_ocr_noise'}]
                    candidates.append(parsed)
    if not candidates:
        raise ValueError('MRZ 資料不足或格式不正確，需完整兩行 44 字元')
    return max(candidates, key=lambda d: (d['verification']['overall'],
                                        sum(d['verification'].values())))
