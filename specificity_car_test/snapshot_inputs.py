"""Snapshot existing outputs only. Never run classification or event construction."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--drive-data', type=Path, required=True, help='Local synced MyDrive/phd/Data directory')
SOURCE = parser.parse_args().drive_data
DEST = ROOT / 'specificity_car_test' / 'inputs'
DEST.mkdir(parents=True, exist_ok=True)
files = {
    'ceo_letter_event_study_panel_US.csv': SOURCE / 'ceo_letter_event_study_panel_US.csv',
    'df_ar_cluster.csv': SOURCE / 'df_ar_cluster.csv',
}
files.update({f'raw_prices/{t}.csv': SOURCE / 'event_study_raw_prices' / f'{t}.csv'
              for t in ['AIG', 'ALL', 'CB', 'MET', 'PGR', 'PRU', 'TRV', 'URTH']})
manifest = {'event_branch_commit': '6693c1d60844c31c76dc6a129cb40863dc50b05c',
            'specificity_branch_commit': '2428ae52895ffba97d280f5d776c9e88b376bfc9',
            'drive_file_id': '1ztptBqL5cdJwob_OuMzjEZduNTtX7ocY', 'files': {}}
branch_path = 'null_check/data/panel_reg_export.csv'
branch_file = DEST / 'panel_reg_export.csv'
branch_file.write_bytes(subprocess.check_output(
    ['git', 'show', manifest['specificity_branch_commit'] + ':' + branch_path], cwd=ROOT))
manifest['files']['panel_reg_export.csv'] = {
    'sha256': hashlib.sha256(branch_file.read_bytes()).hexdigest(),
    'source': 'specificity-pipeline-3d-regression:' + branch_path,
}
for name, source in files.items():
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest['files'][name] = {
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'source': 'MyDrive/phd/Data/' + source.relative_to(SOURCE).as_posix(),
    }
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Snapshotted {len(manifest["files"])} existing input files.')
