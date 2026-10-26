"""Integrity and preservation checks for the standalone peak-list downloader."""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "peak_list_download", ROOT / "scripts" / "download_real_case_peak_lists.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class PeakListDownloadTests(unittest.TestCase):
    def setUp(self):
        # Synthetic fixtures keep tests independent of the experimental data.
        self.fixtures = {}
        cases = []
        for index, case_id in enumerate(downloader.CASES):
            original = f"q_xy,q_z\n0,{index + 1}\n1,{index + 1}\n".encode()
            filtered = f"q_xy,q_z\n1,{index + 1}\n".encode()
            self.fixtures[f"{case_id}.csv"] = original
            self.fixtures[f"{case_id}_without_specular.csv"] = filtered
            cases.append({
                "id": case_id, "filename": f"{case_id}.csv",
                "without_specular": {
                    "filename": f"{case_id}_without_specular.csv",
                    "sha256": hashlib.sha256(filtered).hexdigest(),
                    "source_sha256": hashlib.sha256(original).hexdigest(),
                },
            })
        manifest = json.dumps({"cases": cases}).encode()
        self.fixtures["manifest.json"] = manifest
        patcher = patch.object(downloader, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)

    def payload(self, filename):
        return self.fixtures[filename]

    def test_default_downloads_exact_specular_free_files_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            downloader, "download", side_effect=self.payload
        ):
            paths = downloader.download_peak_lists(Path(tmp))
            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertTrue(path.name.endswith("_without_specular.csv"))
                self.assertEqual(path.read_bytes(), self.payload(path.name))
            self.assertEqual(downloader.download_peak_lists(Path(tmp)), paths)

    def test_include_originals_downloads_all_eight(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            downloader, "download", side_effect=self.payload
        ):
            paths = downloader.download_peak_lists(Path(tmp), include_originals=True)
            self.assertEqual(len(paths), 8)
            self.assertEqual(len({path.name for path in paths}), 8)

    def test_corrupt_csv_leaves_existing_files_untouched(self):
        def corrupt(filename):
            data = self.payload(filename)
            return data + b"corruption" if filename.startswith("naib") else data

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            downloader, "download", side_effect=corrupt
        ):
            sentinel = Path(tmp) / "pq_without_specular.csv"
            sentinel.write_bytes(b"my measurements")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                downloader.download_peak_lists(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"my measurements")

    def test_refuses_different_existing_data_and_untrusted_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            downloader, "download", side_effect=self.payload
        ):
            sentinel = Path(tmp) / "pq_without_specular.csv"
            sentinel.write_bytes(b"my measurements")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                downloader.download_peak_lists(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            downloader, "download", return_value=b'{"cases": []}'
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch for manifest"):
                downloader.download_peak_lists(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
