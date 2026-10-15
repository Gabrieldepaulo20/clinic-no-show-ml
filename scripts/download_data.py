"""Download the public source, validate the CSV, and record its exact hash."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://www.kaggle.com/datasets/joniarroba/noshowappointments"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/joniarroba/noshowappointments?datasetVersionNumber=5"
EXPECTED_SHA256 = "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"
ROOT = Path(__file__).resolve().parents[1]


def main():
    request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "clinic-no-show-ml/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(25_000_001)
    if len(data) > 25_000_000:
        raise ValueError("Unexpected download size")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 20_000_000:
            raise ValueError("Unexpected source archive")
        raw = archive.read(names[0])
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError("Dataset version differs from the reviewed source; inspect before training")
    destination = ROOT / "data/raw/appointments.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    metadata = {"source_url": SOURCE, "download_url": DOWNLOAD, "version": 5,
                "author": "Joni Hoppen / Aquarela Analytics", "license": "CC BY-NC-SA 4.0",
                "csv_sha256": checksum, "bytes": len(raw)}
    (ROOT / "data/source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset downloaded and verified: {len(raw)} bytes. Raw records are ignored by Git.")


if __name__ == "__main__":
    main()
