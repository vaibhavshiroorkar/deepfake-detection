"""Resumable download, verification and extraction of the open datasets of D-021.

Data tooling only: no model, training or detection code.

Usage: python scripts/data/fetch_open.py <step> [<step> ...]
Steps: asvspoof, progan, diffusion, itw, cnn_subset, lavdf
Every step downloads with curl (resumable), checks the size, records the SHA-256
(and compares it with the value the Hugging Face API reports where one exists),
extracts, checks file counts against the documented counts, deletes the archive
only after every check passed, and appends a line to data/_manifests/downloads.jsonl.
The Hugging Face token is read from the HF_TOKEN environment variable (or the
Windows user environment) at run time and is never written to a file.
"""
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INCOMING = DATA / "_incoming"
MANIFESTS = DATA / "_manifests"
UA = {"User-Agent": "Mozilla/5.0"}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


def total_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1]), r.geturl()


def hf_sha256(repo, filename):
    with urllib.request.urlopen(urllib.request.Request(
            f"https://huggingface.co/api/datasets/{repo}?blobs=true", headers=UA), timeout=60) as r:
        info = json.load(r)
    for s in info.get("siblings", []):
        if s["rfilename"] == filename:
            return (s.get("lfs") or {}).get("sha256")
    return None


def hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            tok = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 '[Environment]::GetEnvironmentVariable("HF_TOKEN","User")'],
                capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            tok = ""
    return tok or None


def curl_download(url, dest, expected, token=None, attempts=8):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == expected:
        say(f"already complete: {dest.name}")
        return
    for k in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            break
        say(f"curl attempt {k + 1}/{attempts} for {dest.name}: {have}/{expected} bytes present")
        cmd = ["curl", "-L", "-C", "-", "--fail", "--retry", "5", "--retry-delay", "10",
               "-A", UA["User-Agent"], "-o", str(part), "-sS"]
        stdin = None
        if token:  # header goes through stdin config so it never appears in the process list
            cmd += ["-K", "-"]
            stdin = f'header = "Authorization: Bearer {token}"\n'
        cmd.append(url)
        subprocess.run(cmd, input=stdin, text=True)
    got = part.stat().st_size if part.exists() else 0
    if got != expected:
        raise RuntimeError(f"{dest.name}: size {got} differs from expected {expected}")
    part.replace(dest)


def parallel_download(url, dest, expected, parts=8, attempts=40):
    """Several concurrent range requests for hosts that throttle each connection.

    Each part is kept in its own file and resumed after an interruption; the
    parts are joined and the size is checked before the file is renamed.
    """
    from concurrent.futures import ThreadPoolExecutor
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        say(f"already complete: {dest.name}")
        return
    step = -(-expected // parts)
    ranges = [(i, i * step, min(expected, (i + 1) * step) - 1) for i in range(parts)]

    def one(job):
        i, start, end = job
        pf = Path(f"{dest}.p{i:02d}")
        need = end - start + 1
        for k in range(attempts):
            have = pf.stat().st_size if pf.exists() else 0
            if have == need:
                return
            tmp = Path(f"{pf}.tmp")
            if tmp.exists():
                tmp.unlink()
            subprocess.run(["curl", "-L", "--fail", "-sS", "-A", UA["User-Agent"], "-r",
                            f"{start + have}-{end}", "--max-time", "900", "-o", str(tmp), url])
            if tmp.exists() and tmp.stat().st_size > 0:
                with open(pf, "ab") as out, open(tmp, "rb") as src:
                    while True:
                        chunk = src.read(8 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                tmp.unlink()
        raise RuntimeError(f"part {i} of {dest.name} did not complete")

    say(f"parallel download of {dest.name}: {parts} parts of about {step / 1e6:.0f} MB")
    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(one, ranges))
    part = dest.with_suffix(dest.suffix + ".part")
    with open(part, "wb") as out:
        for i, _, _ in ranges:
            pf = Path(f"{dest}.p{i:02d}")
            with open(pf, "rb") as src:
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    if part.stat().st_size != expected:
        raise RuntimeError(f"{dest.name}: joined size {part.stat().st_size} differs from {expected}")
    part.replace(dest)
    for i, _, _ in ranges:
        Path(f"{dest}.p{i:02d}").unlink()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name, url, path, size, sha, verified_against, extra=None):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    entry = {"date": now(), "name": name, "url": url, "file": str(path), "bytes": size,
             "sha256": sha, "sha256_checked_against": verified_against}
    if extra:
        entry.update(extra)
    with open(MANIFESTS / "downloads.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    with open(MANIFESTS / f"{name}.sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha}  {Path(path).name}\n")
    say(f"recorded {name}: {size} bytes sha256 {sha}")


def fetch_archive(name, url, dest, expected_size=None, hf=None, token=None, parts=1):
    """Download, hash and (for HF files) compare the hash with the API value."""
    if expected_size is None:
        expected_size, url = total_size(url)
    if parts > 1:
        parallel_download(url, dest, expected_size, parts=parts)
    else:
        curl_download(url, dest, expected_size, token=token)
    sha = sha256_file(dest)
    against = "none published, recorded only"
    if hf:
        api = hf_sha256(*hf)
        if api and api != sha:
            raise RuntimeError(f"{name}: SHA-256 {sha} differs from Hugging Face API value {api}")
        if api:
            against = "Hugging Face API (LFS sha256): match"
    record(name, url, dest, dest.stat().st_size, sha, against)
    return sha


def extract_zip(archive, target, expected_files=None, pattern_ok=None):
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"corrupt member {bad} in {archive}")
        for info in z.infolist():
            if info.is_dir():
                continue
            if pattern_ok and not pattern_ok(info.filename):
                continue
            z.extract(info, target)
            n += 1
    say(f"extracted {n} files from {Path(archive).name} to {target}")
    if expected_files is not None and n != expected_files:
        raise RuntimeError(f"{archive}: {n} files, expected {expected_files}")
    return n


def count_ext(root, exts):
    return sum(1 for p in Path(root).rglob("*") if p.suffix.lower() in exts)


# ---------------------------------------------------------------- steps
def step_asvspoof():
    d = INCOMING / "asvspoof2019"
    base = "https://datashare.ed.ac.uk/bitstreams"
    fetch_archive("asvspoof2019_LA_zip", f"{base}/a9f87c35-f055-4015-80e2-2fdff0d46269/download",
                  d / "LA.zip", expected_size=7640952520, parts=8)
    out = DATA / "asvspoof2019_la"
    extract_zip(d / "LA.zip", out)
    # The countermeasure (CM) protocol files are the authoritative lists. The folders also hold
    # speaker-verification enrolment utterances (142 in dev, 696 in eval) that are not in the CM protocols.
    cm = out / "LA" / "ASVspoof2019_LA_cm_protocols"
    docs = {"train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train", 25380),
            "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev", 24844),
            "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval", 71237)}
    for part, (fname, sub, documented) in docs.items():
        ids = [l.split()[1] for l in open(cm / fname, encoding="utf-8") if l.strip()]
        on_disk = {p.stem for p in (out / "LA" / sub / "flac").glob("*.flac")}
        missing = [i for i in ids if i not in on_disk]
        say(f"ASVspoof 2019 LA {part}: protocol {len(ids)} (documented {documented}), "
            f"files on disk {len(on_disk)}, protocol entries without a file {len(missing)}")
        if len(ids) != documented or missing:
            raise RuntimeError(f"ASVspoof 2019 LA {part} differs from the documented protocol")
    req = urllib.request.Request(f"{base}/765597d6-633f-418a-96b0-39274a8d56c9/download", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        (out / "README_edinburgh.txt").write_bytes(r.read())
    (d / "LA.zip").unlink()


def step_progan():
    hfb = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main"
    for fname, sz, folder in (("progan_val.zip", 830792545, "progan_val"),
                              ("progan_testset.zip", 834177578, "progan_test")):
        a = INCOMING / "forensynths" / fname
        fetch_archive(f"forensynths_{folder}", f"{hfb}/{fname}", a, expected_size=sz,
                      hf=("sywang/CNNDetection", fname))
        out = DATA / "forensynths" / folder
        extract_zip(a, out, expected_files=8000)
        a.unlink()


def step_diffusion():
    gid = "1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t"
    a = INCOMING / "diffusion" / "diffusion_datasets.zip"
    a.parent.mkdir(parents=True, exist_ok=True)
    if not a.exists():
        subprocess.run(["uvx", "--quiet", "gdown", "--continue", f"https://drive.google.com/uc?id={gid}", "-O", str(a)], check=True)
    size = a.stat().st_size
    sha = sha256_file(a)
    record("diffusion_test_zip", f"https://drive.google.com/uc?id={gid}", a, size, sha,
           "none published, recorded only")
    out = DATA / "diffusion_test"
    extract_zip(a, out)
    per = {}
    for p in out.rglob("*"):
        if p.is_file():
            per[str(p.parent.relative_to(out))] = per.get(str(p.parent.relative_to(out)), 0) + 1
    say("files per folder: " + json.dumps(per))
    (MANIFESTS / "diffusion_test_counts.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
    a.unlink()


def step_itw():
    a = INCOMING / "in_the_wild" / "release_in_the_wild.zip"
    fetch_archive("in_the_wild_zip",
                  "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip",
                  a, expected_size=8161489023, hf=("mueller91/In-The-Wild", "release_in_the_wild.zip"), parts=8)
    out = DATA / "in_the_wild"
    extract_zip(a, out)
    n = count_ext(out, {".wav", ".flac", ".mp3"})
    say(f"audio files found: {n} (clip count is not stated by the source; recorded as the reference)")
    (MANIFESTS / "in_the_wild_counts.json").write_text(json.dumps({"audio_files": n}), encoding="utf-8")
    a.unlink()


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file using HTTP range requests."""

    def __init__(self, url):
        self.size, self.url = total_size(url)
        self.pos = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    BLOCK = 32 << 20

    def _fetch_block(self, start):
        end = min(self.size, start + self.BLOCK) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    self.fetched += len(data)
                    return data
            except Exception:
                if attempt == 7:
                    raise
        raise RuntimeError("range read failed")

    def read(self, n=-1):
        """Serve reads from a 32 MB read-ahead block so that sequential access costs few requests."""
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        out = b""
        while n > 0:
            blk = getattr(self, "_blk", None)
            if blk is None or not (blk[0] <= self.pos < blk[0] + len(blk[1])):
                self._blk = (self.pos, self._fetch_block(self.pos))
                blk = self._blk
            off = self.pos - blk[0]
            take = blk[1][off:off + n]
            out += take
            self.pos += len(take)
            n -= len(take)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


CNN_SUBSET = ("biggan", "gaugan", "stylegan", "stylegan2", "whichfaceisreal", "deepfake", "stargan", "cyclegan")


def step_cnn_subset():
    url = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip"
    rf = RangeFile(url)
    z = zipfile.ZipFile(rf)
    out = DATA / "forensynths" / "cnn_synth_subset"
    want = [i for i in z.infolist() if not i.is_dir() and i.filename.split("/")[0] in CNN_SUBSET]
    want.sort(key=lambda i: i.header_offset)
    total = sum(i.file_size for i in want)
    say(f"{len(want)} members, {total / 1e9:.2f} GB uncompressed, from an archive of {rf.size} bytes")
    done = 0
    counts = {}
    for k, info in enumerate(want):
        target = out / info.filename
        if target.exists() and target.stat().st_size == info.file_size:
            done += info.file_size
        else:
            z.extract(info, out)  # CRC is checked by zipfile while reading
            done += info.file_size
        counts[info.filename.split("/")[0]] = counts.get(info.filename.split("/")[0], 0) + 1
        if k % 2000 == 0:
            say(f"{k}/{len(want)} members, {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    n_found = sum(1 for p in out.rglob("*") if p.is_file())
    if n_found != len(want):
        raise RuntimeError(f"cnn subset: {n_found} files on disk, {len(want)} expected")
    record("cnn_synth_subset", url, out, total, "not applicable (member CRCs verified by zipfile)",
           "zip member CRC32", {"members": len(want), "per_generator": counts, "archive_bytes": rf.size})


def step_lavdf():
    tok = hf_token()
    if not tok:
        say("HF_TOKEN is not set: LAV-DF needs the user's Hugging Face token. Skipping.")
        return
    a = INCOMING / "lavdf" / "LAV-DF.tar"
    fetch_archive("lavdf_tar", "https://huggingface.co/datasets/ControlNet/LAV-DF/resolve/main/LAV-DF.tar",
                  a, expected_size=25607198720, hf=("ControlNet/LAV-DF", "LAV-DF.tar"), token=tok)
    say("LAV-DF.tar downloaded and hashed; extraction and counting are done by a separate step "
        "after the metadata schema has been read")


STEPS = {"asvspoof": step_asvspoof, "progan": step_progan, "diffusion": step_diffusion,
         "itw": step_itw, "cnn_subset": step_cnn_subset, "lavdf": step_lavdf}

if __name__ == "__main__":
    for s in sys.argv[1:]:
        say(f"=== step {s} ===")
        STEPS[s]()
        say(f"=== step {s} done ===")
