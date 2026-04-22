import hashlib
import os
import pathlib

import pytest

from rattler import PackageCache, PackageName, PackageRecord, RepoDataRecord


TEST_DATA = pathlib.Path(__file__).resolve().parents[3] / "test-data"
PACKAGE = TEST_DATA / "clobber" / "clobber-fd-1-0.1.0-h4616a5c_0.conda"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.asyncio
async def test_package_cache_fetch_from_file_url(tmp_path: pathlib.Path) -> None:
    assert PACKAGE.is_file(), f"missing test data: {PACKAGE}"
    sha = _sha256(PACKAGE)

    pkg_record = PackageRecord(
        name=PackageName("clobber-fd-1"),
        version="0.1.0",
        build="h4616a5c_0",
        build_number=0,
        subdir="linux-64",
        sha256=bytes.fromhex(sha),
    )
    record = RepoDataRecord(
        package_record=pkg_record,
        file_name=PACKAGE.name,
        url=PACKAGE.as_uri(),
        channel="https://example.com",
    )

    cache = PackageCache(tmp_path)
    extracted = await cache.get_or_fetch_from_url(record, PACKAGE.as_uri())

    assert extracted.is_dir()
    # Expected directory name per CacheKey Display: name-version-build
    assert extracted.name == "clobber-fd-1-0.1.0-h4616a5c_0"
    assert extracted.parent == tmp_path
    # Extracted contents should contain info/index.json
    assert (extracted / "info" / "index.json").is_file()

    # Calling a second time should be a no-op and return the same path.
    extracted2 = await cache.get_or_fetch_from_url(record, PACKAGE.as_uri())
    assert os.fspath(extracted) == os.fspath(extracted2)
