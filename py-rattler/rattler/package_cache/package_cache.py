from __future__ import annotations

import os
import pathlib
from typing import Optional

from rattler.networking.client import Client
from rattler.repo_data.record import RepoDataRecord

from rattler.rattler import PackageCache as _PackageCache


class PackageCache:
    """A rattler-shaped package cache directory.

    Wraps ``rattler_cache::package_cache::PackageCache``. Populating a directory
    with this type produces a layout that ``rattler_cache::PackageCache`` can
    consume directly: ``<name>-<version>-<build>/`` subdirectories containing
    the extracted package and a cache-metadata file.
    """

    def __init__(self, cache_dir: os.PathLike[str] | str) -> None:
        self._inner = _PackageCache(os.fspath(cache_dir))

    async def get_or_fetch_from_url(
        self,
        record: RepoDataRecord,
        url: str,
        client: Optional[Client] = None,
    ) -> pathlib.Path:
        """Fetch (if missing) and extract the package identified by ``record``
        into this cache, returning the path to the extracted directory.
        """
        path = await self._inner.get_or_fetch_from_url(
            record,
            url,
            client._client if client is not None else None,
        )
        return pathlib.Path(path)
