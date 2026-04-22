use std::path::PathBuf;

use pyo3::{exceptions::PyValueError, pyclass, pymethods, Bound, PyAny, PyResult, Python};
use pyo3_async_runtimes::tokio::future_into_py;
use rattler_cache::package_cache::{CacheKey, PackageCache};
use rattler_conda_types::RepoDataRecord;
use rattler_networking::LazyClient;
use url::Url;

use crate::{
    error::PyRattlerError, networking::client::PyClientWithMiddleware, record::PyRecord,
};

/// A thin wrapper around `rattler_cache::package_cache::PackageCache` exposing
/// the ability to populate a rattler-shaped cache directory from Python.
#[pyclass(name = "PackageCache")]
#[derive(Clone)]
pub struct PyPackageCache {
    inner: PackageCache,
}

#[pymethods]
impl PyPackageCache {
    #[new]
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            inner: PackageCache::new(cache_dir),
        }
    }

    /// Fetch (if needed) and extract a package into the cache, returning the
    /// path to the extracted package directory.
    #[pyo3(signature = (record, url, client=None))]
    pub fn get_or_fetch_from_url<'a>(
        &self,
        py: Python<'a>,
        record: Bound<'a, PyAny>,
        url: String,
        client: Option<PyClientWithMiddleware>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let record: RepoDataRecord = PyRecord::try_from(record)?.try_into()?;
        let url = Url::parse(&url)
            .map_err(|e| PyValueError::new_err(format!("invalid url: {e}")))?;
        let cache_key: CacheKey = (&record.package_record).into();
        let client: LazyClient = client.map(Into::into).unwrap_or_default();
        let inner = self.inner.clone();

        future_into_py(py, async move {
            let metadata = inner
                .get_or_fetch_from_url(cache_key, url, client, None)
                .await
                .map_err(PyRattlerError::from)?;
            Ok(metadata.path().to_path_buf())
        })
    }
}
