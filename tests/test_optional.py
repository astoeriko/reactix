import pytest

from reactix._optional import (
    _require_package,
    require_notebooks,
    require_pymc,
)


def test_require_package_existing():
    assert _require_package('math')


def test_require_package_missing():
    assert not _require_package('definitely_not_a_package')


def test_require_pymc_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        'reactix._optional._require_package',
        lambda package: package != 'nutpie',
    )

    with pytest.raises(ImportError, match='nutpie'):
        require_pymc()


def test_require_notebooks_missing(monkeypatch):
    monkeypatch.setattr(
        'reactix._optional._require_package',
        lambda package: package != 'matplotlib',
    )

    with pytest.raises(ImportError, match='matplotlib'):
        require_notebooks()
