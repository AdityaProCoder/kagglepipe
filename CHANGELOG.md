# Changelog

All notable changes are documented here. This project follows semantic versioning.

## 0.2.0 - 2026-07-16

### Added

- GitHub Actions CI for Linux and Windows on Python 3.11 and 3.12, including release-package verification.
- Contributor and security guidance.
- Tests for offline validation, unreadable credential files, and custom credential paths.

### Changed

- Declared the Kaggle CLI as a runtime dependency, so normal installs can execute KagglePipe commands.
- Reworked the README around a concise installation-to-first-run workflow.
- `validate` now continues local configuration checks before credentials are configured.
- `auth login --path` now writes and reads the requested credentials file consistently.
- Updated package metadata and the CLI to version 0.2.0.

### Fixed

- Credential-file permission errors now produce a clear `CredentialsError` instead of an uncaught OS exception.
