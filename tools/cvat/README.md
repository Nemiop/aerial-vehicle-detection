# CVAT dependency

The CVAT source tree is intentionally not vendored in this repository. Clone the official CVAT repository into this directory before using `scripts/start_cvat.ps1`, `scripts/stop_cvat.ps1` or CVAT upload utilities:

```powershell
git clone https://github.com/cvat-ai/cvat.git tools/cvat
```

The project scripts expect this conventional local path only; CVAT data and Docker volumes remain outside version control.
