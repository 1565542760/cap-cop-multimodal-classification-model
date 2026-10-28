"""Check dependencies before starting a model pipeline; never install or alter them."""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys


PROFILES = {
    "phenotype": ["numpy", "pandas", "SimpleITK", "pydicom"],
    "clinical": ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost"],
    "segmentation": ["numpy", "pandas", "torch", "SimpleITK", "pydicom", "PyYAML"],
    "representations": ["numpy", "pandas", "scipy", "scikit-learn", "torch", "timm", "SimpleITK", "pydicom"],
    "radiomics": ["numpy", "pandas", "scipy", "scikit-learn", "pyradiomics", "SimpleITK", "joblib"],
    "exploratory": ["numpy", "pandas", "scikit-learn", "torch", "monai", "SimpleITK", "joblib"],
    "plotting": ["numpy", "pandas", "matplotlib"],
}
MODULES = {"scikit-learn": "sklearn", "pyradiomics": "radiomics", "PyYAML": "yaml"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--import-check", action="store_true", help="Actually import each dependency in an isolated subprocess to detect binary/DLL failures.")
    args = parser.parse_args()
    packages = {}
    missing = []
    import_failures = []
    for package in PROFILES[args.profile]:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        found = importlib.util.find_spec(MODULES.get(package, package)) is not None
        packages[package] = {"version": version, "module_found": found}
        if version is None or not found:
            missing.append(package)
        elif args.import_check:
            module = MODULES.get(package, package)
            try:
                result = subprocess.run(
                    [sys.executable, "-B", "-c", f"import {module}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                )
                packages[package]["import_exit_code"] = result.returncode
                if result.returncode:
                    packages[package]["import_error"] = result.stderr
                    import_failures.append(package)
            except subprocess.TimeoutExpired:
                packages[package]["import_error"] = "Import timed out after 120 seconds"
                import_failures.append(package)
    print(json.dumps({"python": sys.version, "executable": sys.executable,
                      "profile": args.profile, "packages": packages, "missing": missing,
                      "import_failures": import_failures,
                      "check": "actual dependency imports" if args.import_check else "package discovery only; use --import-check for binary/DLL checks"}, indent=2))
    return 1 if missing or import_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
