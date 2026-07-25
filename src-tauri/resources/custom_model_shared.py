#!/usr/bin/env python3
"""Shared logic for custom YAML-defined models.

This module is imported by BOTH:
  - mlx_funasr_daemon.py  — to load + transcribe a custom-spec model
                             (long-running, no download/install actions)
  - custom_model_worker.py — to check / install dependencies / download
                             (one-shot per invocation, no model loading)

Splitting the long-running parts (load / transcribe) from the long-running-
but-blocking parts (download / install) lets the worker run in its own
process so that downloads cannot stall the transcription daemon.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Proxy env inheritance — GUI apps don't get shell profile vars
# ---------------------------------------------------------------------------

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "no_proxy", "NO_PROXY")


def _inherit_shell_proxy_env():
    """Inject proxy env vars from the user's login shell.

    macOS GUI apps launched via Finder/Spotlight inherit env from launchd,
    not the user's shell profile. huggingface_hub uses requests which reads
    http_proxy/https_proxy from os.environ. This function runs
    ``$SHELL -l -c 'env'`` to capture shell-profile proxy settings and
    injects them into os.environ (without overwriting existing values).
    """
    # Don't clobber existing env vars (e.g. set by the Rust side explicitly)
    missing = [v for v in _PROXY_VARS if v not in os.environ or not os.environ[v]]
    if not missing:
        return

    shell = os.environ.get("SHELL") or "/bin/bash"
    try:
        result = subprocess.run(
            [shell, "-l", "-i", "-c", "env"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return

    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key in missing:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Spec parsing and small utils
# ---------------------------------------------------------------------------

def parse_spec(spec_path: str) -> dict:
    """Read and YAML-parse a custom model spec file. Lazy yaml import."""
    import yaml
    with open(spec_path, "r") as f:
        return yaml.safe_load(f)


def split_pip_pkg_name(pkg_spec: str) -> str:
    """Extract the importable package name from a pip requirement spec.

    Normal specs: 'mimo_mlx>=0.1.0' -> 'mimo_mlx'
    Git URLs: 'git+https://github.com/Foo/bar.git' -> 'bar'
    """
    s = pkg_spec.strip()
    # Git URL: extract the repo name (last path segment, minus .git suffix).
    if s.startswith(("git+", "git@")) or "://" in s:
        tail = s.rsplit("/", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        return tail.replace("-", "_")
    return re.split(r"[<>=!~;\s]", s, 1)[0].strip()


def resolve_attr(module_name: str, dotted_path: str, custom_models_dir: str = ""):
    """Resolve 'pkg.submod.func' -> callable, after importing pkg.

    The first segment of the dotted path may equal the module name (so
    'mimo_mlx.load_asr' and 'load_asr' both work). Attribute lookup
    falls back to importing as a deeper submodule when an attribute is
    actually a module.

    If ``custom_models_dir`` is given and ``module_name`` matches a
    subdirectory containing ``__init__.py`` there, the directory is
    prepended to ``sys.path`` so a local plugin package (no pip install
    needed) can be imported just like an installed package.
    """
    import importlib

    # Local plugin package support: if the module name matches a directory
    # under custom_models_dir with an __init__.py, make it importable.
    if custom_models_dir:
        local_pkg = Path(custom_models_dir) / module_name / "__init__.py"
        if local_pkg.exists():
            parent = str(Path(custom_models_dir))
            if parent not in sys.path:
                sys.path.insert(0, parent)

    parts = dotted_path.split(".")
    if parts[0] == module_name:
        attrs = parts[1:]
    else:
        attrs = parts
    obj = importlib.import_module(module_name)
    for i, a in enumerate(attrs):
        try:
            obj = getattr(obj, a)
        except AttributeError:
            full = ".".join([module_name] + attrs[: i + 1])
            obj = importlib.import_module(full)
    return obj


def render_kwargs(
    kwargs: dict,
    voiceink_models_dir: str,
    paths: Optional[dict] = None,
    repo_dirs: Optional[list] = None,
) -> dict:
    """Replace {voiceink_models_dir}, {paths.X}, {repo_dirs[N]} in string values."""
    rendered = {}
    paths = paths or {}
    repo_dirs = repo_dirs or []
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = v.replace("{voiceink_models_dir}", voiceink_models_dir)
            for m in list(re.finditer(r"\{paths\.([^}]+)\}", v)):
                key = m.group(1)
                if key not in paths:
                    raise KeyError(f"paths.{key} not resolved (download must run first)")
                v = v.replace(m.group(0), paths[key])
            for m in list(re.finditer(r"\{repo_dirs\[(\d+)\]\}", v)):
                idx = int(m.group(1))
                if idx >= len(repo_dirs):
                    raise IndexError(f"repo_dirs[{idx}] out of range (only {len(repo_dirs)} repos)")
                v = v.replace(m.group(0), str(repo_dirs[idx]))
        rendered[k] = v
    return rendered


# ---------------------------------------------------------------------------
# Dependency check / install
# ---------------------------------------------------------------------------

def check_custom_dependencies(spec_path: str) -> dict:
    """Check whether the YAML spec's pip_packages are importable.

    Returns: {installed: [...], missing: [pkg_spec, ...], all_installed: bool}
    """
    import importlib.util
    spec = parse_spec(spec_path)
    pip_packages = spec.get("pip_packages")
    # No pip_packages declared — if python_module is a local plugin dir,
    # there's nothing to import-check. If python_module is a pip name, fall
    # back to it.
    if not pip_packages:
        pm = spec.get("python_module", "")
        if pm and not (Path(spec_path).parent / pm / "__init__.py").exists():
            pip_packages = [pm]
        else:
            return {"installed": [], "missing": [], "all_installed": True}

    installed, missing = [], []
    # User-declared import name overrides inference (handles git URLs where
    # repo name != package name, e.g. mano-asr.git installs as 'manoasr').
    import_name_override = spec.get("pip_import_name")
    for pkg_spec in pip_packages:
        if import_name_override:
            name = import_name_override
        else:
            name = split_pip_pkg_name(pkg_spec)
        if importlib.util.find_spec(name) is not None:
            installed.append(name)
        else:
            missing.append(pkg_spec)
    return {
        "installed": installed,
        "missing": missing,
        "all_installed": not missing,
    }


def install_custom_dependencies(spec_path: str) -> dict:
    """Run pip install for the spec's pip_packages.

    Returns: {success: bool, stdout: str, stderr: str, error: str|None}
    """
    spec = parse_spec(spec_path)
    pkgs = spec.get("pip_packages")
    # Local plugin packages (no pip_packages) need no installation.
    if not pkgs:
        return {"success": True, "stdout": "", "stderr": "", "error": None}
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None,
        }
    return {
        "success": False,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.stderr.strip() or "pip install failed",
    }


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------

def download_custom_model(
    spec_path: str,
    voiceink_models_dir: str,
    custom_models_dir: str,
) -> dict:
    """Run the download step declared in the YAML spec; write sidecar paths.json."""
    spec = parse_spec(spec_path)
    download = spec.get("download")
    if not download:
        # Write an empty sidecar so load_custom_model doesn't fail on the
        # paths.json existence check. Local plugin packages typically have
        # no download step but still go through the load path.
        cache_dir = Path(custom_models_dir) / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sidecar = cache_dir / f"{spec['id']}.paths.json"
        sidecar.write_text("{}")
        return {"success": True, "paths": {}, "note": "no download step declared"}

    paths_out: dict = {}
    if "function" in download:
        func = resolve_attr(spec["python_module"], download["function"], custom_models_dir)
        rendered = render_kwargs(download.get("kwargs", {}), voiceink_models_dir)
        result = func(**rendered)
        returns = download.get("returns", "tuple")
        if returns == "tuple":
            names = download.get("path_names", [])
            if len(names) != len(result):
                raise ValueError(
                    f"path_names has {len(names)} entries but function returned {len(result)} values"
                )
            paths_out = {n: str(p) for n, p in zip(names, result)}
        elif returns == "dict":
            paths_out = {k: str(v) for k, v in result.items()}
        elif returns == "path":
            names = download.get("path_names", ["model_dir"])
            paths_out = {names[0]: str(result)}
        else:
            raise ValueError(f"unknown returns kind: {returns}")
    elif "hf_repos" in download:
        from huggingface_hub import snapshot_download
        # Allow per-spec or env-var HF endpoint override (e.g. China mirror).
        endpoint = download.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        # GUI apps launched from Finder/Spotlight don't inherit shell profile
        # env vars (http_proxy, https_proxy, all_proxy). Read them from the
        # user's login shell so huggingface_hub can use the proxy.
        _inherit_shell_proxy_env()
        repo_dirs = []
        for repo in download["hf_repos"]:
            sanitized = repo.replace("/", "--")
            local_dir = Path(voiceink_models_dir) / f"custom-{sanitized}"
            snapshot_download(repo, local_dir=str(local_dir))
            repo_dirs.append(str(local_dir))

        raw_paths = download.get("paths", {})
        if not raw_paths:
            paths_out = {f"repo_{i}": p for i, p in enumerate(repo_dirs)}
        else:
            paths_out = render_kwargs(raw_paths, voiceink_models_dir, repo_dirs=repo_dirs)
            paths_out = {k: str(v) for k, v in paths_out.items()}
    else:
        raise ValueError("download must declare either 'function' or 'hf_repos'")

    cache_dir = Path(custom_models_dir) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar = cache_dir / f"{spec['id']}.paths.json"
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(paths_out, indent=2))
    tmp.replace(sidecar)

    return {"success": True, "paths": paths_out}


def load_custom_model(spec_path: str, voiceink_models_dir: str, custom_models_dir: str):
    """Load a custom-spec model: read sidecar, render kwargs, call load.function."""
    spec = parse_spec(spec_path)
    sidecar = Path(custom_models_dir) / ".cache" / f"{spec['id']}.paths.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"paths.json sidecar missing at {sidecar} — run download_custom_model first"
        )
    paths = json.loads(sidecar.read_text())

    load = spec["load"]
    rendered = render_kwargs(load.get("kwargs", {}), voiceink_models_dir, paths=paths)

    func = resolve_attr(spec["python_module"], load["function"], custom_models_dir)
    model = func(**rendered)
    model._daemon_model_type = "custom"
    model._daemon_custom_spec = spec
    return model
