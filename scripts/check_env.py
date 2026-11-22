#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-1 环境自检脚本（微缺智学）。

只做检查与记录：不安装、不升级、不卸载任何依赖，不修改系统配置。

用法（必须使用项目虚拟环境的解释器）：
    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\check_env.py

输出：
    experiments/results/environment_report.json

说明：
  * 若报告中 matplotlib 缓存目录不可写，会优先把 MPLCONFIGDIR 指向项目内
    可写的 logs/mpl_cache（仅在本进程内生效，不改系统环境变量）。
  * 若目标报告已存在且不是本脚本生成的（缺少 generator 标记），
    会另存为带时间戳的新文件，避免覆盖他人内容。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime
from importlib import metadata
from pathlib import Path

GENERATOR = "scripts/check_env.py"
REPORT_SCHEMA = "microdefect.environment_report/1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VENV = PROJECT_ROOT / ".venv"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4060 Laptop GPU"

# (import 名, distribution 名, 是否本项目必需)
DEPENDENCIES = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("cv2", "opencv-python", True),
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("matplotlib", "matplotlib", True),
    ("sklearn", "scikit-learn", True),
    ("streamlit", "streamlit", True),
    ("PIL", "pillow", True),
    ("tqdm", "tqdm", True),
    ("yaml", "PyYAML", True),
]


def path_equal(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _probe_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def prepare_matplotlib_cache() -> dict:
    """在导入 matplotlib 之前确定可写缓存目录，不触碰系统目录权限。

    必须早于 `import matplotlib`：matplotlib 一旦导入就会尝试创建缓存目录并
    在失败时打印告警、退回临时目录。
    """
    info = {
        "env_before": os.environ.get("MPLCONFIGDIR"),
        "chosen": None,
        "created": False,
        "note": None,
        "default_writable": False,
    }

    env_value = os.environ.get("MPLCONFIGDIR")
    local_appdata = os.environ.get("LOCALAPPDATA")
    default_dir = (
        Path(local_appdata) / "matplotlib" if local_appdata else Path.home() / ".matplotlib"
    )
    info["default_cache_dir"] = str(default_dir)

    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(default_dir)
    for candidate in candidates:
        if _probe_writable(candidate):
            info["default_writable"] = True
            if env_value is None:
                os.environ["MPLCONFIGDIR"] = str(candidate)
            info["chosen"] = str(candidate)
            return info

    # 默认目录不可写：退到项目内可写目录（仅本进程内设置环境变量）
    for fallback in (
        PROJECT_ROOT / "logs" / "mpl_cache",
        PROJECT_ROOT / "scripts" / ".cache" / "mpl",
        Path(tempfile.gettempdir()) / "microdefect_mpl",
    ):
        if _probe_writable(fallback):
            os.environ["MPLCONFIGDIR"] = str(fallback)
            info["chosen"] = str(fallback)
            info["created"] = True
            info["note"] = (
                "matplotlib 默认缓存目录不可写，已在本进程内将 MPLCONFIGDIR 指向 "
                f"{fallback}（未修改系统环境变量或目录权限）。"
            )
            return info

    info["note"] = "未找到可写的 matplotlib 缓存目录。"
    return info


def check_dependencies() -> list[dict]:
    results = []
    for import_name, dist_name, required in DEPENDENCIES:
        record = {
            "import_name": import_name,
            "distribution": dist_name,
            "required": required,
            "import_ok": False,
            "module_version": None,
            "dist_version": None,
            "error": None,
        }
        try:
            record["dist_version"] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            record["dist_version"] = None
        except Exception as exc:  # pragma: no cover
            record["dist_version"] = f"<error: {type(exc).__name__}: {exc}>"

        try:
            module = __import__(import_name)
            record["import_ok"] = True
            record["module_version"] = getattr(module, "__version__", None)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
    return results


def check_torch() -> dict:
    info: dict = {
        "import_ok": False,
        "version": None,
        "cuda_compiled_version": None,
        "cudnn_version": None,
        "cuda_available": None,
        "device_count": None,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        info["import_ok"] = True
        info["version"] = torch.__version__
        info["cuda_compiled_version"] = torch.version.cuda
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            info["cudnn_version"] = None
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            for idx in range(info["device_count"]):
                props = torch.cuda.get_device_properties(idx)
                info["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gib": round(props.total_memory / 1024 ** 3, 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": int(props.multi_processor_count),
                    }
                )
            info["memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
            info["memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def check_torchvision() -> dict:
    info = {"import_ok": False, "version": None, "weights_enum_ok": None, "error": None}
    try:
        import torchvision

        info["import_ok"] = True
        info["version"] = torchvision.__version__
        try:
            from torchvision.models import ResNet18_Weights

            info["weights_enum_ok"] = [w.name for w in ResNet18_Weights]
        except Exception as exc:
            info["weights_enum_ok"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def resolve_out_path(path: Path, payload_check) -> tuple[Path, bool]:
    """避免覆盖非本脚本生成的文件。返回 (最终路径, 是否改用新名)。"""
    if not path.exists():
        return path, False
    try:
        with path.open("r", encoding="utf-8") as fh:
            old = json.load(fh)
        if payload_check(old):
            return path, False
    except Exception:
        pass
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}"), True


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P1-1 环境自检")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "experiments" / "results" / "environment_report.json"),
        help="报告输出路径（默认 experiments/results/environment_report.json）",
    )
    args = parser.parse_args()

    mpl_info = prepare_matplotlib_cache()

    sys_info = {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "module_file": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "expected_venv": str(EXPECTED_VENV),
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "path_python": shutil.which("python"),
        "path_pip": shutil.which("pip"),
        "env_VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        "env_CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
    }

    venv_ok = path_equal(Path(sys.prefix), EXPECTED_VENV) and path_equal(
        Path(sys.executable), EXPECTED_VENV / "Scripts" / "python.exe"
    )

    dep_results = check_dependencies()
    torch_info = check_torch()
    tv_info = check_torchvision()

    gpu_names = [d["name"] for d in torch_info.get("devices", [])]
    gpu_expected_ok = any(EXPECTED_GPU_NAME.lower() in n.lower() for n in gpu_names)
    cuda_ok = torch_info.get("cuda_available") is True
    deps_ok = all(d["import_ok"] for d in dep_results if d["required"])

    warnings: list[str] = []
    if not venv_ok:
        warnings.append(
            f"当前解释器不是项目虚拟环境：sys.prefix={sys.prefix}，期望={EXPECTED_VENV}"
        )
    if sys_info["path_python"] and not path_equal(
        Path(sys_info["path_python"]), Path(sys.executable)
    ):
        warnings.append(
            f"PATH 中的 python 指向 {sys_info['path_python']}，与当前解释器不同；"
            "在命令行直接使用 python/pip 会操作到其他环境。"
        )
    if not cuda_ok:
        warnings.append("torch.cuda.is_available() 不为 True，GPU 不可用。")
    if cuda_ok and not gpu_expected_ok:
        warnings.append(f"GPU 名称与预期不符：实际 {gpu_names}")
    if mpl_info.get("note"):
        warnings.append(mpl_info["note"])
    for dep in dep_results:
        if not dep["import_ok"]:
            warnings.append(f"依赖导入失败：{dep['import_name']} -> {dep['error']}")

    acceptance = {
        "use_project_venv": {"expected": str(EXPECTED_VENV), "actual": sys.prefix, "pass": venv_ok},
        "cuda_available": {"expected": True, "actual": cuda_ok, "pass": cuda_ok},
        "gpu_name": {
            "expected": EXPECTED_GPU_NAME,
            "actual": gpu_names,
            "pass": gpu_expected_ok,
        },
        "required_dependencies_importable": {"expected": True, "actual": deps_ok, "pass": deps_ok},
    }
    all_pass = all(v["pass"] for v in acceptance.values())

    payload = {
        "schema": REPORT_SCHEMA,
        "generator": GENERATOR,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python": sys_info,
        "venv_check": {"pass": venv_ok, "expected_prefix": str(EXPECTED_VENV)},
        "dependencies": dep_results,
        "torch": torch_info,
        "torchvision": tv_info,
        "matplotlib_cache": mpl_info,
        "warnings": warnings,
        "acceptance": acceptance,
        "all_pass": all_pass,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_path, renamed = resolve_out_path(
        out_path,
        lambda old: isinstance(old, dict) and old.get("generator") == GENERATOR,
    )
    with final_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print("=" * 66)
    print("微缺智学 P1-1 环境自检")
    print("=" * 66)
    print(f"sys.executable : {sys.executable}")
    print(f"sys.prefix     : {sys.prefix}")
    print(f"Python         : {platform.python_version()}")
    print(f"项目 .venv     : {'是' if venv_ok else '否'}")
    print("-" * 66)
    for dep in dep_results:
        if dep["import_ok"]:
            print(f"[OK]   {dep['import_name']:<12} {dep['dist_version'] or dep['module_version']}")
        else:
            print(f"[FAIL] {dep['import_name']:<12} {dep['error']}")
    print("-" * 66)
    if torch_info["import_ok"]:
        print(f"torch          : {torch_info['version']}  (CUDA {torch_info['cuda_compiled_version']})")
        print(f"cuDNN          : {torch_info['cudnn_version']}")
        print(f"cuda.is_available : {torch_info['cuda_available']}")
        print(f"torchvision    : {tv_info['version']}")
        for dev in torch_info["devices"]:
            print(
                f"GPU{dev['index']}          : {dev['name']}  "
                f"{dev['total_memory_gib']} GiB  cc {dev['compute_capability']}"
            )
    else:
        print(f"torch 导入失败 : {torch_info['error']}")
    print("-" * 66)
    for key, val in acceptance.items():
        print(f"{'PASS' if val['pass'] else 'FAIL'}  {key}")
    for warn in warnings:
        print(f"[warn] {warn}")
    print("-" * 66)
    print(f"报告已写入     : {final_path}" + ("（因原文件非本脚本生成，改用新文件名）" if renamed else ""))
    print(f"总体结果       : {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
