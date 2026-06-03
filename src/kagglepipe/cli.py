"""Top-level argparse CLI for kagglepipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kagglepipe import __version__, config as cfg_mod
from kagglepipe.commands import (
    auth as auth_cmd,
    competitions as comp_cmd,
    config_cmd,
    datasets as ds_cmd,
    feature as feature_cmd,
    kernels as kernels_cmd,
    src as src_cmd,
    status as status_cmd,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kagglepipe",
        description=(
            "Full terminal control over Kaggle. "
            "Thin, configurable orchestrator on top of the official kaggle CLI."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"kagglepipe {__version__}"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to kaggle.toml (default: ./kaggle.toml)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # --- auth ---
    sub.add_parser("whoami", help="Print the current Kaggle username")

    p_login = sub.add_parser("login", help="Bootstrap ~/.kaggle/kaggle.json")
    p_login.add_argument("--username", help="Kaggle username")
    p_login.add_argument("--key", help="Kaggle API key (omit to prompt)")
    p_login.add_argument(
        "--path", type=Path, default=None,
        help="Override credentials file path (default: ~/.kaggle/kaggle.json)",
    )

    # --- config ---
    p_cfg = sub.add_parser("config", help="Manage kaggle.toml")
    cfg_sub = p_cfg.add_subparsers(dest="cfg_cmd", required=True, metavar="CONFIG_CMD")
    p_cfg_init = cfg_sub.add_parser("init", help="Scaffold kaggle.toml in cwd")
    p_cfg_init.add_argument("--path", type=Path, default=None)
    p_cfg_init.add_argument("--name", default=None, help="Project name (default: dir basename)")
    p_cfg_init.add_argument("--force", action="store_true")
    cfg_sub.add_parser("show", help="Print effective config")
    cfg_sub.add_parser("path", help="Print the path kagglepipe will load")

    # --- src ---
    p_src = sub.add_parser("src", help="Source dataset operations")
    src_sub = p_src.add_subparsers(dest="src_cmd", required=True, metavar="SRC_CMD")
    p_up = src_sub.add_parser("upload", help="Package and upload source dataset")
    p_up.add_argument("--version", type=int, default=None)
    p_up.add_argument("--src-root", type=Path, default=None)
    p_up.add_argument("--slug", default=None)

    # --- feature ---
    p_feat = sub.add_parser("feature", help="Run feature branches on Kaggle")
    feat_sub = p_feat.add_subparsers(dest="feat_cmd", required=True, metavar="FEATURE_CMD")
    p_run = feat_sub.add_parser("run", help="Render+push+wait+download one branch")
    p_run.add_argument("branch")
    p_run.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_run.add_argument("--timeout", type=int, default=None)
    p_run.add_argument("--src-dataset", default=None)
    p_run.add_argument("--data-dataset", default=None)
    p_run.add_argument("--src-version", type=int, default=None)
    p_run.add_argument("--features-dir", type=Path, default=None)
    p_run.add_argument("--notebooks-dir", type=Path, default=None)
    p_run.add_argument("--no-download", action="store_true")
    p_all = feat_sub.add_parser("all", help="Run heavy branches sequentially")
    p_all.add_argument("--branches", default=None, help="Comma-separated override")
    p_all.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_all.add_argument("--timeout", type=int, default=None)
    p_all.add_argument("--data-dataset", default=None)
    p_all.add_argument("--features-dir", type=Path, default=None)
    p_all.add_argument("--notebooks-dir", type=Path, default=None)

    # --- status ---
    p_status = sub.add_parser("status", help="List kernels matching the configured prefix")
    p_status.add_argument("--all", action="store_true", help="Show all my kernels")
    p_status.add_argument("--csv", action="store_true")

    # --- kernels ---
    p_kern = sub.add_parser("kernels", help="Kernel operations")
    kern_sub = p_kern.add_subparsers(dest="kern_cmd", required=True, metavar="KERNEL_CMD")
    p_klist = kern_sub.add_parser("list")
    p_klist.add_argument("--user", default=None)
    p_klist.add_argument("--search", default=None)
    p_klist.add_argument("--page-size", type=int, default=20)
    p_klist.add_argument("--csv", action="store_true")
    p_klist.add_argument("--json", dest="json_output", action="store_true")
    p_kstat = kern_sub.add_parser("status")
    p_kstat.add_argument("slug")
    p_kout = kern_sub.add_parser("output")
    p_kout.add_argument("slug")
    p_kout.add_argument("--path", type=Path, default=None)
    p_klogs = kern_sub.add_parser("logs")
    p_klogs.add_argument("slug")
    p_kstop = kern_sub.add_parser("stop")
    p_kstop.add_argument("slug")
    p_kpush = kern_sub.add_parser("push")
    p_kpush.add_argument("path", type=Path)

    # --- datasets ---
    p_ds = sub.add_parser("datasets", help="Dataset operations")
    ds_sub = p_ds.add_subparsers(dest="ds_cmd", required=True, metavar="DATASET_CMD")
    p_dlist = ds_sub.add_parser("list")
    p_dlist.add_argument("--user", default=None)
    p_dlist.add_argument("--search", default=None)
    p_dlist.add_argument("--csv", action="store_true")
    p_dlist.add_argument("--json", dest="json_output", action="store_true")
    p_dver = ds_sub.add_parser("versions")
    p_dver.add_argument("slug")
    p_dget = ds_sub.add_parser("get")
    p_dget.add_argument("slug")
    p_dget.add_argument("path", type=Path)
    p_dcreate = ds_sub.add_parser("create")
    p_dcreate.add_argument("path", type=Path)
    p_dcreate.add_argument("--public", action="store_true")
    p_dv = ds_sub.add_parser("version")
    p_dv.add_argument("path", type=Path)
    p_dv.add_argument("--message", "-m", required=True)
    p_dv.add_argument("--dir-mode", "-r", default="zip", choices=["zip", "tar"])

    # --- competitions ---
    p_comp = sub.add_parser("competitions", help="Competition operations")
    comp_sub = p_comp.add_subparsers(dest="comp_cmd", required=True, metavar="COMP_CMD")
    p_clist = comp_sub.add_parser("list")
    p_clist.add_argument("--csv", action="store_true")
    p_clist.add_argument("--json", dest="json_output", action="store_true")
    p_cfiles = comp_sub.add_parser("files")
    p_cfiles.add_argument("competition")
    p_csubmit = comp_sub.add_parser("submit")
    p_csubmit.add_argument("competition")
    p_csubmit.add_argument("file", type=Path)
    p_csubmit.add_argument("--message", "-m", required=True)
    p_clb = comp_sub.add_parser("leaderboard")
    p_clb.add_argument("competition")
    p_clb.add_argument("--top", type=int, default=20)
    p_clb.add_argument("--csv", action="store_true")
    p_clb.add_argument("--json", dest="json_output", action="store_true")

    return parser


def _add_global_flags(_parser: argparse.ArgumentParser) -> None:
    """Reserved for future --json/--quiet globals."""
    pass


def main(argv: list[str] | None = None) -> int:
    """Top-level entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = cfg_mod.load(args.config) if args.config else cfg_mod.load()

    # --- dispatch ---
    if args.cmd == "whoami":
        return auth_cmd.whoami()
    if args.cmd == "login":
        return auth_cmd.login(username=args.username, key=args.key, path=args.path)
    if args.cmd == "config":
        if args.cfg_cmd == "init":
            return config_cmd.init(args.path, project_name=args.name, force=args.force)
        if args.cfg_cmd == "show":
            return config_cmd.show()
        if args.cfg_cmd == "path":
            return config_cmd.path()
    if args.cmd == "src":
        if args.src_cmd == "upload":
            return src_cmd.upload(
                cfg,
                src_root=args.src_root,
                version=args.version,
                slug=args.slug,
            )
    if args.cmd == "feature":
        gpu = args.gpu or cfg.feature.default_gpu
        if args.feat_cmd == "run":
            return feature_cmd.run_feature(
                cfg,
                args.branch,
                gpu=gpu,
                timeout_sec=args.timeout,
                src_dataset=args.src_dataset,
                src_version=args.src_version,
                data_dataset=args.data_dataset,
                features_dir=args.features_dir,
                notebooks_dir=args.notebooks_dir,
                no_download=args.no_download,
            )
        if args.feat_cmd == "all":
            branches = args.branches.split(",") if args.branches else None
            return feature_cmd.run_all(
                cfg,
                branches=branches,
                gpu=gpu,
                timeout_sec=args.timeout,
                data_dataset=args.data_dataset,
                features_dir=args.features_dir,
                notebooks_dir=args.notebooks_dir,
            )
    if args.cmd == "status":
        return status_cmd.status(cfg, all_kernels=args.all, csv_output=args.csv)
    if args.cmd == "kernels":
        if args.kern_cmd == "list":
            return kernels_cmd.list_kernels(
                user=args.user, search=args.search, page_size=args.page_size,
                csv_output=args.csv, json_output=args.json_output,
            )
        if args.kern_cmd == "status":
            return kernels_cmd.kernel_status(args.slug)
        if args.kern_cmd == "output":
            return kernels_cmd.kernel_output(args.slug, path=args.path)
        if args.kern_cmd == "logs":
            return kernels_cmd.kernel_logs(args.slug)
        if args.kern_cmd == "stop":
            return kernels_cmd.kernel_stop(args.slug)
        if args.kern_cmd == "push":
            return kernels_cmd.kernel_push(args.path)
    if args.cmd == "datasets":
        if args.ds_cmd == "list":
            return ds_cmd.list_datasets(
                user=args.user, search=args.search,
                csv_output=args.csv, json_output=args.json_output,
            )
        if args.ds_cmd == "versions":
            return ds_cmd.dataset_versions(args.slug)
        if args.ds_cmd == "get":
            return ds_cmd.dataset_get(args.slug, args.path)
        if args.ds_cmd == "create":
            return ds_cmd.dataset_create(args.path, public=args.public)
        if args.ds_cmd == "version":
            return ds_cmd.dataset_version(args.path, message=args.message, dir_mode=args.dir_mode)
    if args.cmd == "competitions":
        if args.comp_cmd == "list":
            return comp_cmd.list_competitions(
                csv_output=args.csv, json_output=args.json_output,
            )
        if args.comp_cmd == "files":
            return comp_cmd.competition_files(args.competition)
        if args.comp_cmd == "submit":
            return comp_cmd.competition_submit(
                args.competition, args.file, args.message,
            )
        if args.comp_cmd == "leaderboard":
            return comp_cmd.competition_leaderboard(
                args.competition, top=args.top,
                csv_output=args.csv, json_output=args.json_output,
            )
    parser.error(f"unhandled: {args}")
    return 2  # unreachable
