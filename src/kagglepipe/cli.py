"""Top-level argparse CLI for kagglepipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kagglepipe import __version__, bundle as bundle_mod, cache as cache_mod, config as cfg_mod
from kagglepipe.commands import (
    auth as auth_cmd,
    competitions as comp_cmd,
    config_cmd,
    datasets as ds_cmd,
    experiments as exp_cmd,
    feature as feature_cmd,
    features_reg as features_cmd,
    graph as graph_cmd,
    kernels as kernels_cmd,
    lineage as lineage_cmd,
    retry as retry_cmd,
    src as src_cmd,
    status as status_cmd,
    submissions as submissions_cmd,
    templates as templates_cmd,
    validate as validate_cmd,
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
    p_cfg_show = cfg_sub.add_parser("show", help="Print effective config")
    p_cfg_show.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON.",
    )
    cfg_sub.add_parser("path", help="Print the path kagglepipe will load")

    # --- src ---
    p_src = sub.add_parser("src", help="Source dataset operations")
    src_sub = p_src.add_subparsers(dest="src_cmd", required=True, metavar="SRC_CMD")
    p_up = src_sub.add_parser("upload", help="Package and upload source dataset")
    p_up.add_argument("--version", type=int, default=None)
    p_up.add_argument("--src-root", type=Path, default=None)
    p_up.add_argument("--slug", default=None)
    p_up.add_argument(
        "--dry-run", action="store_true",
        help="P9: print the plan and build the tarball locally, but do not upload.",
    )

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
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="P9: print the plan without rendering/pushing/polling/downloading.",
    )
    p_all = feat_sub.add_parser("all", help="Run heavy branches sequentially")
    p_all.add_argument("--branches", default=None, help="Comma-separated override")
    p_all.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_all.add_argument("--timeout", type=int, default=None)
    p_all.add_argument("--data-dataset", default=None)
    p_all.add_argument("--features-dir", type=Path, default=None)
    p_all.add_argument("--notebooks-dir", type=Path, default=None)
    p_all.add_argument(
        "--parallel", type=int, default=1,
        help="Number of concurrent kernel workers (P1). 1 = sequential.",
    )
    p_all.add_argument(
        "--resume", action="store_true",
        help="Skip branches whose last run completed with an artifact (P2).",
    )
    p_retry = feat_sub.add_parser("retry", help="Retry failed branches (P2)")
    p_retry.add_argument(
        "selector", nargs="?",
        help="What to retry: 'failed' (default), 'error', 'timeout', 'incomplete', 'all', or a single branch name.",
    )
    p_retry.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_retry.add_argument("--timeout", type=int, default=None)
    p_retry.add_argument("--parallel", type=int, default=1)
    p_resume = feat_sub.add_parser("resume", help="Resume a run, skipping completed branches (P2)")
    p_resume.add_argument("--branches", default=None)
    p_resume.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_resume.add_argument("--timeout", type=int, default=None)
    p_resume.add_argument("--parallel", type=int, default=1)
    p_build = feat_sub.add_parser(
        "build", help="Build a feature, running its declared dependency closure first (P4).",
    )
    p_build.add_argument("target", help="Target feature name")
    p_build.add_argument("--gpu", default=None, choices=["p100", "t4x2", "none"])
    p_build.add_argument("--timeout", type=int, default=None)
    p_build.add_argument("--data-dataset", default=None)
    p_build.add_argument("--parallel", type=int, default=1)
    p_plan = feat_sub.add_parser(
        "plan", help="Print the dependency plan for a target (P4) without running.",
    )
    p_plan.add_argument("target")

    # --- status ---
    p_status = sub.add_parser("status", help="List kernels matching the configured prefix")
    p_status.add_argument("--all", action="store_true", help="Show all my kernels")
    p_status.add_argument("--csv", action="store_true")

    # P10: pre-flight validation.
    p_val = sub.add_parser("validate", help="Pre-flight checks (P10).")
    p_val.add_argument("--json", dest="json_output", action="store_true")

    # P12: template library.
    p_tpl = sub.add_parser("template", help="Project templates (P12).")
    tpl_sub = p_tpl.add_subparsers(dest="tpl_cmd", required=True, metavar="TPL_CMD")
    p_tpl_list = tpl_sub.add_parser("list")
    p_tpl_init = tpl_sub.add_parser("init", help="Scaffold a project from a template.")
    p_tpl_init.add_argument("template", help="Template name (tabular|cv|nlp)")
    p_tpl_init.add_argument("--name", default=None, help="Project name (default: dir basename)")
    p_tpl_init.add_argument("--root", type=Path, default=None, help="Target dir (default: cwd)")
    p_tpl_init.add_argument("--force", action="store_true", help="Overwrite existing files")

    # P14: reproducibility bundles.
    p_run = sub.add_parser("run", help="Run reproducibility (P14).")
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True, metavar="RUN_CMD")
    p_run_export = run_sub.add_parser("export", help="Export a run as a portable tarball.")
    p_run_export.add_argument("target", help="Branch name or path to a manifest.json")
    p_run_export.add_argument("--out", type=Path, default=None)
    p_run_export.add_argument("--no-artifacts", action="store_true",
                              help="Don't include the artifact file in the bundle.")
    p_run_reproduce = run_sub.add_parser("reproduce", help="Reproduce a run from a bundle.")
    p_run_reproduce.add_argument("bundle", type=Path)
    p_run_reproduce.add_argument(
        "--no-dry-run", action="store_true",
        help="Actually re-execute (default: print the plan only).",
    )

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

    # --- submit (P3) ---
    p_sub = sub.add_parser(
        "submit", help="Submit a file to a competition (P3). Reads [competition] from kaggle.toml.",
    )
    p_sub.add_argument("--competition", default=None)
    p_sub.add_argument("--file", type=Path, default=None)
    p_sub.add_argument("--message", "-m", default=None)
    p_sub.add_argument(
        "--train", action="store_true",
        help="Run [competition].train_command before submitting.",
    )
    p_sub.add_argument(
        "--experiment-id", default=None,
        help="Link this submission to an experiment id (P11.5).",
    )

    # --- submissions (P3) ---
    p_subs = sub.add_parser("submissions", help="Local history of competition submissions (P3).")
    subs_sub = p_subs.add_subparsers(dest="subs_cmd", required=True, metavar="SUB_CMD")
    p_subs_list = subs_sub.add_parser("list")
    p_subs_list.add_argument("--competition", default=None)
    p_subs_list.add_argument("--csv", action="store_true")
    p_subs_list.add_argument("--json", dest="json_output", action="store_true")
    p_subs_latest = subs_sub.add_parser("latest")
    p_subs_latest.add_argument("competition", nargs="?")
    p_subs_watch = subs_sub.add_parser("watch", help="Poll for new submission scores (P11).")
    p_subs_watch.add_argument("competition")
    p_subs_watch.add_argument("--current", default=None)
    p_subs_watch.add_argument("--poll-sec", type=int, default=60)
    p_subs_watch.add_argument("--max-wait-sec", type=int, default=1800)
    p_subs_watch.add_argument("--json", dest="json_output", action="store_true")
    p_subs_best = subs_sub.add_parser("best", help="Show the best-scoring submission with full provenance (P11.5).")
    p_subs_best.add_argument("competition", nargs="?")
    p_subs_best.add_argument("--json", dest="json_output", action="store_true")
    p_subs_show = subs_sub.add_parser("show", help="Show full provenance for a submission id (P11.5).")
    p_subs_show.add_argument("submission_id")
    p_subs_show.add_argument("--json", dest="json_output", action="store_true")

    # --- leaderboard (P11) ---
    p_lb_top = sub.add_parser("leaderboard", help="Competition leaderboard helpers (P11).")
    lb_sub = p_lb_top.add_subparsers(dest="lb_cmd", required=True, metavar="LB_CMD")
    p_lb_latest = lb_sub.add_parser("latest")
    p_lb_latest.add_argument("competition")
    p_lb_latest.add_argument("--top", type=int, default=20)
    p_lb_latest.add_argument("--json", dest="json_output", action="store_true")

    # --- cache (P5) ---
    p_cache = sub.add_parser("cache", help="Artifact cache (P5).")
    cache_sub = p_cache.add_subparsers(dest="cache_cmd", required=True, metavar="CACHE_CMD")
    p_cache_status = cache_sub.add_parser("status")
    p_cache_status.add_argument("--json", dest="json_output", action="store_true")
    p_cache_clear = cache_sub.add_parser("clear")
    p_cache_clear.add_argument("branch", nargs="?")

    # --- experiments (P6) ---
    p_exp = sub.add_parser("experiments", help="Experiment tracking (P6).")
    exp_sub = p_exp.add_subparsers(dest="exp_cmd", required=True, metavar="EXP_CMD")
    p_exp_rec = exp_sub.add_parser("record")
    p_exp_rec.add_argument("--id", default=None)
    p_exp_rec.add_argument("--notes", default="")
    p_exp_rec.add_argument("--submission-id", default=None)
    p_exp_rec.add_argument("--score", type=float, default=None)
    p_exp_rec.add_argument("--feature-branches", default=None)
    p_exp_list = exp_sub.add_parser("list")
    p_exp_list.add_argument("--csv", action="store_true")
    p_exp_list.add_argument("--json", dest="json_output", action="store_true")
    p_exp_show = exp_sub.add_parser("show")
    p_exp_show.add_argument("exp_id")

    # --- features (P7) ---
    p_feats = sub.add_parser("features", help="Local feature registry (P7).")
    feats_sub = p_feats.add_subparsers(dest="feats_cmd", required=True, metavar="FEAT_CMD")
    p_feats_list = feats_sub.add_parser("list")
    p_feats_list.add_argument("--csv", action="store_true")
    p_feats_list.add_argument("--json", dest="json_output", action="store_true")
    p_feats_show = feats_sub.add_parser("show")
    p_feats_show.add_argument("name")

    # --- lineage (P8) ---
    p_lin = sub.add_parser("lineage", help="Dataset lineage (P8).")
    lin_sub = p_lin.add_subparsers(dest="lin_cmd", required=True, metavar="LIN_CMD")
    p_lin_show = lin_sub.add_parser("show")
    p_lin_show.add_argument("feature")
    p_lin_show.add_argument("--json", dest="json_output", action="store_true")
    p_lin_add = lin_sub.add_parser("add-parent")
    p_lin_add.add_argument("child")
    p_lin_add.add_argument("parent")
    p_lin_rm = lin_sub.add_parser("remove")
    p_lin_rm.add_argument("feature")

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
            return config_cmd.show(json_output=getattr(args, "json_output", False))
        if args.cfg_cmd == "path":
            return config_cmd.path()
    if args.cmd == "src":
        if args.src_cmd == "upload":
            return src_cmd.upload(
                cfg,
                src_root=args.src_root,
                version=args.version,
                slug=args.slug,
                dry_run=getattr(args, "dry_run", False),
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
                dry_run=getattr(args, "dry_run", False),
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
                parallel=getattr(args, "parallel", 1),
                resume=getattr(args, "resume", False),
            )
        if args.feat_cmd == "retry":
            selector = args.selector or "failed"
            return retry_cmd.cmd_retry(
                cfg,
                selector,
                gpu=gpu,
                timeout_sec=args.timeout,
                parallel=getattr(args, "parallel", 1),
            )
        if args.feat_cmd == "resume":
            branches = args.branches.split(",") if args.branches else None
            return retry_cmd.cmd_resume(
                cfg,
                branches=branches,
                gpu=gpu,
                timeout_sec=args.timeout,
                parallel=getattr(args, "parallel", 1),
            )
        if args.feat_cmd == "build":
            return graph_cmd.cmd_feature_build(
                cfg,
                args.target,
                gpu=gpu,
                parallel=getattr(args, "parallel", 1),
                timeout_sec=args.timeout,
                data_dataset=args.data_dataset,
            )
        if args.feat_cmd == "plan":
            return graph_cmd.cmd_feature_plan(cfg, args.target)
    if args.cmd == "status":
        return status_cmd.status(cfg, all_kernels=args.all, csv_output=args.csv)
    if args.cmd == "validate":
        return validate_cmd.cmd_validate(json_output=getattr(args, "json_output", False))
    if args.cmd == "template":
        if args.tpl_cmd == "list":
            return templates_cmd.cmd_template_list()
        if args.tpl_cmd == "init":
            return templates_cmd.cmd_template_init(
                args.template,
                project_name=args.name,
                root=args.root,
                force=args.force,
            )
    if args.cmd == "run":
        if args.run_cmd == "export":
            return bundle_mod.cmd_run_export(
                args.target,
                out=args.out,
                include_artifacts=not args.no_artifacts,
            )
        if args.run_cmd == "reproduce":
            return bundle_mod.cmd_run_reproduce(
                args.bundle,
                dry_run=not args.no_dry_run,
            )
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
    if args.cmd == "submit":
        return submissions_cmd.cmd_submit(
            cfg,
            competition=args.competition,
            file=args.file,
            message=args.message,
            train=args.train,
            experiment_id=args.experiment_id,
        )
    if args.cmd == "submissions":
        if args.subs_cmd == "list":
            return submissions_cmd.cmd_submissions_list(
                competition=args.competition,
                csv_output=args.csv,
                json_output=args.json_output,
            )
        if args.subs_cmd == "latest":
            return submissions_cmd.cmd_submissions_latest(args.competition)
        if args.subs_cmd == "watch":
            return submissions_cmd.cmd_submissions_watch(
                args.competition,
                current=args.current,
                poll_sec=args.poll_sec,
                max_wait_sec=args.max_wait_sec,
                json_output=args.json_output,
            )
        if args.subs_cmd == "best":
            return submissions_cmd.cmd_submissions_best(
                args.competition, json_output=args.json_output
            )
        if args.subs_cmd == "show":
            return submissions_cmd.cmd_submissions_show(
                args.submission_id, json_output=args.json_output
            )
    if args.cmd == "leaderboard":
        if args.lb_cmd == "latest":
            return submissions_cmd.cmd_leaderboard_latest(
                args.competition, top=args.top, json_output=args.json_output
            )
    if args.cmd == "cache":
        if args.cache_cmd == "status":
            return cache_mod.cmd_cache_status(json_output=args.json_output)
        if args.cache_cmd == "clear":
            return cache_mod.cmd_cache_clear(args.branch)
    if args.cmd == "experiments":
        if args.exp_cmd == "record":
            fbs = args.feature_branches.split(",") if args.feature_branches else None
            return exp_cmd.cmd_experiments_record(
                cfg,
                id=args.id,
                notes=args.notes,
                submission_id=args.submission_id,
                score=args.score,
                feature_branches=fbs,
            )
        if args.exp_cmd == "list":
            return exp_cmd.cmd_experiments_list(
                csv_output=args.csv, json_output=args.json_output
            )
        if args.exp_cmd == "show":
            return exp_cmd.cmd_experiments_show(args.exp_id)
    if args.cmd == "features":
        if args.feats_cmd == "list":
            return features_cmd.cmd_features_list(
                csv_output=args.csv, json_output=args.json_output
            )
        if args.feats_cmd == "show":
            return features_cmd.cmd_features_show(args.name)
    if args.cmd == "lineage":
        if args.lin_cmd == "show":
            return lineage_cmd.cmd_lineage(args.feature, json_output=args.json_output)
        if args.lin_cmd == "add-parent":
            return lineage_cmd.cmd_lineage_add_parent(args.child, args.parent)
        if args.lin_cmd == "remove":
            return lineage_cmd.cmd_lineage_remove(args.feature)
    parser.error(f"unhandled: {args}")
    return 2  # unreachable
