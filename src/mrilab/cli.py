"""mrilab CLI (Typer)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from mrilab.settings import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="The Model MRI backend.")


@app.command()
def serve(host: str | None = None, port: int | None = None) -> None:
    """Run the API server (bind 0.0.0.0 on WSL so the Windows browser reaches it)."""
    import uvicorn
    s = get_settings()
    uvicorn.run("mrilab.api.app:app", host=host or s.api_host,
                port=port or s.api_port)


@app.command()
def worker(lane: str = typer.Argument(..., help="gpu | cpu")) -> None:
    """Run a Huey consumer for one lane (gpu: 1 worker; cpu: 2 threads)."""
    from huey.consumer import Consumer

    from mrilab.engine.queue import queues
    if lane not in ("gpu", "cpu"):
        raise typer.BadParameter("lane must be gpu or cpu")
    workers = 1 if lane == "gpu" else 2
    Consumer(queues()[lane], workers=workers, worker_type="thread").run()


@app.command()
def run(instrument: str, repo: str = typer.Option(None), revision: str = typer.Option(None),
        path: str = typer.Option(None), dtype: str = typer.Option(...),
        params: str = typer.Option("{}", help="JSON dict of instrument params"),
        sync: bool = typer.Option(False, help="execute in-process instead of enqueueing")) -> None:
    """Submit (or run synchronously) one instrument job."""
    from mrilab.engine.jobs import JobSpec, Subject
    from mrilab.engine.queue import _run, submit
    spec = JobSpec(instrument=instrument,
                   subject=Subject(repo=repo, revision=revision, path=path),
                   dtype=dtype, params=json.loads(params))
    s = get_settings()
    if sync:
        # create the Job row first (skip enqueue), then run the full wrapper
        # in-process: skip-check -> thermal gate -> subprocess -> validate ->
        # fingerprint -> index
        job = submit(s, spec, enqueue=False)
        if job.state.value == "FAILED":
            typer.echo(f"{job.id}: BLOCKED — {job.error}")
            raise typer.Exit(1)
        outcome = _run(job.id, spec.model_dump_json())
        typer.echo(f"{job.id}: {outcome}")
        if outcome == "failed":
            raise typer.Exit(1)
    else:
        job = submit(s, spec)
        typer.echo(f"{job.id}: {job.state}")


@app.command()
def report(face: str, label: str = typer.Option(None), repo: str = typer.Option(None),
           revision: str = typer.Option(None), target: str = typer.Option(None),
           horizon_steps: int = typer.Option(None), loss: float = typer.Option(None),
           md: bool = typer.Option(True, help="print markdown")) -> None:
    """Build a report from stored rows and save it to the workspace."""
    from mrilab.engine.jobs import Subject
    from mrilab.faces import (arena as arena_face, certificate, common as kit,
                              mri_report, preflight, render_md)
    s = get_settings()
    if face == "model_report":
        rep = mri_report.build_from_stored(
            s, label or f"{repo.split('/')[-1]}@{revision or 'main'}",
            repo=repo, revision=revision)
    elif face == "preflight":
        rep = preflight.build(s, Subject(repo=repo, revision=revision),
                              target=target, horizon_steps=horizon_steps,
                              loss=loss)
    elif face == "release_cert":
        rep = certificate.build(s, label)
    elif face == "arena":
        rep = arena_face.build()
    else:
        raise typer.BadParameter(f"face {face} not CLI-wired; use the API")
    p = kit.save_report(rep, s.reports_dir)
    typer.echo(f"saved {p}")
    if md:
        typer.echo(render_md.render(rep))


@app.command()
def rescan() -> None:
    """Rebuild the row index from disk (the DB is a rebuildable index)."""
    from mrilab.engine.rowstore import rescan as do_rescan
    n = do_rescan(get_settings())
    typer.echo(f"indexed {n} new row(s)")


@app.command()
def fingerprint() -> None:
    """Check the research venv against runtime_lock.json."""
    from mrilab.engine import fingerprint as fp
    bad = fp.check(get_settings())
    if bad:
        typer.echo("MISMATCH:\n  " + "\n  ".join(bad))
        raise typer.Exit(1)
    typer.echo("research venv matches runtime_lock.json")


@app.command("vendor-diff")
def vendor_diff(research_root: str = typer.Option(
        "/home/jacopo/mri_compressor", help="research repo root")) -> None:
    """Detect upstream drift of the vendored instruments."""
    s = get_settings()
    script = s.code_root / "scripts" / "vendor.py"
    rc = subprocess.run([sys.executable, str(script), "--check",
                        "--research-root", research_root]).returncode
    raise typer.Exit(rc)


@app.command("pools-verify")
def pools_verify() -> None:
    """torch.load every cached pool in a research-venv subprocess; delete corrupt ones."""
    s = get_settings()
    pools = sorted(s.pools_dir.glob("_pool_*.pt")) + sorted(s.pools_dir.glob("_dp_pool_*.pt"))
    if not pools:
        typer.echo("no pools cached")
        return
    for p in pools:
        rc = subprocess.run(
            [str(s.gpu_python), "-c",
             f"import torch; torch.load({str(p)!r}, map_location='cpu'); print('ok')"],
            capture_output=True).returncode
        if rc == 0:
            typer.echo(f"ok      {p.name}")
        else:
            p.unlink()
            typer.echo(f"CORRUPT {p.name} — deleted (rebuilt on next use)")


@app.command()
def gc(min_gb: float = typer.Option(20.0, help="report repos larger than this")) -> None:
    """Report HF cache usage (weights are reclaimable cache; results are precious)."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    typer.echo(f"HF cache: {info.size_on_disk / 1e9:.1f} GB, {len(info.repos)} repos")
    for r in sorted(info.repos, key=lambda r: -r.size_on_disk):
        if r.size_on_disk >= min_gb * 1e9:
            typer.echo(f"  {r.size_on_disk / 1e9:6.1f} GB  {r.repo_id}")
    typer.echo("delete via huggingface-cli or scan_cache_dir().delete_revisions(...)")


if __name__ == "__main__":
    app()
