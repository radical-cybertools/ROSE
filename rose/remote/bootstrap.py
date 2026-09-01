"""rose.remote.bootstrap — `rose setup`: interactive first-time wizard for `rose run --remote`.

Guides a first-time user through getting `--remote` working end to end,
asking only simple questions (facility, account, names) and automating or
clearly explaining everything technical (broker TLS cert/token, IRI/SFAPI
credential files, the remote `~/.amsc/ve` environment).

The wizard's job ends once setup is verified working. It does not wire its
output into any workflow spec — it prints the resolved `remote:` block and
tells the user where it goes; wiring it into a specific workflow.yaml is
the user's own step.

Reuses the exact bootstrap/wait/teardown machinery `rose.remote.run_remote()`
uses for a real run (`_launch_iri_or_sfapi`, `_wait_for_endpoint`,
`_teardown`), so "verified" here means the same code path a real
`rose run --remote` takes, not a separate approximation of it.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from ..spec.schema import TargetConfig
from . import execute as _remote

ORBIT_DIR = Path.home() / ".radical" / "orbit"
CERT_PATH = ORBIT_DIR / "broker_cert.pem"
KEY_PATH = ORBIT_DIR / "broker_key.pem"
TOKEN_PATH = ORBIT_DIR / "broker.token"

_FACILITY_RESOURCE = {"nersc": "perlmutter", "olcf": "odo"}
_FACILITY_KIND = {"nersc": "sfapi", "olcf": "iri"}
_FACILITY_LOGIN_HOST = {
    "nersc": "perlmutter.nersc.gov",
    "olcf": "login1.frontier.olcf.ornl.gov",
}
_FACILITY_LABEL = {"nersc": "NERSC (Perlmutter)", "olcf": "OLCF (Frontier / Odo)"}


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  not a number: {raw!r} — try again")


def confirm(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        answer = input(f"{prompt}{suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  please answer y or n")


def choose(prompt: str, options: list[tuple[str, Any]]) -> Any:
    """`options`: list of (label, value).

    Returns the chosen value.
    """
    print(f"\n{prompt}\n")
    for i, (label, _) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    print()
    while True:
        raw = ask("Enter a number", "1")
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
        except ValueError:
            pass
        print(f"  please enter a number between 1 and {len(options)}")


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _step(text: str) -> None:
    print(f"\n— {text} —")


# ─────────────────────────────────────────────────────────────────────────────
#  Broker TLS cert / token — shared by both branches
# ─────────────────────────────────────────────────────────────────────────────


def _have_broker_credentials() -> bool:
    return CERT_PATH.exists() and KEY_PATH.exists() and TOKEN_PATH.exists()


def ensure_broker_credentials() -> bool:
    """Make sure the broker's TLS cert/key/token exist, offering to generate them automatically.

    Returns True once they're confirmed present.
    """
    if _have_broker_credentials():
        print(f"Broker certificate and token already present under {ORBIT_DIR} — skipping.")
        return True

    _step("Broker certificate + token")
    print(
        "`rose run --remote` needs a TLS certificate and a shared secret token so\n"
        "your workflow can talk to itself securely. This is a one-time, local-only\n"
        "step — nothing is generated or sent anywhere else."
    )

    if confirm("Generate these automatically now?", default=True):
        if _auto_generate_broker_credentials():
            print("Certificate and token created successfully.")
            return True
        print("\nAutomatic generation didn't work. Here's how to do it by hand:")
    else:
        print("\nOkay — here's exactly how to do it yourself:")

    _print_manual_tls_and_token_fallback()
    while True:
        if _have_broker_credentials():
            print("Found them — continuing.")
            return True
        if not confirm("Have you completed the steps above?", default=False):
            return False


def _auto_generate_broker_credentials() -> bool:
    try:
        ORBIT_DIR.mkdir(parents=True, exist_ok=True)

        if not (CERT_PATH.exists() and KEY_PATH.exists()):
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:4096",
                    "-nodes",
                    "-keyout",
                    str(KEY_PATH),
                    "-out",
                    str(CERT_PATH),
                    "-days",
                    "365",
                    "-subj",
                    f"/CN={socket.getfqdn()}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            KEY_PATH.chmod(0o600)

        if not TOKEN_PATH.exists():
            import secrets

            TOKEN_PATH.write_text(secrets.token_urlsafe(32) + "\n")
            TOKEN_PATH.chmod(0o600)

        return _have_broker_credentials()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"  automatic generation failed: {exc}")
        return False


def _print_manual_tls_and_token_fallback() -> None:
    from radical.orbit.utils import TLS_RECIPE, TOKEN_RECIPE

    print("\n[Local computer] Run these commands yourself:\n")
    print(TLS_RECIPE)
    print(TOKEN_RECIPE)
    print(
        "Expected result: three files under "
        f"{ORBIT_DIR} — broker_cert.pem, broker_key.pem, broker.token."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  IRI / SFAPI credential collection
# ─────────────────────────────────────────────────────────────────────────────


def _try_automatic_nersc_iri_token(path) -> bool:
    """NERSC-only: offer to fetch/refresh the IRI token via Globus login,
    embedding rose.globus_auth instead of a manual paste. Returns True if a
    valid token ended up written to *path*."""
    if not confirm("Fetch/refresh this automatically via Globus login?", default=True):
        return False

    try:
        import rose.globus_auth as globus_auth
    except ImportError:
        print('  the "globus-sdk" package is not installed — run: pip install "ROSE[globus]"')
        return False

    try:
        token = globus_auth.get_nersc_iri_token()
    except Exception as exc:
        print(f"  Globus login didn't work: {exc}")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    path.chmod(0o600)
    print(f"  saved the token to {path}")
    return True


def _ensure_iri_token(facility: str) -> bool:
    path = _remote.AMSC_DIR / f"token_{facility}"
    while True:
        if path.exists() and path.read_text().strip():
            print(f"  using the IRI token at {path}")
            return True

        _step(f"{_FACILITY_LABEL[facility]} IRI token")
        print(f"You need a bearer token from {_FACILITY_LABEL[facility]}'s IRI service.")

        if facility == "nersc" and _try_automatic_nersc_iri_token(path):
            continue

        print("This has to be obtained through their own sign-in — it can't be automated.")
        token = ask(f"Paste the token here, or press Enter if you already saved it to {path}")
        if token:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token + "\n")
            path.chmod(0o600)
            continue

        print(f"\n[Local computer] Save your token to: {path}")
        if not confirm("Try checking again?", default=True):
            return False


def _ensure_sfapi_credentials(facility: str) -> bool:
    key_path = _remote.AMSC_DIR / f"sfapi_key_{facility}.pem"
    id_path = _remote.AMSC_DIR / f"sfapi_client_id_{facility}"

    while True:
        import os as _os

        have_id = bool(_os.environ.get("SFAPI_CLIENT_ID", "").strip()) or (
            id_path.exists() and id_path.read_text().strip()
        )
        have_key = key_path.exists() and key_path.read_text().strip()
        if have_id and have_key:
            print(f"  using the SFAPI credentials at {id_path} / {key_path}")
            return True

        _step("NERSC Superfacility API (SFAPI) credentials")
        print(
            "1. In your browser, sign in at https://iris.nersc.gov\n"
            "2. Go to Profile -> Superfacility API Keys -> Create Key\n"
            "3. Download the private key file"
        )
        client_id = ask(
            "Paste your client ID here (or press Enter if $SFAPI_CLIENT_ID is already set)"
        )
        if client_id:
            id_path.parent.mkdir(parents=True, exist_ok=True)
            id_path.write_text(client_id + "\n")
            id_path.chmod(0o600)
            if have_key:
                continue  # re-check immediately — nothing else to wait on

        if not have_key:
            print(f"\n[Local computer] Save the downloaded private key to: {key_path}")

        if not confirm("Ready to check again?", default=True):
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  Remote environment check (IRI / SFAPI only — PsiJ runs where you already are)
# ─────────────────────────────────────────────────────────────────────────────


def _job_state(status: dict) -> str:
    if isinstance(status.get("state"), str):
        return status["state"].lower()
    return (status.get("status") or {}).get("state", "unknown").lower()


def _poll_job(
    instance, resource_id: str, job_id: str, timeout: float = 180.0, poll: float = 5.0
) -> str:
    terminal = {"completed", "done", "failed", "canceled", "cancelled"}
    start = time.time()
    while time.time() - start < timeout:
        state = _job_state(instance.get_job_status(resource_id, job_id))
        if state in terminal:
            return state
        time.sleep(poll)
    return "timeout"


def _print_manual_remote_env_fallback(facility: str) -> None:
    login_host = _FACILITY_LOGIN_HOST[facility]
    print("\n[HPC] If you have your own SSH access, you can do this by hand instead:\n")
    print(f"  ssh {login_host}")
    print("  python3 -m venv ~/.amsc/ve")
    print("  ~/.amsc/ve/bin/pip install radical.orbit")
    print(
        "\nExpected result: no errors, and running "
        "'~/.amsc/ve/bin/radical-orbit-endpoint-wrapper.sh --help' works."
    )


def _ensure_remote_environment(
    instance, resource_id: str, account: str, queue_name, facility: str, timeout: float = 300.0
) -> bool:
    _step("Checking whether ROSE/ORBIT is installed on the remote system")
    setup_cmd = (
        "test -x ~/.amsc/ve/bin/radical-orbit-endpoint-wrapper.sh || "
        "(python3 -m venv ~/.amsc/ve && "
        "~/.amsc/ve/bin/pip install --quiet radical.orbit)"
    )
    job_spec = {
        "executable": "/bin/bash",
        "arguments": ["-lc", setup_cmd],
        "name": "rose-setup-env-check",
        "resources": {"node_count": 1, "process_count": 1},
        "attributes": {"queue_name": queue_name, "duration": 300, "account": account},
    }
    try:
        job = instance.submit_job(resource_id, job_spec)
        job_id = job["job_id"]
    except Exception as exc:
        print(f"  could not submit the environment-check job: {exc}")
        _print_manual_remote_env_fallback(facility)
        return confirm("Have you completed the manual steps above?", default=False)

    print(f"  submitted (job {job_id}) — waiting for it to finish...")
    state = _poll_job(instance, resource_id, job_id, timeout=timeout)

    if state not in ("completed", "done"):
        print(f"  the environment-check job ended with state '{state}', not success.")
        _print_manual_remote_env_fallback(facility)
        return confirm("Have you completed the manual steps above?", default=False)

    print("  the remote environment is ready.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup(rt, eb) -> None:
    if rt is not None:
        try:
            rt.stop()
        except Exception:
            pass
    if eb is not None:
        try:
            eb.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  IRI / SFAPI branch
# ─────────────────────────────────────────────────────────────────────────────


def run_iri_sfapi_setup(wait_timeout: float = 300.0) -> bool:
    _banner("IRI / SFAPI setup — everything runs from this computer")

    facility = choose(
        "Which HPC facility do you want to use?",
        [
            (_FACILITY_LABEL["nersc"], "nersc"),
            (_FACILITY_LABEL["olcf"], "olcf"),
        ],
    )
    resource_id = _FACILITY_RESOURCE[facility]

    if facility == "nersc":
        kind = choose(
            "How do you want to authenticate with NERSC?",
            [
                (
                    "SFAPI (Recommended) — NERSC's classic IRI /compute/* API is "
                    "broken server-side; this is the well-validated path.",
                    "sfapi",
                ),
                ("Classic IRI via Globus login", "iri"),
            ],
        )
    else:
        kind = _FACILITY_KIND[facility]

    if not ensure_broker_credentials():
        print("\nSetup did NOT succeed — broker credentials are required before continuing.")
        return False

    _step("A few details about your account")
    account = ask("Account / project name")
    home_dir = ask(f"Your $HOME directory on {_FACILITY_LABEL[facility]}")
    queue_name = ask("Queue / partition (leave blank for the default)") or None
    walltime_min = ask_int("Walltime in minutes", 30)

    if kind == "sfapi":
        got_creds = _ensure_sfapi_credentials(facility)
    else:
        got_creds = _ensure_iri_token(facility)
    if not got_creds:
        print("\nSetup did NOT succeed — a credential is required before continuing.")
        return False

    target = TargetConfig(
        kind=kind,
        endpoint=facility,
        resource_id=resource_id,
        account=account,
        home_dir=home_dir,
        queue_name=queue_name,
        walltime_min=walltime_min,
    )

    _step("Connecting")
    rt = None
    eb = None
    created = None
    try:
        from radical.orbit import EndpointRuntime
        from radical.orbit.embedded import EmbeddedBroker

        plugin_name = _remote._PLUGIN_FOR_KIND[kind]
        eb = EmbeddedBroker(plugins=plugin_name)
        eb.start()
        rt = EndpointRuntime(broker_url=eb.url)
        rt.start(wait=True)
        print("  connected.")

        connect_client = rt.get_plugin(_remote._BROKER, plugin_name)
        if kind == "sfapi":
            client_id, private_key = _remote._read_sfapi_credentials(facility)
            instance = connect_client.connect(facility, client_id, private_key)
        else:
            token = _remote._read_iri_token(facility)
            instance = connect_client.connect(facility, token)
        print(f"  authenticated with {_FACILITY_LABEL[facility]}.")
    except Exception as exc:
        print(f"\nCould not connect: {exc}")
        _cleanup(rt, eb)
        print(
            "\nSetup did NOT succeed. Double-check the credential you entered and "
            "the account/queue details, then run `rose setup` again."
        )
        return False

    if not _ensure_remote_environment(
        instance, resource_id, account, queue_name, facility, timeout=wait_timeout
    ):
        _cleanup(rt, eb)
        print("\nSetup did NOT succeed — the remote environment isn't ready yet.")
        return False

    _step("Submitting a test job to confirm everything works end to end")
    try:
        created = _remote._launch_iri_or_sfapi(rt, target, rt.broker_url)
        _remote._wait_for_endpoint(
            rt, created["endpoint_name"], timeout=wait_timeout, poll=5.0, heartbeat=30.0
        )
        print(f"  endpoint {created['endpoint_name']!r} registered successfully.")
        verified = True
    except Exception as exc:
        print(f"\nThe test job didn't come up: {exc}")
        verified = False
    finally:
        _remote._teardown(created)
        _cleanup(rt, eb)

    if not verified:
        print("\nSetup did NOT succeed — see the message above for what to fix.")
        return False

    _print_iri_sfapi_success(
        kind, facility, resource_id, account, queue_name, walltime_min, home_dir
    )
    return True


def _print_iri_sfapi_success(
    kind, facility, resource_id, account, queue_name, walltime_min, home_dir
) -> None:
    _banner("Setup verified — everything works!")
    print("Add this to your workflow.yaml, under `remote:`:\n")
    print("remote:")
    print("  target:")
    print(f"    kind: {kind}")
    print(f"    endpoint: {facility}")
    print(f"    resource_id: {resource_id}")
    print(f"    account: {account}")
    if queue_name:
        print(f"    queue_name: {queue_name}")
    print(f"    walltime_min: {walltime_min}")
    print(f"    home_dir: {home_dir}")
    print("\nThen run:\n")
    print("  rose run your_workflow.yaml --remote\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PsiJ branch
# ─────────────────────────────────────────────────────────────────────────────


def _has_scheduler() -> bool:
    return bool(shutil.which("sbatch") or shutil.which("qsub"))


def _print_psij_no_scheduler_instructions() -> None:
    _step("This computer doesn't have SLURM or PBS")
    print("PsiJ needs to run directly on an HPC login node.")
    login_host = ask(
        "What is the hostname of your HPC login node? (e.g. perlmutter.nersc.gov) "
        "— leave blank to skip"
    )
    print("\nHere is exactly what to do next:\n")
    if login_host:
        print("1. [Local computer] Connect to the login node:")
        print(f"     ssh {login_host}")
    else:
        print("1. [Local computer] Connect to your HPC login node over SSH.")
    print("2. [HPC login node] Install ROSE there:")
    print("     pip install ROSE")
    print("3. [HPC login node] Run this wizard again:")
    print("     rose setup")
    print("\nIt will detect the scheduler automatically and continue from here.")


def run_psij_setup() -> bool:
    _banner("PsiJ setup — submit jobs directly from this login node")

    if not _has_scheduler():
        _print_psij_no_scheduler_instructions()
        return False

    if not ensure_broker_credentials():
        print("\nSetup did NOT succeed — broker credentials are required before continuing.")
        return False

    _step("A few details")
    account = ask("Account / project name")
    queue_name = ask("Queue / partition (leave blank for the default)") or None

    _step("Verifying PsiJ works on this login node")
    rt = None
    eb = None
    verified = False
    try:
        from radical.orbit import EndpointRuntime
        from radical.orbit.embedded import EmbeddedBroker

        eb = EmbeddedBroker(plugins="psij")
        eb.start()
        rt = EndpointRuntime(broker_url=eb.url)
        rt.start(wait=True)
        rt.get_plugin(_remote._BROKER, "psij")
        print("  PsiJ is reachable and ready.")
        verified = True
    except Exception as exc:
        print(f"\nCould not verify PsiJ: {exc}")
    finally:
        _cleanup(rt, eb)

    if not verified:
        print("\nSetup did NOT succeed — see the message above for what to fix.")
        return False

    _print_psij_success(account, queue_name)
    return True


def _print_psij_success(account, queue_name) -> None:
    _banner("Setup verified — everything works!")
    print("Add this to your workflow.yaml, under `remote:`:\n")
    print("remote:")
    print("  embedded: true")
    print("  target:")
    print("    kind: psij")
    print(f"    account: {account}")
    if queue_name:
        print(f"    queue_name: {queue_name}")
    print("\nThen, from this same login node, run:\n")
    print("  rose run your_workflow.yaml --remote\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_setup_wizard(wait_timeout: float = 300.0) -> bool:
    _banner("ROSE first-time setup")
    print(
        "This wizard gets `rose run --remote` working end to end.\n"
        "You'll answer a few simple questions; anything technical (certificates,\n"
        "tokens, environments) is handled for you or explained exactly."
    )

    mode = choose(
        "How do you want to run remote HPC jobs?",
        [
            (
                "IRI/SFAPI — everything runs from right here on this computer. "
                "No SSH into the HPC system needed. Works today for NERSC and OLCF.",
                "iri",
            ),
            (
                "PsiJ — jobs submit directly from an HPC login node you're already on. "
                "Works on any SLURM/PBS system, but this wizard needs to run there.",
                "psij",
            ),
        ],
    )

    if mode == "iri":
        return run_iri_sfapi_setup(wait_timeout=wait_timeout)
    return run_psij_setup()
