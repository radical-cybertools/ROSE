"""
rose.remote — bootstrap and run a ROSE workflow on a remote ORBIT endpoint.

Three phases, driven by the ``remote.target`` block of the workflow spec
(see ``rose.spec.schema.TargetConfig``) instead of interactive prompts:

* **Bootstrap** — submit a job that starts a ``radical-orbit-endpoint``
  process on the configured target, via the broker's ``iri_connect`` /
  ``sfapi_connect`` plugin (NERSC should use ``sfapi`` — IRI's
  ``/compute/*`` routes are broken server-side there; OLCF stays on
  ``iri``) or via PsiJ on an already-connected login-node endpoint.
* **Execute** — wait for the endpoint to register, then run the workflow's
  task graph against it (``spec.workflow``, built in ``rose.spec``).
* **Teardown** — cancel the bootstrap job and disconnect.

Ported from ``amsc/use-cases/service_utils.py``'s interactive discover/
launch/teardown flow. That module's ``BridgeClient`` (``radical.edge.client``)
no longer exists — every participant, including this one, is now a
``radical.orbit.EndpointRuntime``.

``remote.embedded: true`` hosts the ORBIT broker in this process
(``radical.orbit.embedded.EmbeddedBroker``) instead of connecting to an
already-running one — no separate ``radical-orbit-broker.py`` deployment
needed. Combined with ``target.kind: psij``, PsiJ runs on the embedded
broker itself (no separate login-node endpoint either).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .spec import WorkflowSpec
    from .spec.schema import TargetConfig

_ENDPOINT_WRAPPER = 'radical-orbit-endpoint-wrapper.sh'
_BROKER = 'broker'  # the broker's own participant name (src='broker'), embedded or not

_PLUGIN_FOR_KIND = {'iri': 'iri_connect', 'sfapi': 'sfapi_connect', 'psij': 'psij'}

AMSC_DIR = Path(os.environ.get('AMSC_DIR') or Path.home() / '.amsc').expanduser()


def _wrapper_path(home_dir: str) -> str:
    return f"{home_dir.rstrip('/')}/.amsc/ve/bin/{_ENDPOINT_WRAPPER}"


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap: iri / sfapi
# ─────────────────────────────────────────────────────────────────────────────

def _read_iri_token(endpoint: str) -> str:
    path = AMSC_DIR / f'token_{endpoint}'
    if not path.exists():
        raise RuntimeError(
            f'IRI token file missing: {path}  (put your IRI bearer token there)')
    token = path.read_text().strip()
    if not token:
        raise RuntimeError(f'IRI token file is empty: {path}')
    return token


def _read_sfapi_credentials(endpoint: str) -> tuple:
    client_id = os.environ.get('SFAPI_CLIENT_ID', '').strip()
    if not client_id:
        raise RuntimeError(
            "SFAPI_CLIENT_ID env var is required for remote.target.kind='sfapi'")
    key_path = AMSC_DIR / f'sfapi_key_{endpoint}.pem'
    if not key_path.exists():
        raise RuntimeError(f'SFAPI private key not found: {key_path}')
    private_key = key_path.read_text().strip()
    if not private_key:
        raise RuntimeError(f'SFAPI private key file is empty: {key_path}')
    return client_id, private_key


def _launch_iri_or_sfapi(rt, target: 'TargetConfig', broker_url: str) -> Dict[str, Any]:
    plugin_name    = 'sfapi_connect' if target.kind == 'sfapi' else 'iri_connect'
    connect_client = rt.get_plugin(_BROKER, plugin_name)

    if target.kind == 'sfapi':
        client_id, private_key = _read_sfapi_credentials(target.endpoint)
        instance = connect_client.connect(target.endpoint, client_id, private_key)
    else:
        token = _read_iri_token(target.endpoint)
        instance = connect_client.connect(target.endpoint, token)

    endpoint_name = f'rose-{target.endpoint}-{uuid.uuid4().hex[:6]}'
    args = ['--name', endpoint_name, '--url', broker_url]
    if target.tunnel == 'forward':
        if not target.login_host:
            raise RuntimeError(
                "remote.target.login_host is required when tunnel='forward'")
        args += ['--tunnel', 'forward', '--tunnel-via', target.login_host]
    elif target.tunnel == 'reverse':
        args += ['--tunnel', 'reverse']

    attrs: Dict[str, Any] = {
        'queue_name': target.queue_name,
        'duration'  : target.walltime_min * 60,
        'account'   : target.account,
    }
    if target.constraint:  attrs['constraint']  = target.constraint
    if target.reservation: attrs['reservation'] = target.reservation

    env = {'RADICAL_ORBIT_BROKER_URL': broker_url}
    env.update(target.environment)
    if target.setup:
        env['RADICAL_ORBIT_ENDPOINT_SETUP'] = '; '.join(target.setup)

    job_spec: Dict[str, Any] = {
        'executable' : _wrapper_path(target.home_dir),
        'arguments'  : args,
        'name'       : endpoint_name,
        'resources'  : {'node_count': target.n_nodes, 'process_count': 1},
        'attributes' : attrs,
        'environment': env,
    }
    if target.workdir:
        job_spec['directory'] = target.workdir

    print(f'[rose --remote] submitting {target.kind} job '
          f'({target.endpoint} -> {target.resource_id}, '
          f'endpoint: {endpoint_name})...')
    job = instance.submit_job(target.resource_id, job_spec)
    print(f"[rose --remote] {target.kind} job_id: {job['job_id']}")

    return {
        'kind'         : target.kind,
        'client'       : instance,
        'connect_client': connect_client,
        'endpoint_key' : target.endpoint,
        'endpoint_name': endpoint_name,
        'resource_id'  : target.resource_id,
        'job_id'       : job['job_id'],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap: psij (login-node submission)
# ─────────────────────────────────────────────────────────────────────────────

def _launch_psij(rt, target: 'TargetConfig', broker_url: str,
                 embedded: bool = False) -> Dict[str, Any]:
    if embedded:
        # PsiJ runs on the embedded broker itself (this process) — no
        # separate login-node endpoint to connect to, and no remote
        # sysinfo.homedir() to call: we ARE the host, so use the local home.
        edge_name = _BROKER
        home      = target.home_dir or str(Path.home())
    else:
        edge_name = target.edge_name
        home      = rt.get_plugin(edge_name, 'sysinfo').homedir()
    psij     = rt.get_plugin(edge_name, 'psij')
    executor = target.executor or 'local'

    attrs: Dict[str, Any] = {'duration': target.walltime_min * 60,
                             'account' : target.account}
    if target.queue_name:
        attrs['queue_name'] = target.queue_name
    custom_attrs: Dict[str, Any] = {}
    if target.constraint:
        custom_attrs[f'{executor}.constraint'] = target.constraint

    env = {'RADICAL_ORBIT_BROKER_URL': broker_url}
    if target.setup:
        env['RADICAL_ORBIT_ENDPOINT_SETUP'] = '; '.join(target.setup)

    child_name = f'rose-{edge_name}-{uuid.uuid4().hex[:6]}'
    job_spec = {
        'executable'        : _wrapper_path(home),
        'arguments'         : ['--name', child_name, '--url', broker_url],
        'attributes'        : attrs,
        'custom_attributes' : custom_attrs,
        'resources'         : {'node_count': target.n_nodes, 'process_count': 1},
        'environment'       : env,
    }
    print(f'[rose --remote] submitting psij job via {edge_name} '
          f'(executor: {executor}, endpoint: {child_name})...')
    res = psij.submit_tunneled(job_spec, executor=executor, tunnel=target.tunnel)
    print(f"[rose --remote] psij job_id: {res['job_id']}")

    return {
        'kind'         : 'psij',
        'client'       : psij,
        'endpoint_name': res.get('endpoint_name', child_name),
        'job_id'       : res['job_id'],
        'parent_edge'  : edge_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Wait for the bootstrapped endpoint to register
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_endpoint(rt, name: str, timeout: float = 30 * 60,
                       poll: float = 3.0, heartbeat: float = 30.0) -> None:
    print(f'[rose --remote] waiting for endpoint {name!r} to register...')
    start     = time.time()
    last_beat = start
    while time.time() - start < timeout:
        if name in rt.topology():
            return
        time.sleep(poll)
        if time.time() - last_beat >= heartbeat:
            elapsed = int(time.time() - start)
            print(f'[rose --remote] ...{elapsed}s elapsed, '
                  f'{int(timeout - elapsed)}s left')
            last_beat = time.time()
    raise TimeoutError(f'endpoint {name!r} did not register within {timeout}s')


# ─────────────────────────────────────────────────────────────────────────────
#  Teardown
# ─────────────────────────────────────────────────────────────────────────────

def _teardown(created: Optional[Dict[str, Any]]) -> None:
    if not created:
        return
    print('[rose --remote] tearing down bootstrap job...')
    try:
        if created['kind'] == 'psij':
            created['client'].cancel_job(created['job_id'])
        else:
            created['client'].cancel_job(created['resource_id'], created['job_id'])
        print(f"[rose --remote] cancelled {created['kind']} job {created['job_id']}")
    except Exception as exc:
        print(f"[rose --remote] could not cancel {created['kind']} "
              f"job {created['job_id']}: {exc}")

    if created['kind'] in ('iri', 'sfapi'):
        try:
            created['connect_client'].disconnect(created['endpoint_key'])
            print(f"[rose --remote] disconnected {created['kind']} "
                  f"endpoint {created['endpoint_key']}")
        except Exception as exc:
            print(f"[rose --remote] could not disconnect "
                  f"{created['endpoint_key']}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_remote(spec: 'WorkflowSpec') -> None:
    """Bootstrap the ``remote.target`` from *spec*, run the workflow against
    it, then tear the bootstrap job down.

    Raises ``RuntimeError`` if ``spec.config.remote.target`` is unset — the
    CLI is expected to check this and produce a clearer error before calling
    in, but this guard keeps the function safe to call directly (e.g. from
    tests) too.

    ``spec.config.remote.embedded`` swaps connecting to an already-running
    broker for hosting one in this process (``EmbeddedBroker``, loaded with
    only the plugin the target needs) — no separate
    ``radical-orbit-broker.py`` deployment required.
    """
    from radical.orbit import EndpointRuntime

    remote_cfg = spec.config.remote
    target     = remote_cfg.target
    if target is None:
        raise RuntimeError(
            "run_remote() requires 'remote.target' to be set in the spec")

    eb = None
    if remote_cfg.embedded:
        from radical.orbit.embedded import EmbeddedBroker

        eb = EmbeddedBroker(plugins=_PLUGIN_FOR_KIND[target.kind])
        eb.start()
        print(f'[rose --remote] embedded broker: {eb.url}')
        try:
            rt = EndpointRuntime(broker_url=eb.url)
            rt.start(wait=True)
        except Exception:
            eb.stop()
            raise
    else:
        rt = EndpointRuntime(broker_url=remote_cfg.broker_url)
        rt.start(wait=True)

    broker_url = rt.broker_url
    print(f'[rose --remote] broker: {broker_url}')

    created: Optional[Dict[str, Any]] = None
    try:
        if target.kind == 'psij':
            if not remote_cfg.embedded and target.edge_name not in rt.topology():
                raise RuntimeError(
                    f'remote.target.edge_name {target.edge_name!r} is not '
                    f'currently connected to the broker')
            created = _launch_psij(rt, target, broker_url,
                                   embedded=remote_cfg.embedded)
        else:
            created = _launch_iri_or_sfapi(rt, target, broker_url)

        _wait_for_endpoint(rt, created['endpoint_name'])
        print(f"[rose --remote] endpoint {created['endpoint_name']!r} is up")

        await spec.workflow(broker_url, created['endpoint_name'])

    finally:
        _teardown(created)
        rt.stop()
        if eb is not None:
            eb.stop()
        print('[rose --remote] done.')
