# Command-Line Interface

Installing ROSE (`pip install .`) installs the `rose` command. It has two
subcommands: `run`, for executing a workflow YAML, and `setup`, an
interactive wizard for getting remote HPC execution working for the first
time.

```bash
rose --help
```

## `rose run`

```bash
rose run <yaml> --local [--backend NAME]
rose run <yaml> --remote
```

`<yaml>` is the path to a ROSE workflow spec. Exactly one of `--local` or
`--remote` is required — passing both, or neither, is an error.

- **`--local`** runs the workflow on this machine via a
  [rhapsody](https://github.com/radical-cybertools/rhapsody) backend.
  `--backend NAME` selects which one (default: `dragon`).
- **`--remote`** runs the workflow on a remote HPC target through the ORBIT
  broker, spec-driven — no separate bootstrap step. It requires a
  `remote.target` block in the YAML; see [ROSE as a Service](../user-guide/rose-aas.md)
  for the concept and
  [`remote.target`](../user-guide/spec-api.md#remotetarget-bootstrapping-the-endpoint-for-rose-run-remote)
  for the field reference. If `remote.target` is missing, `rose run
  --remote` exits immediately with a message telling you where to add it.

## `rose setup`

```bash
rose setup [--wait-timeout SECONDS]
```

An interactive, first-time wizard that gets `rose run --remote` working
end to end. It asks only simple questions — facility, account, names — and
handles everything technical itself: broker TLS cert/token setup, IRI/SFAPI
credentials, and the remote Python environment.

You choose one of two modes:

- **IRI / SFAPI** — everything runs from this computer; no SSH to the HPC
  system needed. Works today for NERSC and OLCF.
- **PsiJ** — jobs submit directly from an HPC login node you're already
  on; works on any SLURM/PBS system, but the wizard needs to run there.

The wizard verifies the setup by actually bringing up a real ORBIT
endpoint on the target resource and waiting for it to register — the same
code path a real `rose run --remote` takes, not an approximation. On
success it prints a ready-to-paste `remote:` YAML block and the exact
follow-up command, `rose run your_workflow.yaml --remote`.

`--wait-timeout SECONDS` controls how long the wizard waits for HPC jobs
(the environment check and the endpoint-verification job) to come up.
Default is 300 seconds; if you omit the flag, the wizard prints a note
saying it's using the default and how to change it. On a busy queue, raise
it: `rose setup --wait-timeout 900`.

See [ROSE as a Service](../user-guide/rose-aas.md) and
[`remote.embedded`](../user-guide/spec-api.md#remoteembedded-run-without-a-standalone-broker)
for what the resulting `remote:` block means.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success. |
| `1`  | General failure — `rose setup` didn't complete, or `rose run --remote` is missing `remote.target`. |
| `2`  | Argument error — `--local`/`--remote` both given or neither given. |
