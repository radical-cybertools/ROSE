#!/usr/bin/env python3
"""
Example: Test ROSE Edge Plugin

Connects to a running RADICAL-Edge bridge, submits a workflow via the
ROSE plugin, and monitors its status.

Prerequisites:
  1. Bridge running:        radical-edge-bridge
  2. Edge with ROSE plugin: radical-edge-service (with ROSE plugin loaded)
  3. ROSE ServiceManager:   rose launch --job-id local_job_0

Usage:
    export RADICAL_BRIDGE_URL=https://localhost:8443
    python example_rose_plugin.py [--workflow FILE] [--job-id ID]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
)
for name in ["httpx", "httpcore", "urllib3"]:
    logging.getLogger(name).setLevel(logging.DEBUG)

log = logging.getLogger("rose.example")


def notification_cb(topic: str, data: dict):
    """Handle workflow notifications."""
    log.info(f"[NOTIFY] {topic}: {data}")


def main():
    parser = argparse.ArgumentParser(description="Test ROSE Edge Plugin")
    parser.add_argument("--workflow", "-w", default="debug_workflow.yaml")
    parser.add_argument("--job-id", "-j", default="local_job_0")
    parser.add_argument(
        "--bridge-url", "-b", default=os.environ.get("RADICAL_BRIDGE_URL", "https://localhost:8443")
    )
    args = parser.parse_args()

    # Resolve workflow path
    workflow = Path(args.workflow)
    if not workflow.exists():
        workflow = Path(__file__).parent / args.workflow
    if not workflow.exists():
        log.error(f"Workflow not found: {args.workflow}")
        sys.exit(1)

    log.info(f"Bridge:   {args.bridge_url}")
    log.info(f"Workflow: {workflow}")
    log.info(f"Job ID:   {args.job_id}")
    log.info("-" * 60)

    from radical.edge import BridgeClient

    # Import ROSE plugin to register client class locally
    import rose.service.api.rest  # noqa: F401

    try:
        bc = BridgeClient(url=args.bridge_url)
        edges = bc.list_edges()
    except Exception as e:
        log.error(f"Cannot connect to bridge: {e}")
        log.error("Make sure bridge and edge service are running.")
        sys.exit(1)

    if not edges:
        log.error("No edges connected to bridge")
        sys.exit(1)

    edge_id = edges[0]
    log.info(f"Using edge: {edge_id}")

    try:
        ec = bc.get_edge_client(edge_id)
        rose = ec.get_plugin("rose", job_id=args.job_id)
    except Exception as e:
        log.error(f"Cannot get ROSE plugin: {e}")
        log.error("Make sure ROSE plugin is loaded on edge service.")
        sys.exit(1)

    log.info(f"Session: {rose.sid}")
    rose.on_workflow_state(notification_cb)

    try:
        # Submit workflow
        log.info(f"Submitting {workflow}...")
        result = rose.submit_workflow(str(workflow.absolute()))
        wf_id = result["wf_id"]
        log.info(f"Submitted: {wf_id}")

        # Monitor status
        log.info("Monitoring (Ctrl+C to cancel)...")
        terminal = {"COMPLETED", "FAILED", "CANCELED"}
        last_state = None

        for _i in range(120):
            time.sleep(2)
            try:
                status = rose.get_workflow_status(wf_id)
                state = status.get("state", "UNKNOWN")
                if state != last_state:
                    log.info(f"State: {state}")
                    last_state = state
                if state in terminal:
                    if state == "FAILED":
                        log.error(f"Error: {status.get('error')}")
                    break
            except Exception as e:
                log.warning(f"Status error: {e}")

        # Final state
        log.info("-" * 60)
        for wid, info in rose.list_workflows().items():
            log.info(f"{wid}: {info.get('state')}")

    except KeyboardInterrupt:
        log.warning("Interrupted")

    finally:
        rose.off_workflow_state(notification_cb)
        rose.close()
        bc.close()
        log.info("Done.")


if __name__ == "__main__":
    main()
