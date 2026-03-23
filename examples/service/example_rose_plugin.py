#!/usr/bin/env python3
"""
Example: ROSE Workflow Execution via RADICAL-Edge Plugin

This example demonstrates how to submit and monitor Active Learning
workflows using the ROSE plugin for RADICAL-Edge.

Architecture:
    Client (this script)
        | HTTP/REST
        v
    RADICAL-Edge Bridge
        | WebSocket
        v
    Edge Service (ROSE plugin with embedded workflow execution)

Prerequisites:
    1. RADICAL-Edge bridge running
    2. RADICAL-Edge service running with ROSE plugin loaded

Usage:
    export RADICAL_BRIDGE_URL=https://localhost:8443
    python example_rose_plugin.py --workflow debug_workflow.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
)

# Reduce noise from HTTP libraries
for name in ['httpx', 'httpcore', 'urllib3', 'hpack']:
    logging.getLogger(name).setLevel(logging.WARNING)

log = logging.getLogger("rose.example")


def on_state_change(topic: str, data: dict):
    """Callback for workflow state change notifications."""
    wf_id = data.get('wf_id', '?')
    state = data.get('state', '?')
    stats = data.get('stats', {})

    if stats:
        iteration = stats.get('iteration', '-')
        metric = stats.get('metric_value', '-')
        log.info(f'[{wf_id}] {state} (iteration={iteration}, metric={metric})')
    else:
        log.info(f'[{wf_id}] {state}')

def notification_cb(topic: str, data: dict):
    """Handle workflow notifications."""
    log.info(f"[NOTIFY] {topic}: {data}")


def main():
    parser = argparse.ArgumentParser(
        description='Submit ROSE workflow via RADICAL-Edge plugin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--workflow', '-w',
        default='debug_workflow.yaml',
        help='Path to workflow YAML file (default: debug_workflow.yaml)'
    )
    parser.add_argument(
        '--bridge-url', '-b',
        default=os.environ.get('RADICAL_BRIDGE_URL', 'https://localhost:8443'),
        help='Bridge URL (default: $RADICAL_BRIDGE_URL or https://localhost:8443)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    args = parser.parse_args()

    # Resolve workflow path
    workflow = Path(args.workflow)
    if not workflow.exists():
        workflow = Path(__file__).parent / args.workflow
    if not workflow.exists():
        log.error(f"Workflow not found: {args.workflow}")
        sys.exit(1)

    workflow = workflow.resolve()

    log.info(f"Bridge:   {args.bridge_url}")
    log.info(f"Workflow: {workflow}")
    log.info(f"Job ID:   {args.job_id}")
    log.info("-" * 60)

    # Import dependencies
    from radical.edge import BridgeClient
    import rose.service.api.rest  # Register ROSE plugin client class

    # Connect to bridge
    try:
        bc = BridgeClient(url=args.bridge_url)
        edges = bc.list_edges()
    except Exception as e:
        log.error(f"Cannot connect to bridge: {e}")
        sys.exit(1)

    if not edges:
        log.error("No edges connected to bridge")
        sys.exit(1)

    edge_id = edges[0]
    log.info(f"Using edge: {edge_id}")

    # Get ROSE plugin client
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
        start_time = time.time()
        terminal = {"COMPLETED", "FAILED", "CANCELED"}
        last_state = None

        while time.time() - start_time < args.timeout:

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

            state = status.get('state', 'UNKNOWN')

            if state in terminal:
                log.info('-' * 50)
                if state == 'COMPLETED':
                    log.info(f'Workflow {wf_id} completed successfully')
                elif state == 'FAILED':
                    log.error(f'Workflow {wf_id} failed: {status.get("error")}')
                else:
                    log.warning(f'Workflow {wf_id} was canceled')
                break
        else:
            log.warning(f'Timeout after {args.timeout}s')

        # Show final workflow list
        log.info("-" * 60)
        for wid, info in rose.list_workflows().items():
            log.info(f"  {wid}: {info.get('state')}")

    except KeyboardInterrupt:
        log.warning('Interrupted by user')

    finally:
        rose.off_workflow_state(on_state_change)
        rose.close()
        bc.close()
        log.info("Done")


if __name__ == "__main__":
    main()
