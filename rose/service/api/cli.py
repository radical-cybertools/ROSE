import argparse
import asyncio
import os
import sys
import json
from pathlib import Path

from rose.service.manager import ServiceManager
from rose.service.client import ServiceClient

def get_job_id():
    """Get SLURM_JOB_ID from env or argument."""
    # Priority: env var -> but for client, user might validly pass it as arg.
    # For 'launch', we usually rely on env var if running inside job.
    return os.environ.get("SLURM_JOB_ID", "local_job_0")

def cmd_launch(args):
    """Start the Service Manager."""
    job_id = args.job_id or get_job_id()
    print(f"Launching ROSE Service for Job ID: {job_id}")
    
    manager = ServiceManager(job_id)
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        print("Service stopping...")
        # graceful shutdown could be added here

def cmd_submit(args):
    """Submit a workflow."""
    job_id = args.job_id or get_job_id()
    client = ServiceClient(job_id)
    
    try:
        req_id = client.submit_workflow(args.workflow_file)
        print(f"Submitted workflow request. Request ID: {req_id}")
        print(f"Note: This is the request ID. The Workflow ID (wf_id) will be assigned by the service.")
    except Exception as e:
        print(f"Error submitting workflow: {e}")
        sys.exit(1)

def cmd_cancel(args):
    """Cancel a workflow."""
    job_id = args.job_id or get_job_id()
    client = ServiceClient(job_id)
    
    try:
        req_id = client.cancel_workflow(args.wf_id)
        print(f"Sent cancellation request for {args.wf_id}. Request ID: {req_id}")
    except Exception as e:
        print(f"Error cancelling workflow: {e}")
        sys.exit(1)

def cmd_status(args):
    """Get status."""
    job_id = args.job_id or get_job_id()
    client = ServiceClient(job_id)
    
    try:
        if args.wf_id:
            status = client.get_workflow_status(args.wf_id)
            if status:
                print(json.dumps(status, indent=2))
            else:
                print(f"Workflow {args.wf_id} not found.")
        else:
            # List all
            registry = client.list_workflows()
            print(f"Workflows in Job {job_id}:")
            for wfid, wfdata in registry.items():
                print(f" - {wfid}: {wfdata.get('state')}")
    except Exception as e:
        print(f"Error getting status: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ROSE Service CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arg for job id
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--job-id", help="SLURM Job ID (default: $SLURM_JOB_ID)")

    # Launch
    p_launch = subparsers.add_parser("launch", parents=[parent_parser], help="Start the Service Manager daemon")
    p_launch.set_defaults(func=cmd_launch)

    # Submit
    p_submit = subparsers.add_parser("submit", parents=[parent_parser], help="Submit a workflow")
    p_submit.add_argument("workflow_file", help="Path to workflow YAML file")
    p_submit.set_defaults(func=cmd_submit)

    # Cancel
    p_cancel = subparsers.add_parser("cancel", parents=[parent_parser], help="Cancel a workflow")
    p_cancel.add_argument("wf_id", help="Workflow ID to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    # Status
    p_status = subparsers.add_parser("status", parents=[parent_parser], help="Get workflow status")
    p_status.add_argument("wf_id", nargs="?", help="Optional Workflow ID")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
