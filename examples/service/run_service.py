import asyncio
import os
import time
import json
import shutil
import logging
from pathlib import Path
from rose.service.manager import ServiceManager
from rose.service.client import ServiceClient

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_service")

# Job ID for this service instance
JOB_ID = "rose_service_run"
SERVICE_ROOT = Path.home() / ".rose" / "services" / JOB_ID

def cleanup():
    if SERVICE_ROOT.exists():
        logger.info(f"Cleaning up previous service root at {SERVICE_ROOT}")
        shutil.rmtree(SERVICE_ROOT)

async def run_workflow():
    print(f"--- Starting ROSE Service for Job {JOB_ID} ---")
    cleanup()
    
    # 1. Start Service Manager in a background task
    manager = ServiceManager(JOB_ID)
    service_task = asyncio.create_task(manager.run())
    
    # Wait for service initialization
    await asyncio.sleep(1)

    # 2. Initialize Client
    client = ServiceClient(JOB_ID)

    # 3. Submit the realistic workflow
    wf_path = "service_real.yaml"
    if not os.path.exists(wf_path):
        print(f"Error: Workflow file not found at {wf_path}")
        await manager.shutdown()
        return

    print(f"Submitting workflow: {wf_path}")
    req_id = client.submit_workflow(wf_path)
    print(f"Submitted. Request ID: {req_id}")
    
    # 4. Wait for workflow to be picked up and assigned a wf_id
    wf_id = None
    print("Waiting for service to assign Workflow ID...")
    for _ in range(20):
        await asyncio.sleep(1)
        registry = client.list_workflows()
        if registry:
            wf_id = list(registry.keys())[0]
            print(f"Assigned Workflow ID: {wf_id}")
            break
            
    if not wf_id:
        print("Error: Workflow was not picked up by the service.")
        await manager.shutdown()
        return

    # 5. Monitor progress until completion
    print(f"Monitoring workflow {wf_id}...")
    last_state = None
    while True:
        status = client.get_workflow_status(wf_id)
        if not status:
            print("Error: Workflow status lost.")
            break
            
        current_state = status.get('state')
        if current_state != last_state:
            print(f"Status Change: {current_state}")
            last_state = current_state
            
        if current_state in ["COMPLETED", "FAILED", "CANCELED"]:
            print(f"Workflow reached terminal state: {current_state}")
            if current_state == "FAILED":
                print(f"Error Details: {status.get('error')}")
            break

        # Optional: Print iteration progress if available in stats
        stats = status.get('stats', {})
        if 'iteration' in stats:
            print(f"  Iteration: {stats['iteration']} | Metric: {stats.get('metric_value', 'N/A')}", end='\r')
            
        await asyncio.sleep(2)
    
    # 6. Final Report
    status = client.get_workflow_status(wf_id)
    print("\n--- Final Workflow Status ---")
    print(json.dumps(status, indent=2))
    
    # 7. Graceful Shutdown
    print("Shutting down service...")
    await manager.shutdown()
    try:
        # Give some time for internal tasks to finish before canceling the service loop
        await asyncio.wait_for(service_task, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
        
    print("--- Service Run Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_workflow())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
