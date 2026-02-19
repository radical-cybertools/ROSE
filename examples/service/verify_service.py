import asyncio
import os
import time
import json
import shutil
from pathlib import Path
from rose.service.manager import ServiceManager
from rose.service.client import ServiceClient

# Mock Job ID
JOB_ID = "test_job_123"
SERVICE_ROOT = Path.home() / ".rose" / "services" / JOB_ID

def cleanup():
    if SERVICE_ROOT.exists():
        shutil.rmtree(SERVICE_ROOT)

async def run_verification():
    print(f"--- Starting Verification for Job {JOB_ID} ---")
    cleanup()
    
    # 1. Start Service in Background Task
    manager = ServiceManager(JOB_ID)
    service_task = asyncio.create_task(manager.run())
    
    # Allow service to init
    await asyncio.sleep(2)
    
    # 2. Initialize Client
    client = ServiceClient(JOB_ID)
    
    # 3. Submit Workflow
    wf_path = "service_real.yaml"
    print(f"Submitting {wf_path}...")
    req_id = client.submit_workflow(wf_path)
    print(f"Submitted. Request ID: {req_id}")
    
    # 4. Poll for Status until Running
    wf_id = None
    for _ in range(10):
        await asyncio.sleep(1)
        registry = client.list_workflows()
        if registry:
            wf_id = list(registry.keys())[0]
            state = registry[wf_id]['state']
            print(f"Workflow {wf_id} State: {state}")
            if state in ["RUNNING", "COMPLETED"]:
                break
    
    if not wf_id:
        print("Failed to get workflow ID")
        await manager.shutdown()
        return

    # 5. Cancel Workflow (if still running)
    print(f"Canceling {wf_id}...")
    client.cancel_workflow(wf_id)
    
    # 6. Check for Canceled State
    for _ in range(5):
        await asyncio.sleep(1)
        status = client.get_workflow_status(wf_id)
        print(f"Workflow {wf_id} State: {status['state']}")
        if status['state'] == "CANCELED":
            print("SUCCESS: Workflow Canceled")
            break
        if status['state'] == "COMPLETED":
            print("Workflow finished before cancel (acceptable for short test)")
            break
            
    # Shutdown
    await manager.shutdown()
    try:
        await service_task
    except asyncio.CancelledError:
        pass
    print("--- Verification Finished ---")

if __name__ == "__main__":
    asyncio.run(run_verification())
