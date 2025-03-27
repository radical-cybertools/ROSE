import os
import argparse
import radical.utils as ru
import radical.pilot as rp
import radical.analytics as ra


def process_session(sid):
    session = ra.Session(sid, 'radical.pilot')
    data = {'session': session,
            'pilot'  : session.filter(etype='pilot', inplace=False).get()[0],
            'tasks'  : session.filter(etype='task',  inplace=False)}
    data['sid'] = sid
    data['pid'] = data['pilot'].uid
    data['smt'] = os.environ.get('RADICAL_SMT') or \
                data['pilot'].cfg['resource_details']['rm_info']['threads_per_core']
    return data


def get_runtime(data, ntasks=None):
    ttx = data['tasks'].duration(event=[{ru.EVENT: 'task_run_start'},
                                        {ru.EVENT: 'task_run_stop'}])


    runtime = data['pilot'].duration(event=[{ru.EVENT: 'bootstrap_0_start'},
                                            {ru.EVENT: 'bootstrap_0_stop'}])
    
    
    rose_ovh1 = data['tasks'].duration(event=[{ru.EVENT: 'detect_deps_start'},
                                              {ru.EVENT: 'detect_deps_stop'}])



    rose_ovh2 = data['tasks'].duration(event=[{ru.EVENT: 'resolve_al_task_start'},
                                              {ru.EVENT: 'resolve_al_task_stop'}])


    print(f'ROSE OVH: {rose_ovh1 + rose_ovh2}s')
    print(f'TOTAL RUNTIME: {round(runtime)}s | RCT OVH: {round(runtime - ttx)}s')
    if ntasks and ntasks > 1:
        task_idx = ntasks - 1
        print(f'# Following rates are based on the first {ntasks} tasks')
    else:
        task_idx = -1
        ntasks = None
        print(f'# Following rates are based on all tasks')
    
    # calculate scheduling throughput (for the first N tasks or for all tasks)
    ts_schedule_ok = sorted(data['session'].timestamps(event={ru.STATE: 'AGENT_SCHEDULING'}))
    total_tasks = ntasks or len(ts_schedule_ok)
    print('scheduling rate: ', total_tasks / (ts_schedule_ok[task_idx] - ts_schedule_ok[0]))
    
    # calculate launching rate (for the first N tasks or for all tasks)
    ts_agent_executing = sorted(data['session'].timestamps(event=[{ru.EVENT: 'launch_submit'}]))
    total_tasks = ntasks or len(ts_agent_executing)
    print('launching rate: ', total_tasks / (ts_agent_executing[task_idx] - ts_agent_executing[0]))

    
    
def main():

    parser = argparse.ArgumentParser(description="Example Python program using argparse.")

    # Add arguments
    parser.add_argument('session_ids', type=str,
                        nargs='+', help='list of sessions separated by spaces')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print(f"Sessions: {args.session_ids}")
    
    sessions = [item for sublist in args.session_ids for item in sublist.split(',')]
    
    for sid in sessions:
        print('================================Now Extracting Session Data====================================\n')
        print(sid)
        sid = os.path.join(os.getcwd(), sid)
        data = process_session(sid)
        get_runtime(data)

main()
