import argparse
from pathlib import Path

from distutils.util import strtobool
import os, time, subprocess
from concurrent.futures import ThreadPoolExecutor

def run_sbatch_script(script_path):
    print(f"Running {script_path}")
    subprocess.run(["sbatch", "-W", f"{script_path}"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-nd", "--num_days", type=int, default=1, help="number of days")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-no", "--node", type=str, default="", help="node to run on")
    parser.add_argument("-ram", "--ram_mem", type=int, default=40, help="ram in gigs")
    parser.add_argument("-gpu", "--gpus", type=int, default=1, help="num of gpus to use")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-nj", "--num_jobs", type=int, default=1, help="number of jobs to run at a time")
    args = parser.parse_args()
    scripts = [] 
    project_root = Path(__file__).resolve().parent.parent
    cur_dir = Path(__file__).resolve().parent
    sh_dir = os.path.join(cur_dir, 'sh')
    if os.path.exists(sh_dir) == False:
        os.makedirs(sh_dir)

    py_path = os.path.join(project_root, 'extractor.py')


    script_idx = 0
    start_time = str(int(time.time() * 1000))

    job_str = f'extr'
    slurm_strarr1 = ["#!/bin/bash"]
    slurm_strarr2 = [f"#SBATCH -p {args.partition}"]
    if args.partition != 'preempt':
        if args.partition != 'soundbendor':
            slurm_strarr2 = ['#SBATCH -A eecs', f"#SBATCH -p {args.partition}"]
        else:
            slurm_strarr2 = ['#SBATCH -A soundbendor', f"#SBATCH -p {args.partition}"]
    if len(args.node) > 0:
        slurm_strarr2.append(f"#SBATCH -w {args.node}")
    slurm_strarr3 = [f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", f"#SBATCH -t {args.num_days}-00:00:00", f"#SBATCH --job-name={job_str}", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/out_symr1ex/{job_str}-%j.out", ""]
    slurm_strarr = slurm_strarr1 + slurm_strarr2 + slurm_strarr3
    p_str = f"python {py_path}" 
    slurm_strarr.append(p_str)
    script_fname = f"{start_time}_{job_str}.sh"
    script_idx += 1
    script_path = os.path.join(sh_dir, script_fname)
    script_str = "\n".join(slurm_strarr)
    print(f"Creating {script_fname}")
    with open(script_path, 'w') as f:
        f.write(script_str)
    subprocess.run(["chmod", "u+x", f"{script_path}"])
    scripts.append(script_path)

    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        executor.map(run_sbatch_script, scripts)





