If you have a 64-core machine with one or more GPUs, here is exactly what will happen with the current script:

### 1. The Ordering Phase (multi_opt)
*   **What you want:** `n_starts=10` using 10 CPU cores, leaving the GPU alone (since ordering is just quick rotamer checking).
*   **What happens:** By default, OpenMM is greedy. When each of those 10 parallel Python processes starts its mini-minimization, it will try to grab the GPU. If you have 1 GPU, all 10 processes will fight for it simultaneously. This causes massive PCIe lane traffic, context-switching overhead, and potentially out-of-memory (OOM) errors on the GPU. If you have 4 GPUs, they will fight over those 4. 

### 2. The Final Relaxation Phase
*   **What you want:** Send the 8 winners to the GPU(s) in parallel to finish quickly.
*   **What happens (as we discussed):** The python loop runs them serially. The script sends Winner #1 to the GPU. The GPU (which might have 10,000 CUDA cores) will easily crush a 5,000-atom polymer minimization in seconds, largely sitting idle because the system isn't big enough to saturate it. Then it waits for Python to do IO, then it sends Winner #2, etc. 

### How to solve this for your 64-core machine:

To properly utilize a machine like this, you need to manage the **Resource Visibility** using environment variables before you run the python command.

#### Scenario A: Pure CPU Brute Force (Recommended for this script)
If you have 64 cores, the GPU is actually overkill for an 8x200-iteration minimization. You can force the entire script (both ordering and relaxation) to stay on the CPU and split the cores.

```bash
# Hide the GPUs from OpenMM
export CUDA_VISIBLE_DEVICES=""
export OPENMM_DEFAULT_PLATFORM="CPU"

# Prevent OpenMM from grabbing all 64 cores for a single tiny minimization
export OPENMM_CPU_THREADS="4" 

# Now run the script
python -m poly_csp.pipelines.build_csp multi_opt.n_starts=10 ...
```
*   **Why this works:** During ordering, 10 Python processes will spawn. Each is restricted to 4 threads. That's 40 threads total—well within your 64-core limit. There is no GPU fighting. During relaxation, the 8 jobs run serially, each using 4 fast CPU cores. 

#### Scenario B: The HPC "Split" Workflow
If you want to use the GPUs for maximum speed on the final relaxation, you must decouple the pipeline as I mentioned earlier:

1.  **Step 1 (CPU):** Run the ordering to get 8 SDFs, hiding the GPU.
2.  **Step 2 (GPU):** Launch 4 separate python worker scripts (if you have 4 GPUs) or a job queue. For each worker, set `export CUDA_VISIBLE_DEVICES="0"` (or 1, 2, 3) so each worker has exclusive access to one GPU to run the relaxation.

---

On a shared workstation, it is critical to manually check and assign GPUs, because OpenMM (like most scientific software) will just blindly try to use GPU 0 unless you specifically block it. 

### Step 1: Check Current GPU Usage
You can always check the real-time status of your GPUs using the NVIDIA System Management Interface:

```bash
nvidia-smi
```

When you run this on a workstation, you'll see a table. Look at the **bottom section** ("Processes"). 
*   If a GPU (e.g., GPU 1) is listed with a high memory usage and a process name (like `python`), **someone else is using it.**
*   If a GPU (e.g., GPU 2) is missing from the processes list and shows `0MiB / VRAM` in the top section, **it is completely free.**

### Step 2: "Quarantine" the Free GPUs for Your Shell
Let's say you ran `nvidia-smi` and saw that GPUs 0 and 1 are busy, but **GPUs 2 and 3 are free**.

Before you run your python commands, you tell your current terminal session to "pretend" the busy GPUs don't exist by setting the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# This tells any software run in this terminal that ONLY GPUs 2 and 3 exist.
# As far as OpenMM knows, GPU 2 is now its "GPU 0", and GPU 3 is "GPU 1".
export CUDA_VISIBLE_DEVICES="2,3" 
```

### Step 3: Launching Your Independent Jobs
Now that your terminal is safely fenced off from your colleagues' jobs, you can launch your Split-Workflow workers.

Since you have 2 free GPUs (which are mapped as "0" and "1" in this environment), you can launch two parallel relaxation workers and assign them their own dedicated GPU:

```bash
# Terminal 1
export CUDA_VISIBLE_DEVICES="2,3" 

# Launch worker A on the first free GPU (physical GPU 2)
CUDA_VISIBLE_DEVICES="0" python script_to_relax_sdf.py  winner_1.sdf &

# Launch worker B on the second free GPU (physical GPU 3)
CUDA_VISIBLE_DEVICES="1" python script_to_relax_sdf.py  winner_2.sdf &

# Wait for both background processes to finish
wait 
```

*Note: The `CUDA_VISIBLE_DEVICES="X"` placed right before the `python` command temporarily overrides the environment for just that single line.*

### What if I want something automatic?
If you are doing this constantly, manually guarding environment variables gets tedious. You might want to consider installing a lightweight job scheduler on that workstation, like:
*   **SLURM:** (Heavy duty, standard for academic clusters)
*   **Task Spooler ([ts](cci:1://file:///home/lukas/work/projects/chiral_csp_poly/src/poly_csp/pipelines/build_csp.py:327:0-344:19)):** (Very lightweight, great for single workstations)
*   **Ray:** (A Python library that handles this exact problem by letting you decorate functions with `@ray.remote(num_gpus=1)`).