#!/usr/bin/env python
import os, socket
import torch.distributed as dist
print("before_init", socket.gethostname(), {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK")}, flush=True)
dist.init_process_group(os.environ.get("PCALM_BACKEND", "gloo"))
print("after_init", socket.gethostname(), dist.get_rank(), flush=True)
dist.barrier()
print("after_barrier", socket.gethostname(), dist.get_rank(), flush=True)
dist.destroy_process_group()
