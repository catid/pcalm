# Cross-node NCCL attempt (Phase 0)

On 2026-10-29 UTC, a one-GPU-per-node `torchrun --nnodes=2` all-reduce was attempted between foureyes and ripper with `MASTER_ADDR=10.10.0.216`, `MASTER_PORT=29518`, `NCCL_IB_DISABLE=1`, and `NCCL_SOCKET_IFNAME=enp211s0f0np0`. TCP rendezvous had already passed. Foureyes used `torch 2.12.0+cu130`; ripper first used its unpinned source build `2.10.0a0+git066997f5` with CUDA 13.0. Both processes remained in initialization for over 75 seconds without producing a result and were terminated. This is a failed environment-parity experiment, not an all-reduce performance result.

After copying the exact 5.8 GiB `/home/catid/venvs/dnn` environment to ripper (both then reported `torch 2.12.0+cu130`, CUDA 13.0, NCCL 2.29.7) and copying the source tree to `/var/tmp/pcalm`, the `torchrun` wrapper still remained silent. A direct explicit-rank launch then initialized NCCL successfully and produced a result, so the earlier symptom is a launcher/entrypoint issue rather than a fabric failure. The direct-rank measurements are recorded in `reports/scaling.csv`.

The attempt confirms why cross-node scaling remains blocked: the hosts do not share a pinned PyTorch/NCCL environment. Provision an identical runtime or digest-pinned CUDA image on every worker, repeat the same command with `NCCL_DEBUG=INFO`, and only then admit 8/12-GPU DDP measurements.
