# Phase 0 cluster inventory (2026-10-29 UTC)

Collected from `foureyes` using passwordless SSH (`~/.ssh/config` has aliases `ripper`, `work`, `loki`, identity `~/.ssh/id_ed25519_dora_study`, user `catid`). All aliases resolved and accepted batch-mode SSH. `/etc/hosts`/DNS advertise `.lan` names but `hostname -I` provided the 10.10.0.x addresses below.

| host | address | GPUs | GPU/driver | VRAM each | RAM | `/` free | `/home` free | PyTorch on host |
|---|---|---:|---|---:|---:|---:|---:|---|
| foureyes | 10.10.0.216 | 4 | RTX PRO 6000 Blackwell Max-Q, 615.71.10 | 97,887 MiB | 1.0 TiB | 2.4 TiB | 638 GiB | no host `torch` (`python` absent; `/usr/bin/python3` has no torch) |
| ripper | 10.10.0.161 | 4 | RTX PRO 6000 Blackwell Max-Q, 615.71.10 | 97,887 MiB | 1.0 TiB | 2.5 TiB | 1.8 TiB | no host `torch` |
| work | 10.10.0.236 | 2 | RTX PRO 6000 Blackwell, 615.71.10 | 97,887 MiB | 503 GiB | 3.3 TiB | 4.6 TiB | Python 3.12.3, `torch 2.11.0.dev20251230+cu130`, CUDA 13.0, available=True |
| loki | 10.10.0.113 | 2 | RTX PRO 6000 Blackwell Max-Q, 615.71.10 | 97,887 MiB | 1.0 TiB | 3.3 TiB | 9.4 TiB | no host `torch` |

All machines run kernel `6.8.0-147-generic`; CPUs are 128 logical cores, one NUMA node, AMD Threadripper Pro 9985WX on foureyes/ripper/loki and Threadripper 9980X on work. `nvidia-smi topo -m` reports `NODE` between every local GPU pair (PCIe/host-bridge topology; no NVLink entries). No InfiniBand devices (`/dev/infiniband` absent, `ibstat` absent). Active physical NICs are 10.10.0.x Ethernet (`enp211s0f0np0` on foureyes/ripper/loki; `enp175s0` on work); no second active fabric was observed. `iperf3` is installed only on ripper. Docker is available on all hosts (29.9.2 except loki 29.1.3).

Clock and access: `timedatectl` reports `NTPSynchronized=yes` and `systemd-timesyncd active` on every host; sampled UTC epoch was identical (1793304459). SSH batch commands succeed in <1 s. ICMP from ripper/work/loki to foureyes 10.10.0.216 succeeds with 0% loss and ~0.16--0.52 ms RTT; TCP/22 and 8787 reachable. A temporary rendezvous listener bound to 10.10.0.216:29517 was not reachable (connection refused after the listener exited before probes), so a persistent-port test remains to be run. 

Source tree: `/home/catid/pcalm` exists only on foureyes (the project directory was created there); no shared `/home` mount is present (`findmnt -t nfs,nfs4` empty), and the path is absent on ripper/work/loki. Code must be copied/checked out per host or run from a container/image. `/home` utilization is already 82--83% on foureyes/ripper; use `/` NVMe or other high-free-space paths for bulk artifacts.

Constraints before distributed training: install/use a pinned project environment on every worker (host Python environments differ/missing); choose Ethernet NCCL settings after an actual all-reduce benchmark; copy source and dataset metadata to workers; run a persistent rendezvous connectivity test. Do not assume foureyes host Python can launch GPU jobs without the project container/environment.

Persistent rendezvous connectivity was subsequently verified: a temporary TCP listener on foureyes `10.10.0.216:29517` accepted clients from ripper (`10.10.0.161`), work (`10.10.0.236`), and loki (`10.10.0.113`), each receiving `OK`. This validates basic cross-node TCP reachability for a later `torchrun` rendezvous (NCCL performance still unmeasured).
