# Artifact locations

Pilot checkpoints are preserved on the local NVMe path `/var/tmp/pcalm-runs` rather than in the Git working tree. The `runs/<run-id>/checkpoint-*.pt` entries are symlinks to those files, while JSONL metrics and manifests remain tracked under `runs/`. Future long runs should set an explicit NVMe checkpoint root before launch.
