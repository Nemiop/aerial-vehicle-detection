#!/usr/bin/env bash
set -euo pipefail

CVAT_DIRECTORY="/mnt/d/Ruslan/Test_Project/tools/cvat"

systemctl start docker
cd "$CVAT_DIRECTORY"
docker compose up -d

exec sleep infinity
