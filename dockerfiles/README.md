# Dockerfiles Directory

Store Docker configurations and container setup files here.

## Contents

- **`Dockerfile`** — Main image definition for the application
- **`docker-compose.yaml`** — Multi-container orchestration (in project root)
- Docker-related configs and build scripts

## Usage

```bash
# Build image
docker compose build (--no-cache) --> builds all the containers and mounts a shared volume and network
                                      [no-cache flag is optional if you want to force a new build
                                      without cached layers]

# Run container
docker compose up (--build)       --> runs the containerized prediction pipeline,
                                      will also build the containers if they haven't been
                                      [--build flag is optional if you want to force a rebuild of containers]
```

## Phase

Phase 2 deliverable — Docker infrastructure setup.
