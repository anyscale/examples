TAG=1
REPO_ROOT=$HOME/git/cosmos-curate
docker build \
  --ulimit nofile=65536 \
  --progress=auto \
  --network=host \
  -f $REPO_ROOT/docker/cosmos-curate.Dockerfile \
  -t cosmos-curate:$TAG \
  -t cosmos-curate:latest \
  $REPO_ROOT
