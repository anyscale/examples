REPO_ROOT=$HOME/git/cosmos-curate
cosmos-curate image build \
  --curator-path "${REPO_ROOT}" \
  --image-name cosmos-curate \
  --image-tag 1 \
  --dry-run \
  --verbose \
  --dockerfile-output-path "${REPO_ROOT}/docker/cosmos-curate.Dockerfile"
