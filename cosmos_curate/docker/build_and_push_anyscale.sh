TAG=${1:-2}
COSMOS_TAG=${2:-2}
REGISTRY=aws
REPO_ROOT=$HOME/git/cosmos-curate
IMAGE=anyscale-cosmos-curate

docker build \
  --ulimit nofile=65536 \
  --progress=auto \
  --network=host \
  --build-arg COSMOS_TAG=${COSMOS_TAG} \
  -f $REPO_ROOT/docker/anyscale.Dockerfile \
  -t ${IMAGE}:$TAG \
  -t ${IMAGE}:latest \
  $REPO_ROOT

SRC=${IMAGE}:${TAG}

if [ "$REGISTRY" = "aws" ]; then
  AWS_ACCOUNT=367974485317
  AWS_REGION=us-west-2
  AWS_REPO=wagner-west-2
  DST_BASE=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${AWS_REPO}
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
else
  PROJECT_ID=troubleshootingorg-gcp-pub
  REGION=us-central1
  REPO=wagner-docker
  DST_BASE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}
fi

docker tag ${SRC} ${DST_BASE}:${TAG}
docker push ${DST_BASE}:${TAG}
docker tag ${SRC} ${DST_BASE}:latest
docker push ${DST_BASE}:latest
