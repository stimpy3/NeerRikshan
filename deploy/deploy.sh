#!/usr/bin/env bash
# Deploy NeerRikshan to AWS (serverless). Run from the laptop in Git Bash:
#
#   bash deploy/deploy.sh
#
# Needs: AWS CLI v2 (logged in via `aws configure`), Docker running, Node, and
# server/.env with GEE_PROJECT, GEE_SERVICE_ACCOUNT_EMAIL and
# GEE_SERVICE_ACCOUNT_KEY_PATH filled in.
#
# Safe to re-run: every step updates in place. Steps:
#   1. CloudFormation stack (buckets, Lambda, API Gateway, CloudFront, ECS, schedule)
#   2. API code -> Lambda
#   3. pipeline image -> ECR
#   4. results, model and Earth Engine key -> private data bucket
#   5. website + thumbnails -> site bucket, CloudFront cache cleared

set -euo pipefail
# Git Bash rewrites arguments that look like paths ("/*", "/api") before the
# Windows aws.exe sees them. Turn that off.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STACK="${STACK:-neerrikshan}"
export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
BUILD="deploy/build"

envval() { grep -E "^$1=" server/.env | tail -1 | cut -d= -f2- | tr -d '\r"'; }
GEE_PROJECT="$(envval GEE_PROJECT)"
GEE_EMAIL="$(envval GEE_SERVICE_ACCOUNT_EMAIL)"
GEE_KEY="$(envval GEE_SERVICE_ACCOUNT_KEY_PATH)"
for v in GEE_PROJECT GEE_EMAIL GEE_KEY; do
    [[ -n "${!v}" ]] || { echo "server/.env is missing ${v/GEE_EMAIL/GEE_SERVICE_ACCOUNT_EMAIL}" >&2; exit 1; }
done
[[ -f "$GEE_KEY" ]] || { echo "key file not found: $GEE_KEY" >&2; exit 1; }

out() {
    aws cloudformation describe-stacks --stack-name "$STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
step() { echo; echo "=== $* ==="; }

step "1/5 CloudFormation stack '$STACK' in $AWS_REGION"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)"
SUBNETS="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" \
    --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
echo "default VPC $VPC, subnets $SUBNETS"
aws cloudformation deploy --stack-name "$STACK" \
    --template-file deploy/template.yaml \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
        "GeeProject=$GEE_PROJECT" \
        "GeeServiceAccountEmail=$GEE_EMAIL" \
        "VpcId=$VPC" \
        "SubnetIds=$SUBNETS"
DATA_BUCKET="$(out DataBucketName)"
SITE_BUCKET="$(out SiteBucketName)"
API_FUNCTION="$(out ApiFunctionName)"
REPO_URI="$(out RepositoryUri)"
DIST_ID="$(out DistributionId)"
SITE_URL="$(out SiteUrl)"

step "2/5 API code -> Lambda $API_FUNCTION"
rm -rf "$BUILD/lambda" && mkdir -p "$BUILD/lambda"
# Linux wheels, whatever this machine is: Lambda runs Amazon Linux on x86_64.
server/.venv/Scripts/python.exe -m pip install --quiet \
    --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 \
    --only-binary=:all: --target "$BUILD/lambda" -r server/requirements-lambda.txt
cp -r server/app server/lambda_handler.py "$BUILD/lambda/"
server/.venv/Scripts/python.exe -I -c "
import shutil; shutil.make_archive('$BUILD/api', 'zip', '$BUILD/lambda')"
aws lambda update-function-code --function-name "$API_FUNCTION" \
    --zip-file "fileb://$BUILD/api.zip" >/dev/null
aws lambda wait function-updated --function-name "$API_FUNCTION"
echo "uploaded $(du -h "$BUILD/api.zip" | cut -f1)"

step "3/5 pipeline image -> $REPO_URI"
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO_URI%%/*}"
docker build -t neerrikshan-pipeline:latest server
docker tag neerrikshan-pipeline:latest "$REPO_URI:latest"
docker push "$REPO_URI:latest"

step "4/5 results, model and key -> s3://$DATA_BUCKET"
for f in lake_change.json lake_timeseries.json lake_exposure.json; do
    aws s3 cp "server/data/$f" "s3://$DATA_BUCKET/data/$f" --content-type application/json
done
aws s3 cp server/data/models/unet_ndwi_fit_kv22.pt "s3://$DATA_BUCKET/models/unet_ndwi_fit_kv22.pt"
# private bucket, encrypted at rest; only the pipeline task's role reads secrets/
aws s3 cp "$GEE_KEY" "s3://$DATA_BUCKET/secrets/gee-key.json" >/dev/null
echo "Earth Engine key stored in s3://$DATA_BUCKET/secrets/"

step "5/5 website -> s3://$SITE_BUCKET"
( cd client && MSYS2_ENV_CONV_EXCL='VITE_API_BASE_URL' VITE_API_BASE_URL=/api npm run build )
grep -q 'baseURL:`/api`' client/dist/assets/*.js \
    || { echo "site build does not point at /api" >&2; exit 1; }
aws s3 sync client/dist "s3://$SITE_BUCKET/" --delete --exclude 'thumbnails/*'
aws s3 sync server/data/thumbnails "s3://$SITE_BUCKET/thumbnails/" --size-only
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths '/*' >/dev/null

step "done"
echo "website : $SITE_URL"
echo "api     : $SITE_URL/api/lakes/stats"
echo "(a brand-new CloudFront distribution can take ~5 minutes to answer)"
