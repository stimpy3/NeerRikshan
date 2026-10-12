#!/usr/bin/env bash
# Start the yearly update immediately instead of waiting for 15 November -
# the same Fargate task EventBridge would start. Then follow its logs:
#
#   bash deploy/run_now.sh           # current year
#   bash deploy/run_now.sh 2026      # a specific season

set -euo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
STACK="${STACK:-neerrikshan}"
export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
YEAR="${1:-$(date -u +%Y)}"

out() {
    aws cloudformation describe-stacks --stack-name "$STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

TASK_ARN="$(aws ecs run-task \
    --cluster "$(out ClusterName)" \
    --task-definition "$(out TaskDefinitionArn)" \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$(out PipelineSubnetIds)],securityGroups=[$(out PipelineSecurityGroupId)],assignPublicIp=ENABLED}" \
    --overrides "{\"containerOverrides\":[{\"name\":\"pipeline\",\"environment\":[{\"name\":\"YEAR\",\"value\":\"$YEAR\"}]}]}" \
    --query 'tasks[0].taskArn' --output text)"

echo "started pipeline for $YEAR: $TASK_ARN"
echo "following logs (Ctrl+C stops watching, not the task):"
aws logs tail "$(out PipelineLogGroup)" --follow --since 1m
