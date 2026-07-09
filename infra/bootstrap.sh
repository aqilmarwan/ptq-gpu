#!/usr/bin/env bash
#
# One-time, in-cluster prerequisites that the manifests in infra/k8s depend on
# but that don't ship with EKS:
#
#   * metrics-server           -> the inference HPA reads CPU metrics from it
#   * AWS Load Balancer Ctrl    -> turns infra/k8s/ingress.yaml into a real ALB
#   * cluster-autoscaler        -> grows the gpu-l40s node group past desired size
#   * NVIDIA device plugin      -> advertises nvidia.com/gpu (usually auto-installed
#                                  by eksctl for GPU node groups; applied here as a
#                                  safety net)
#
# Run once after `eksctl create cluster -f infra/eksctl-cluster.yaml`. The IRSA
# service accounts (aws-load-balancer-controller, cluster-autoscaler) are created
# by eksctl, so the Helm installs below reuse them (serviceAccount.create=false).
#
# Requires: kubectl, helm, and a kubeconfig already pointed at the cluster.
set -euo pipefail

CLUSTER="${EKS_CLUSTER:-quant-studio}"
REGION="${AWS_REGION:-asia-southeast-5}"

echo ">> metrics-server"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

echo ">> NVIDIA device plugin (safety net; eksctl usually installs this)"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
helm repo add autoscaler https://kubernetes.github.io/autoscaler >/dev/null 2>&1 || true
helm repo update >/dev/null

echo ">> AWS Load Balancer Controller"
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName="$CLUSTER" \
  --set region="$REGION" \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

echo ">> cluster-autoscaler"
helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set "autoDiscovery.clusterName=$CLUSTER" \
  --set "awsRegion=$REGION" \
  --set rbac.serviceAccount.create=false \
  --set rbac.serviceAccount.name=cluster-autoscaler

echo ">> done. Verify:"
echo "   kubectl top nodes                       # metrics-server up"
echo "   kubectl -n kube-system get deploy aws-load-balancer-controller cluster-autoscaler"
