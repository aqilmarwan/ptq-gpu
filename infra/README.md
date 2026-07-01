# infra — ship to EKS (GPU), images on Docker Hub

Registry is **Docker Hub** (no ECR). AWS is used only for the EKS cluster and
`kubectl` access. **EKS GPU nodes cost money — tear the cluster down when done.**

## 0. One-time CI secrets / variables (GitHub repo settings)

Secrets:
| Name                 | Value                                                |
| -------------------- | ---------------------------------------------------- |
| `DOCKERHUB_USERNAME` | your Docker Hub username                              |
| `DOCKERHUB_TOKEN`    | a Docker Hub access token                            |
| `AWS_ROLE_ARN`       | IAM role ARN the GitHub OIDC provider can assume      |

Variables:
| Name                  | Value                                               |
| --------------------- | --------------------------------------------------- |
| `DEPLOY_ENABLED`      | `true` to enable the deploy job                     |
| `NEXT_PUBLIC_API_URL` | browser-reachable inference URL (the LB, see step 4)|

`.github/workflows/ci.yml` then runs: test → build & push both images to Docker
Hub → (if enabled) deploy to EKS.

## 1. Create the GPU cluster

```bash
eksctl create cluster -f infra/eksctl-cluster.yaml      # ~15-20 min
kubectl get nodes -L role                                # cpu-general + gpu-a10g
```

eksctl installs the NVIDIA device plugin for the GPU node group automatically, so
`nvidia.com/gpu` is schedulable. Verify:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

## 2. First deploy

Replace the image placeholder, or just let CI do `set image`. Manually:

```bash
# point the manifests at your Docker Hub user
sed -i '' "s/DOCKERHUB_USERNAME/<you>/g" infra/k8s/*.yaml   # macOS sed

kubectl apply -f infra/k8s/
kubectl -n quant-studio get pods -w
```

## 3. Get the inference URL

```bash
kubectl -n quant-studio get svc inference \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

## 4. Wire the web app to it

Set repo variable `NEXT_PUBLIC_API_URL=http://<inference-lb-hostname>` and push to
`main` (or re-run the workflow) so the web image is rebuilt with that URL baked
in. Then grab the web URL:

```bash
kubectl -n quant-studio get svc web \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

## 5. Watch autoscaling (the screenshot)

```bash
kubectl -n quant-studio get hpa inference -w
# drive load against the inference LB, then:
kubectl -n quant-studio get pods -l app=inference -w
```

The HPA scales 1→3 on CPU; new GPU pods pull in new `gpu-a10g` nodes (maxSize 3).

## 6. TEAR DOWN (stop the bill)

```bash
kubectl delete -f infra/k8s/                 # releases the ELBs first
eksctl delete cluster -f infra/eksctl-cluster.yaml --wait
```

> Deleting the cluster before the Services removes their ELBs can orphan load
> balancers — delete the manifests first, then the cluster.
