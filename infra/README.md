# infra — EKS cluster + manifests

## Layout

```
eksctl-cluster.yaml   # cluster + general/gpu node groups + IRSA for the controllers
bootstrap.sh          # in-cluster prerequisites (metrics-server, ALB controller, autoscaler)
k8s/                  # namespace, deployments, services (ClusterIP), HPA, ingress (ALB+TLS)
```

## First-time bring-up (bootstrap order matters)

```bash
# 1. Cluster + node groups + IRSA service accounts
eksctl create cluster -f infra/eksctl-cluster.yaml

# 2. In-cluster controllers the manifests depend on
EKS_CLUSTER=quant-studio AWS_REGION=us-east-1 infra/bootstrap.sh

# 3. Build + publish the TensorRT engine bundles to S3 (once, on a GPU box):
#      STUDIO_ENGINE_S3_URI=s3://<your-bucket> \
#        python pipelines/build_flow.py run --sync --engine-s3 s3://<your-bucket>
#    then point the deployment + eksctl IRSA at that bucket (replace REPLACE_ME-*).

# 4. Namespace + workloads (CI normally does this; manual is fine too)
kubectl apply -f infra/k8s/
```

Then wire DNS/TLS. The **web app is on Vercel**, so the ALB fronts only the
inference API (`k8s/ingress.yaml`):

1. Request an **ACM certificate** for `api.<domain>`; put its ARN in
   `k8s/ingress.yaml` (`ACM_CERT_ARN`) and replace `example.com` with your domain.
2. `kubectl -n quant-studio get ingress quant-studio` → copy the ALB hostname.
3. Route 53 **ALIAS** record `api.<domain>` → that ALB.
4. On **Vercel**, set `NEXT_PUBLIC_API_URL=https://api.<domain>`; on the inference
   deployment, set `STUDIO_CORS_ORIGINS` to your Vercel origin. Now the browser
   (Vercel) calls the API (ALB) over HTTPS with no mixed-content/CORS block.

> Fully self-hosting instead of Vercel? Add an apex-host rule routing to the
> in-cluster `web` Service in `k8s/ingress.yaml` (the Deployment/Service already
> exist), and point `NEXT_PUBLIC_API_URL` at `https://api.<domain>` at build time.

## Why the ordering

- The **HPA** (`k8s/inference-hpa.yaml`) reads CPU from **metrics-server** — install
  it (step 2) or the HPA reports `<unknown>` and never scales.
- The **Ingress** (`k8s/ingress.yaml`) is inert until the **AWS Load Balancer
  Controller** exists to reconcile it into an ALB (step 2).
- The **cluster-autoscaler** is what actually adds `gpu-l40s` nodes when the HPA
  wants more inference replicas than fit; its IAM permissions come from the IRSA
  service account eksctl created in step 1.

## Tear down

```bash
kubectl delete -f infra/k8s/            # releases the ALB
eksctl delete cluster -f infra/eksctl-cluster.yaml
```

Delete the k8s resources first so the ALB Controller removes the ALB before the
cluster (and its node roles) disappear — otherwise the load balancer can leak.
