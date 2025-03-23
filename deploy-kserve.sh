#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing Knative Serving CRDs..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-crds.yaml

echo "Installing Knative Serving core components..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.16.0/serving-core.yaml

echo "Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.1/cert-manager.yaml

echo "Installing Istio..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/istio.yaml

echo "Installing Istio integration for Knative..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.16.0/net-istio.yaml

echo "Installing KServe v0.13.1..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.1/kserve.yaml

#echo "Installing KServe v0.14.0..."
#kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.14.0/kserve.yaml

echo "Waiting for 1 minute after applying kserve.yaml..."
sleep 60

kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.1/kserve-cluster-resources.yaml

# Wait for KServe controller to be ready
echo "Waiting for KServe controller to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/kserve-controller-manager -n kserve

# Create TorchServe InferenceService
echo "Creating TorchServe InferenceService..."
cat <<EOF > torchserve.yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "torchserve"
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 3
    model:
      modelFormat:
        name: pytorch
      # storageUri: gs://kfserving-examples/models/torchserve/image_classifier/v1
      storageUri: gs://sampletorch/torchserve
EOF
kubectl apply -f torchserve.yaml

# Create Istio Gateway
echo "Creating Istio Gateway for TorchServe..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: torchserve-gateway
  namespace: default
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "torchserve.default.svc.cluster.local"
EOF

# Create VirtualService
echo "Creating VirtualService for TorchServe..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: torchserve-external
  namespace: default
spec:
  hosts:
  - "torchserve.default.svc.cluster.local"
  gateways:
  - torchserve-gateway
  http:
  - match:
    - uri:
        prefix: /v1/models
    route:
    - destination:
        host: torchserve-predictor.default.svc.cluster.local
        port:
          number: 80
EOF

# Wait for InferenceService to be ready
echo "Waiting for TorchServe InferenceService to be ready (this might take several minutes)..."
kubectl wait --for=condition=ready --timeout=600s inferenceservice/torchserve

# Get ingress information
echo "Retrieving Ingress information..."
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)

echo "============================================================"
echo "Setup complete! Your TorchServe service is ready."
echo "Ingress Host: ${INGRESS_HOST}"
echo "Ingress Port: ${INGRESS_PORT}"
echo "Service Hostname: ${SERVICE_HOSTNAME}"
echo "============================================================"
#echo "Example curl command for inference:"
#echo "curl -v -H \"Host: ${SERVICE_HOSTNAME}\" -H \"Content-Type: application/json\" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/mnist:predict -d @mnist01.json"
#echo "============================================================"
#echo "Note: Ensure you have the mnist01.json file in your current directory before running the curl command."

# # Create a service account (if you don't already have one)
# gcloud iam service-accounts create kserve-sa

# # Grant the service account access to your bucket
# gsutil iam ch serviceAccount:kserve-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectViewer gs://sampletorch/

# # Create and download a key for this service account
# gcloud iam service-accounts keys create kserve-sa-key.json --iam-account=kserve-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com

# kubectl create secret generic gcs-creds --from-file=gcloud-application-credentials.json=kserve-sa-key.json
