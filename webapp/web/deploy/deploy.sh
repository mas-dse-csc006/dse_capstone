#!/usr/bin/env bash

NS=sharknado-recsys
docker-compose build
docker tag webapp_web gcr.io/com-eggie5-blog/sharknado-web
gcloud docker push gcr.io/com-eggie5-blog/sharknado-web
kubectl apply -f k8s/ --namespace=$NS
# kubectl create secret generic db-secret --from-literal=username=postgres --from-literal=password= --namespace $NS


# gcloud compute disks create sharknado-db-disk --type pd-standard  --zone us-west1-a