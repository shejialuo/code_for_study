# Chapter 8 Accessing pod metadata and other resources from applications

## 8.1 Passing metadata through the Downward API

### 8.1.1 Understanding the available metadata

The Downward API enables you to expose the pod's own metadata to the
processes running inside the pod. Currently, it allows you to pass
the following information to your containers:

+ The pod's name
+ The pod's IP address
+ The namespace the pod belongs to
+ The name of the pod is running on
+ The name of the service account the pod is running under.
+ The CPU and memory requests for each container.
+ The CPU and memory limits for each container.
+ The pod's labels.
+ The pod's annotations.

### 8.1.2 Exposing metadata through environment variables

First, let's look at how you can pass the pod's and container's metadata
to the container through environment variables.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: downward
spec:
  containers:
  - name: main
    image: busybox
    command: ["sleep", "9999999"]
    resources:
      requests:
        cpu: 15m
        memory: 100Ki
      limits:
        cpu: 100m
        memory: 4Mi
    env:
    - name: POD_NAME
      valueFrom:
        fieldRef:
          fieldPath: metadata.name
    - name: POD_NAMESPACE
      valueFrom:
        fieldRef:
          fieldPath: metadata.namespace
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
    - name: NODE_NAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
    - name: SERVICE_ACCOUNT
      valueFrom:
        fieldRef:
          fieldPath: spec.serviceAccountName
    - name: CONTAINER_CPU_REQUEST_MILLICORES
      valueFrom:
        resourceFieldRef:
          resource: requests.cpu
          divisor: 1m
    - name: CONTAINER_MEMORY_LIMIT_KIBIBYTES
      valueFrom:
        resourceFieldRef:
          resource: limits.memory
          divisor: 1Ki
```

### 8.1.3 Passing metadata through files in a downwardAPI volume

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: downward
  labels:
    foo: bar
  annotations:
    key1: value1
    key2: |
      multi
      line
      value
spec:
  containers:
  - name: main
    image: busybox
    command: ["sleep", "9999999"]
    resources:
      requests:
        cpu: 15m
        memory: 100Ki
      limits:
        cpu: 100m
        memory: 4Mi
    volumeMounts:
    - name: downward
      mountPath: /etc/downward
  volumes:
  - name: downward
    downwardAPI:
      items:
      - path: "podName"
        fieldRef:
          fieldPath: metadata.name
      - path: "podNamespace"
        fieldRef:
          fieldPath: metadata.namespace
      - path: "labels"
        fieldRef:
          fieldPath: metadata.labels
      - path: "containerCpuRequestMilliCores"
        resourceFieldRef:
          containerName: main
          resource: requests.cpu
          divisor: 1m
      - path: "containerMemoryLimitBytes"
        resourceFieldRef:
          containerName: main
          resource: limits.memory
          divisor: 1
```

## 8.2 Talking to the Kubernetes API server

### 8.2.1 Exploring the Kubernetes REST API

You can get API server's URL by running `kubectl cluster-info`

```sh
kubectl cluster-info
```

The `kubectl proxy` command runs a proxy server that accepts HTTP
connections on your local machine and proxies them to the API server
while taking care of authentication.

```sh
kubectl proxy
```

### 8.2.2 Talking to the API server form within a pod

To talk to the API server from inside a pod, you need to take care
of three things:

+ Find the location of the API server.
+ Make sure you're talking to the API server.
+ Authenticate with the server.
