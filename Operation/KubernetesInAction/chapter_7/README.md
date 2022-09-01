# Chapter 7 ConfigMaps and Secrets: configuring applications

## 7.1 Configuring containerized applications

You can configure your apps by

+ Passing command-line arguments to containers.
+ Setting custom environment variables for each container.
+ Mounting configuration files into containers through a special type
of volume.

## 7.2 Passing command-line arguments to containers

In Kubernetes, when specifying a container, you can choose to override
both `ENTRY-POINT` AND `CMD`.

```yaml
kind: Pod
spec:
  containers:
  - image: some/image
    command: ["bin/command"]
    args: ["agr1", "arg2", "arg3"]
```

## 7.3 Setting environment variables for a container

```yaml
kind: Pod
spec:
  containers:
  - image: fortune
    env:
    - name: INTERVAL
      value: "30"
  name: html-generator
```

## 7.4 Decoupling configuration with a ConfigMap

### 7.4.1 Introducing ConfigMaps

Kubernetes allows separating configuration options into a separate
object called a ConfigMap, which is a map containing key/value pairs with
the values ranging from short literals to full config files.

An application doesn't need to read the ConfigMap directly or
even know that it exists. The contents of the map are instead passed
to containers as either environment variables or as files in a volume.

### 7.4.2 Creating a ConfigMap

```sh
kubectl create configmap fortune-config --from-literal=sleep-interval=25
```

ConfigMaps usually contain more than one entry. To create a
ConfigMap with multiple literal entries, you add multiple
`--from-literal` arguments.

ConfigMaps can also store coarse-grained config data, such as complete
config giles.

```sh
kubectl create configmap my-config --from-file=config-file.conf
```

Instead of importing each file individually, you can even import
all files from a file directory.

```sh
kubectl create configmap my-config --from-file=/path/to/dir
```

### 7.4.3 Passing a ConfigMap entry to a container as an environment variable

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-env-from-configmap
  - image: fortune
    env:
    - name: INTERVAL
      valueFrom:
        configMapKeyRef:
          name: fortune-config
          key: sleep-interval
```

### 7.4.4 Passing all entries of a ConfigMap as environment variables at once

```yaml
spec:
  containers:
  - image: some-image
    envFrom:
    - prefix: CONFIG_
      configMapRef:
        name: my-config-map
```

### 7.4.5 Passing a ConfigMap entry as a command-line argument

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-args-from-configmap
spec:
  containers:
  - image: fortune:args
    env:
    - name: INTERVAL
      valueFrom:
        configMapKeyRef:
          name: fortune-config
          key: sleep-interval
    args: ["$(INTERVAL)"]
```

### 7.4.6 Using a configMap volume to expose ConfigMap entries as files

A `configMap` volume will expose each entry of the ConfigMap as a file.

Let's say you want your Nginx server to compress responses it sends
to the client. To enable compression, the config file for Nginx needs
to look like the following listing.

```text
server {
  listen              80;
  server_name         www.kubia-example.com;
  gzip on;
  gzip_types text/plain application/xml;
  location / {
    root   /usr/share/nginx/html;
    index  index.html index.htm;
  }
}
```

Create a new directory called `configmap-files` and store the Nginx
config above into `configmap-files/my-nginx-config.conf`. To make
the ConfigMap also contain the `sleep-interval` entry, add a plain text
file called `sleep-interval` to the same directory and store the number 25 in it.

Now create a ConfigMap from all files in the directory:

```sh
kubectl create configmap fortune-config --from-file=configmap-files
```

#### Using the ConfigMap's entries in a volume

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-configmap-volume
spec:
  containers:
  - image: nginx:alpine
    name: web-server
    volumeMounts:
    - name: config
      mountPath: /etc/nginx/conf.d
      readOnly: true
    ...
  volumes:
  - name: config
    configMap:
      name: fortune-config
```
