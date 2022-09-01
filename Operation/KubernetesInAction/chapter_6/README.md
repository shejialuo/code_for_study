# Chapter 6 Volumes: attaching disk storage to containers

## 6.1 Introducing volumes

Kubernetes volumes are a component of a pod and are thus defined in
the pod's specification. They aren't a standalone Kubernetes object
and cannot be created or deleted on their own.

A wide variety of volume types is available. Several are generic,
while others are specific to the actual storage technologies used.
Here's a list of several of the available volume types:

+ `emptyDir`: A simple empty directory used for storing transient data.
+ `hostPath`: Used for mounting directories from the worker node's
filesystem into the pod.
+ `gitRepo`: A volume initialized by checking out the contents
of a Git repository.
+ `nfs`: An NFS share mounted into the pod.
+ `gcePersistentDisk`, `aswElasticBlockStore`, `azureDisk`.
+ Others.

## 6.2 Using volumes to share data between containers

### 6.2.1 Using an emptyDir volume

An `emptyDir` volume is especially useful for sharing files between containers
running in the same pod. But it can also be used by a single container
for when a container needs to write data to disk temporarily.

Now, first we should create a container image in local called `fortune`:

```sh
#!/bin/bash
trap "exit" SIGINT
mkdir /var/htdocs
while :
do
  echo $(date) Writing fortune to /var/htdocs/index.html
  /usr/games/fortune > /var/htdocs/index.html
  sleep 10
done
```

Next we create a Dockerfile.

```Dockerfile
FROM ubuntu:latest
RUN apt-get update; apt-get -y install fortune
ADD fortuneloop.sh /bin/fortuneloop.sh
RUN chmod u+x /bin/fortuneloop.sh
ENTRYPOINT /bin/fortuneloop.sh
```

#### Creating the pod

Create a file called `fortune-pod.yaml`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune
spec:
  containers:
  - image: fortune
    name: html-generator
    volumeMounts:
    - name: html
      mountPath: /var/htdocs
    imagePullPolicy: Never
  - image: nginx:alpine
    name: web-server
    volumeMounts:
    - name: html
      mountPath: /usr/share/nginx/html
      readOnly: true
    ports:
    - containerPort: 80
      protocol: TCP
  volumes:
    - name: html
      emptyDir: {}
```

The pod contains two containers and a single volume that's mounted in
both of them, yet a different paths.

#### Seeing the pod in action

```sh
kubectl port-forward fortune 8080:80
curl http://localhost:8080
```

#### Specifying the medium to use for the emptyDir

The `emptyDir` you used as the volume was created on the actual disk
of the worker node hosting your pod. You can tell Kubernetes to create
the `emptyDir` on a tmpfs filesystem. To do this, set the `emptyDir`'s
`medium` to `Memory` like this:

```yaml
volumes:
  - name: html
    emptyDir:
      medium: Memory
```

### 6.2.2 Using a Git repository as the starting point for a volume

A `gitRepo` volume is basically an `emptyDir` volume that gets populated
by cloning a Git repository and checking out a specific revision
when the pod is starting up.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gitrepo-volume-pod
spec:
  containers:
  - image: nginx:alpine
    name: web-server
    volumeMounts:
    - name: html
      volumeMounts:
      - name: html
        mountPath: /usr/share/nginx/html
        readOnly: true
    ports:
    - containerPort: 80
      protocol: TCP
  volumes:
  - name: html
    gitRepo:
      repository:
      revision: master
      directory: .
```

When you create the pod, the volume is first initialized as an empty
directory and then the specified Git repository is cloned into it.

#### Introducing sidecar containers

The Git sync process shouldn't run in the same container as the Nginx
web server, but in a second container: a *sidecar container*.

#### Wrapping up the gitRepo volume

A `gitRepo` volume, like the `emptyDir` volume, is basically a dedicated
directory created specifically for, and used exclusively by,
the pod that contains the volume. When the pod is deleted, the
volume and its contents are deleted. Other types of volumes, however,
don't create a new directory, but instead mount an existing external
directory into the pod's container's filesystem.

## 6.3 Accessing files on the worker node's filesystem

Certain system-level pods do need to either read the node's files
or use the node's filesystem to access the node's devices through
the file system. Kubernetes makes this possible through a `hostPath` volume.

A `hostPath` volume points to a specific file or directory on the
node's filesystem. Pods running on the same node and using the same
path in their `hostPath` volume see the same files.

## 6.4 Using persistent storage

When an application running in a pod needs to persist data to disk
and have that same data available even when the pod is rescheduled
to another node, it must be stored on some type of network-attached storage(NSA).

## 6.5 Decoupling pods from the underlying storage technology

All the persistent volume types we've explored so for have required
the developer of the pod to have knowledge of the actual network
storage infrastructure available in the cluster. This is against
the basic idea of Kubernetes.

Ideally, a developer deploying their apps on Kubernetes should
never have to know what kind of storage technology is used underneath,
the same way they don't have to know what type of physical servers
are being used to run their pods.

When a developer needs a certain amount of persistent storage for
their application, they can request it from Kubernetes.

### 6.5.1 Introducing PersistentVolumes and PersistentVolumeClaims

To enable apps to request storage in a Kubernetes cluster without
having to deal with infrastructure specifics, two new resources were introduced.
They are PersistentVolumes and PersistentVolumeClaims

Using a PersistentVolume inside a pod is a little more complex than
using a regular pod volume.

Instead of the developer adding a technology-specific volume to their
pod, it's the cluster administrator who sets up the underlying storage
and then registers it in Kubernetes by creating a PersistentVolume
resource through the Kubernetes API server.

When a cluster user needs to use persistent storage in one of their
pods, they first create a PersistentVolumeClaim manifest, specifying
the minimum size and the access mode they require.

### 6.5.2 Creating a PersistentVolume

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mongodb-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
    - ReadOnlyMany
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /tmp/mongodb
```

### 6.5.3 Claiming a PersistentVolume by creating a PersistentVolumeClaim

Claiming a PersistentVolume is a completely separate process from
creating a pod, because you want the same PersistentVolumeClaim to
stay available even if the pod is rescheduled.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mongodb-pvc
spec:
  resources:
    requests:
      storage: 1Gi
  accessModes:
  - ReadWriteOnce
  storageClassName: ""
```

### 6.5.4 Using a PersistentVolumeClaim in a pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mongodb
spec:
  containers:
  - image: mongo
    name: mongodb
    volumeMounts:
    - name: mongodb-data
      mountPath: /data/db
    ports:
    - containerPort: 27017
      protocol: TCP
  volumes:
    - name: mongodb-data
      persistentVolumeClaim:
        claimName: mongodb-pvc
```

## 6.6 Dynamic provisioning of PersistentVolumes

The cluster admin, instead of creating PersistentVolumes, can deploy
a PersistentVolume provisioner and define one or more storageClass objects to
let users choose what type of PersistentVolume they want. The
users can refer to the `StorageClass` in their PersistentVolumeClaims
and the provisioner will take that into account when provisioning the
persistent storage.
