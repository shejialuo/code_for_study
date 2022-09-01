# Chapter 9 Deployments: updating applications declaratively

## 9.1 Updating applications running in pods

Let's start off with a simple example shown below.

![The basic outline of an application](https://s2.loli.net/2022/09/01/9zkojYEVOgbSGq5.png)

Initially, the pods run the first version of your application. You then
develop a newer version of the app. You'd like to replace all
the pods with this new version.

You have two ways of updating all those pods. You can do one of the following:

+ Delete all existing pods first and then start the new ones.
+ Start new ones and, once they're up, delete the old ones.

### 9.1.1 Deleting pods and replacing them with new ones

If you have a ReplicationController managing a set of `v1` pods,
you can easily replace them by modifying the pod template so it
refers to version `v3` of the image and then deleting the old pod
instances.

### 9.1.2 Spinning up new pods and then deleting the old ones

#### Switching from the old to the new version at once

Pods are usually fronted by a Service. It's possible to have the
Service front only the initial version of the pods while you bring up
the pods running the new version. Then once all the new pods are up,
you can change the Service'label selector and have the Service switch
over to the new pods, as shown below. This is called a *blue-green deployment*.

![Switching a Service from the old pods to the new ones](https://s2.loli.net/2022/09/01/6ETDSaPQevsorL7.png)

#### Performing a rolling update

Instead of bringing up all thew new pods and deleting the old pods at once,
you can also perform a rolling update, which replaces pods step by step.
You can do this by slowly scaling down the previous ReplicationController
and scaling up the new one. In this case, you'll want the Service's
pod selector to include both the old and the new pods.

![A rolling update of pods using two ReplicationControllers](https://s2.loli.net/2022/09/01/QceudjKHWhXm31B.png)

## 9.2 Performing an automatic rolling update with a ReplicationController

Instead of performing rolling updates using ReplicationControllers manually,
you can have `kubectl` perform them. Using `kubectl` to perform the update
makes the process much easier, but this is now an outdated way
of updating apps.

### 9.2.1 Running the initial version of the app

#### Create the v1 app

```js
// v1/app.js
const http = require('http');
const os = require('os');

console.log("Kubia server starting...");
var handler = function(request, response) {
  console.log("Received request from " + request.connection.remoteAddress);
  response.writeHead(200);
  response.send("This is v1 running in pod " + os.hostname() + "\n");
}
var www = http.createServer(handler);
www.listen(8080);
```

#### Running the app

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: kubia-v1
spec:
  replicas: 3
  template:
    metadata:
      name: kubia
      labels:
        app: kubia
    spec:
      containers:
      - image: kubia:v1
        name: nodejs
        imagePullPolicy: Never
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  type: LoadBalancer
  selector:
    app: kubia
  ports:
  - port: 80
    targetPort: 8080
```

### 9.2.2 Performing a rolling update with kubectl

Well, it is deprecated. Omit

## 9.3 Using Deployments for updating apps declaratively

When you create a Deployment, a ReplicaSet resource is created underneath.
When using a Deployment, the actual pods are create and managed by
the Deployment's ReplicaSets, not by the Deployment directly.

### 9.3.1 Creating a Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kubia
  template:
    metadata:
      name: kubia
      labels:
        app: kubia
    spec:
      containers:
      - image: kubia:v1
        name: nodejs
        imagePullPolicy: Never
```

```sh
kubectl create -f kubia-deployment-v1.yaml --record
```

### 9.3.2 Updating a Deployment

The only thing you need to do is modify the pod template defined
in the Deployment resource and Kubernetes will take all the steps
necessary to get the actual system state to what's defined in the resource.

```sh
kubectl set image deployment kubia nodejs=kubia:v2
```
