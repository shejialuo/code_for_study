# Chapter 2 First steps with Istio

## 2.1 Deploying Istio on Kubernetes

Simple.

## 2.2 Getting to know the Istio control plane

For Istio, the control plane provides the following functions:

+ APIs for operators to specify desired routing/resilience behavior
+ APIs for the data plane to consume their configuration
+ A service discovery abstraction for the data plane
+ APIs for specifying usage policies
+ Certificate issuance and rotation
+ Workload identity assignment
+ Unified telemetry collection
+ Service-proxy sidecar injection
+ Specification of network boundaries and how to access them

The bulk of these responsibilities is implemented in a single control-plane component
called `istiod`. Below shows `istiod` along with gateways responsible for ingress
traffic and egress traffic. We also see supporting components that are typically integrated
with a service mesh to support observability or security use cases.

![Istio control plane and supporting components](https://s2.loli.net/2022/12/11/SwWQc7FaRtmgMGe.png)

### 2.2.1 Istiod

Istio's control-plane responsibilities are implemented in the `istiod` component. `istiod`
sometimes referred to as Istio Pilot, is responsible for taking higher-level Istio configurations
specified bt the user/operator and turning them into proxy-specific configurations for each
data-plane service proxy.

![Istio control plane](https://s2.loli.net/2022/12/11/4A3VzFkB5xjoeEI.png)

### 2.2.2 Ingress and egress gateway

For our applications and services to provide anything meaningful, they need to interact with
applications outside of our cluster. To do this, operators need to configure Istio to allow
traffic into the cluster and be very specific about what traffic is allowed to leave the
cluster.

Below shows the Istio components that provide this functionality: `istio-ingressgateway` and
`istio-egressgateway`.

![Incoming and outgoing traffic flow through istio gateways](https://s2.loli.net/2022/12/11/cfZN1RMFwVkLzne.png)

These components are really Envoy proxies that can understand Istio configurations.
Although they are not technically part of the control plane, they are instrumental in
any real-word usage of a service-mesh.
