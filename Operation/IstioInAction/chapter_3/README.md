# Chapter 3 Istio's data plane: The Envoy proxy

## 3.1 What is the Envoy proxy?

The Envoy proxy is specifically an application-level proxy that we can insert
into the request path of our applications to provide things like service discovery,
load balancing, and health checking, but Envoy can do more than that.

As a proxy, Envoy is designed to shield developers from networking concerns by
running out-of-process from applications. This means any application written in
any programming language or with any framework can take advantage of these
features.

### 3.1.1 Envoy's core features

Envoy has many features useful for inter-service communication. To help understand
these features and capabilities, you should be familiar with the following Envoy
concepts at a high level:

+ *Listeners*: Expose a port to the outside world to which applications can connect.
+ *Routes*: Routing rules for how to handle traffic that comes in on *listeners*.
+ *Clusters*: Specify upstream services to which Envoy can route traffic.

Envoy uses terminology similar to that of other proxies when conveying traffic directionality.
For example, traffic comes into a *listener* from a *downstream* system. This traffic is
routed to one of Envoy's *clusters*, which is responsible for sending that traffic to an
*upstream* system.

## 3.2 Configuring Envoy

Envoy is driven by a configuration file in either JSON or YAML format. The configuration
file specifies listeners, routes, and clusters as well as server-specific settings. Envoy's
v3 configuration API is built on gRPC. Envoy and implements of the v3 API can take
advantage of streaming capabilities when calling the API and reduce the time required for
Envoy proxies to converge on the correct configuration.

### 3.2.1 Static configuration

We can specify listeners, route rules, and clusters using Envoy's configuration file. The
following is a very simple Envoy configuration:

```yaml
static_resources:
  listeners:
  - name: httpbin-demo
    address:
      socket_address: {address: 0.0.0.0, port_value: 15001}
    filter_chains:
    - filters:
      - name: envoy.http_connection_manager
        config:
          stat_prefix: egress_http
          route_config:
            name: httpbin_local_route
            virtual_hosts:
            - name: httpbin_local_service
              domains: ["*"]
              routes:
              - match: { prefix: "/" }
                route:
                  auto_host_rewrite: true
                  cluster: httpbin_service
          http_filters:
          - name: envoy.router
  clusters:
    - name: httpbin_service
      connect_timeout: 5s
      type: LOGICAL_DNS
      dns_lookup_family: V4_ONLY
      lb_policy: ROUND_ROBIN
      hosts: [ {socket_address: {address: httpbin, port_value: 8000}} ]
```

### 3.2.2 Dynamic configuration

Envoy can use a set of APIs to do inline configuration updates without any downtime or
restarts. It just needs a bootstrap configuration file that points the configuration to
the correct discovery service APIs; the rest is configured dynamically. Envoy uses the
following APIs for dynamic configuration:

+ *Listener discovery service*: An API that allows Envoy to query what listeners should
be exposed on this proxy.
+ *Route discovery service*: Part of the configuration for listeners that specifiers which
routes to use.
+ *Cluster discovery service*: An API that allows Envoy to discover that clusters and
respective configuration for each cluster this proxy should have.
+ *Endpoint discovery service*: Part of the configuration for clusters that specifies
which endpoints to use for a specific cluster..
+ *Secret discovery service*: An API used to distribute certificates.
+ *Aggregate discovery service*: A serialized stream of all the changes to the rest of the
APIs.
