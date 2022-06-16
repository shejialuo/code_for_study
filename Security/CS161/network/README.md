# Network

There are three types of attacks we concern about.

+ Confidentiality: eavesdropping (aka sniffing)
+ Integrity: injection of spoofed packets
+ Availability: delete legit packets (e.g., jamming)

## TCP

### Eavesdropping

For subnets using broadcast technologies, attacker can eavesdrop.

+ Each attached system's NIC can capture any communication on the subnet.
+ Tools: `tcpdump`/`windump`, wireshark.

### Spoofing

Attacker can inject spoofed packets, and lie about the source address:

+ With physical access to local network, attacker can create any packet they like.
+ Particularly powerful when combined with eavesdropping, because attacker can
understand exact state of victim's communication and craft their spoofed traffic
to match it.

### On-path vs Off-path Spoofing

+ On-path attackers can see victim's traffic => spoofing is easy.
+ Off-path attackers can't see victim's traffic
  + They have to resort to blind spoofing
  + Often must guess/infer header values to succeed.
  + There is a possibility to brute force.

### Abrupt Termination

![A sends RST to B](https://s2.loli.net/2022/06/16/asFXVDLJrMtHuh1.png)

When A sends a TCP packet with RST flag to B and sequence number fits, connection
is immediately terminated.

So the attacker can inject RST packets and block connection. Well, unfortunately,
the GFW does this to TCP requests. The on-path attacker and man-in-the-middle
can do this attack.

### Data Injection

If attacker knows *ports* and *sequence numbers* (on-path attacker), attacker can
inject data into any TCP connection, termed *connection hijacking*.

### Blind Hijacking

It is also possible for off-path attacker to inject into a TCP connection even if
they can't see the port and sequence numbers because they could infer or guess
the port and sequence numbers.

### Blind Spoofing on TCP handshake

When the attacker wants to do blind spoofing on TCP handshake, the attacker could
send a packet to the server. There are two situations here:

+ If alleged client receives the response from the server, the alleged client
will be confused, sends a RST back to server. So attacker may need to hurry!
But firewalls may inadvertently stop this reply to the alleged client so
it never ends the RST.
+ It's a problem for attacker to get the value of the `seq` field of the response,
in the older time, the value could be guessed, now the value is total random. This
is the reason why `seq` fields need to random.

### Summary

+ An attacker who can observe your TCP connection can manipulate it:
  + Forcefully terminate by forging a RST packet.
  + Inject (spoof) data into either direction by forging data packets
  + Remains a major threat today
+ Blind spoofing no longer a thread
  + Due to randomization of TCP initial sequence numbers.

## DHCP

There are many threats for DHCP

+ Substitute a fake DNS server
  + Redirect any of a host's lookups to a machine of attacker's choice
+ Substitute a fake gateway router
  + Intercept all of a host's off-subnet traffic
  + Relay contents back and forth between host and remote server
+ An invisible Man In The Middle

Thus, we can conclude the following ideas:

+ Broadcast protocols inherently at risk of local attacker spoofing.
+ When initializing, systems are particularly vulnerable because they can lack
a trusted foundation to build upon.
+ MITM attacks insidious because no indicators they're occurring.

## TLS
