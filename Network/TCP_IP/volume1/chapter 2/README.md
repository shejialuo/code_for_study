# Chapter 2 The Internet Address Architecture

## 2.1 Introduction

+ Every device connected to the Internet has at least one IP address.
+ When devices are attached to the global internet, they are assigned addresses that
must be coordinated so as to not duplicate other addresses in use on the network.

## 2.2 Expressing IP Address

In IPv6, addresses are 128 bits in length, four times larger than IPv4 addresses. The
conventional notation adopted for IPv6 addresses is a series of four hex numbers
called *blocks* or *fields* separated by colons.For example,
`5f05:2000:80ad:5800:0058:0800:2023:1d71`. A number of agreed-upon simplifications
have been standardized for expressing IPv6 addresses (RFC4291):

1. Leading zeros of a block need not be written. `5f05:2000:80ad:5800:0058:800:2023:1d71`
2. Blocks of all zeros can be omitted and replaced by the notation `::`. To avoid
ambiguities, the `::` notation may be used only once in an IPv6 address.
3. Embedded IPv4 addresses represented in the IPv6 format can use a form of hybrid
notation in which the block immediately preceding the IPv4 portion of the address
has the value `ffff` and the remaining part of the address is formatted using dotted-quad.
For example, the IPv6 address `::ffff:10.0.0.1` represents the IPv4 address `10.0.0.1`. This
is called an *IPv4-mapped IPv6 address*.

In some circumstances the colon delimiter in an IPv6 address may be confused with another separator
such as the colon used between an IP address and a port number. The bracket characters `[]` are used
to surround the IPv6 address.

The flexibility provided by `RFC4291` resulted in unnecessary confusion due to the ability to
represent the same IPv6 in multiple ways. To remedy this situation, `RFC5952` imposes some
rules to narrow the range of options while remaining compatible with `RFC4291`:

1. Leading zeros must be suppressed.
2. The `::` construct must be used to its maximum possible effect but not for only 16-bit blocks.
If multiple blocks contain equal-length runs of zeros, the first is replaced with `::.`.
3. The hex digits a through `f` should be represented in lowercase.

## 2.3 Basic IP Address Structure

Because of large number of addresses, it is convenient to divide the address space into
chunks.

### 2.3.1 Classful Addressing

When the Internet's address structure was originally defined, every unicast IP address
had a *network* portion, and a *host* portion.

With the realization that different networks might have different numbers of hosts, and
that each host requires a unique IP address, a partitioning was devised wherein
different-size allocation units of IP address space could be given out to different sites.
The partitioning of the address space involved five *classes*.

![Classful Addressing of IPv4](https://s2.loli.net/2022/03/12/mOJCaUNfMWDxI6o.png)

### 2.3.2 Subnet Addressing

One of the earliest difficulties encountered with the Internet began to grow was the
inconvenience of having to allocate a new network number for any new network segment
that was to be attached to the Internet. To address the problem, it was natural to
consider a way that a site attached to the Internet could be allocated a network
number centrally that could then be subdivided locally by site administrators.

Implementing this idea would require the ability to alter the line between the
network portion of an IP address and the host portion, but only for local purposes
at a site. The approach adopted to support this capability is called
*subnet addressing*.

Below is an example of how a class B address might be "subnetted". The first 16 bits
of every address the site will use are fixed at some particular number because these
bits have been allocated by a central authority. The last 16 bits can be divided by
the site network administrator.

![An example of a subnetted class B address](https://s2.loli.net/2022/03/12/5Dfyp7HIemRLEvP.png)

### 2.3.3 Subnet Masks

The *subnet mask* is an assignment of bits used by a host or router to determine how the
network and subnetwork information is partitioned from the host information in a
corresponding IP addresses. They are typically configured into a host or router in the
same way as IP addresses or using some dynamic system such as the DHCP.
