# Cryptography

## Introduction to Cryptography

There are three main goals of cryptography:

+ Confidentiality: preventing adversaries from reading our private data.
+ Integrity: preventing attackers from altering some data.
+ Authenticity: ensuring that the expected user created some data.

There are two ways of modern cryptography:

+ Symmetric-key cryptography: the same secret key is used by both
endpoints of a communication.
+ Public-key (asymmetric-key) cryptography: sender and receiver use
different keys.

## Kerckhoff's Principle

+ Crypto systems should remain secure even when the attacker knows
all internal details of the algorithm/system.
+ The key should be the only thing that must be kept secret, and
the system should be designed to make it easy to change keys that
are leaked (or suspected to be leaked).

There are three reasons:

+ If your secrets are leaked, it is usually a lot easier to
change the key than to replace every instance of the running software
+ You want to people to analyze and improve the algorithm.
+ Multiple different people (with different trust models) should
be able to use the algorithm.

## Symmetric Encryption Scheme

+ `keygen() -> K`
+ `Enc(K, M) -> C`
+ `Dec(K, C) -> M`
+ Correctness: $\forall K, \forall M, (Enc(K,M) \to C) \to (Dec(K,C) = M)$.
+ Security: No partial information about $M$ may leak! Because
an adversary can couple it with side information to reconstruct $M$.

### IND-CPA

To formally define no partial information leak, we define the game
IND-CPA.

![IND-CPA game](https://s2.loli.net/2022/06/17/qIXNPAMlg53Q1zO.png)

### Block Ciphers

![Block Cipher definition](https://s2.loli.net/2022/06/17/lO3cUBIAz2mfuQF.png)

## Asymmetric Encryption

For asymmetric encryption, Alice would produce a public key and a private
key. Also, Bob would produce a public key and a private key.
When Alice wants to send the message to Bob, Alice uses Bob's public
key to encrypt the message, and Bob uses its private key to decrypt the message.

So for encrypt function, it should be a one-way function.

### Discrete Logarithm

Discrete Logarithm is a famous one-way function.

$$
\begin{align*}
f(x) = g^{x} \mod p, &p \text{ is a large prime (2048 bits long)} \\
&g \text{ is a random value in } [2,p - 1]
\end{align*}
$$

### Diffie-Hellman Key Exchange and Security

![Diffie-Hellman Key Exchange and Security](https://s2.loli.net/2022/06/17/WGT2rQIeYA6y9XU.png)

## Integrity and Authentication

Sometimes, we do Not care about confidentiality but integrity and authentication.

### Message Authentication Codes (MACs)

For symmetric, we use ASE-EMAC to provide MAC, for asymmetric,
we use digital signature to provide MAC.
