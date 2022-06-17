# Web

## Same-Origin-Policy

The same-origin-policy means one origin should not be able to
access the resources of another origin. And javascript on one page
cannot read or modify pages from different origins.

+ The origin of a page is derived from the URL it was loaded from.
+ Special cases: Javascript runs with the origin of the page that loaded it.

### Origin

+ Granularity of protection of same origin policy
+ Origin = (protocol, hostname, port)
+ It's a **string match**! If these match, it is same origin, else it
is not. Even though in some cases, it is logically the same origin,
if there is no match, it is not.

## SQL Injection

Attacker user provides bad input and web server does not check
input format which enables attacker to execute arbitrary code
on the server.

![SQL Injection](https://s2.loli.net/2022/06/17/QpGIjgMKimkwPJH.png)

There are some ways to prevent SQL injection:

+ Sanitizate user input: check or enforce that value/string does not have
commands of any sort.
  + Disallow special characters
  + Escape input string
+ Avoid building a SQL command based on raw user input, use
existing tools or frameworks
+ Use parameterized/prepared SQL

## XSS

XSS is Cross-site scripting attack. Attacker injects a *malicious script*
into the webpage viewed by a victim user. Script runs in user's
browser with access to page's data.

### Stored XSS

For stored XSS, attacker leaves Javascript lying around on
web service for victim to load.

+ The attacker manages to store a malicious script at the web server.
+ The server later unwittingly sends script to a victim's browser.
+ Browser runs script in the same origin.

![Stored XSS](https://s2.loli.net/2022/06/17/QBwV4RLjzFCgtAW.png)

### Reflected XSS

For reflected XSS, attacker gets user to click on specially-crafted
URL with script in it, web service reflects it back.

+ The attacker gets the victim user to visit a URL that embeds a
malicious JavaScript
+ The server echoes it back to victim user in its response.
+ Victim's browser executes the script within the same origin.

![Reflected XSS](https://s2.loli.net/2022/06/17/bqkIKvV2BtEjHfs.png)

## CSRF

CSRF is cross site request forgery.

![CSRF](https://s2.loli.net/2022/06/17/Tv6YIzMPijcwxbd.png)

+ User logs in to bank.com, session cookie remains in browser state.
+ User visits malicious site containing:

  ```html
  <form name=F action=http://bank.com/BillPay.php>
  <input name=recipient value=badguy> ...
  <script> document.F.submit(); </script>
  ```
+ Browser sends user auth cookie with request

There are three basic ways to defend CSRF:

+ CSRF token: embed a new token in the web

  ```html
  <input type=hidden value=23a3af01b>
  ```

+ Referer Validation

  ```txt
  Referer: http://www.facebook.com/home.php
  ```

+ Custom HTTP Header (User LocalStorage to store the session, and
add the value to the header).
