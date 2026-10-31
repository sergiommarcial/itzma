# Security policy

## Supported versions

Fixes land on the latest release. itzma is before 1.0, so only the newest
minor version (`0.x`) gets security fixes; upgrade to it to receive them.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub instead: open the repository's
**Security** tab and choose **Report a vulnerability**, or go straight to
<https://github.com/sergiommarcial/itzma/security/advisories/new>.

Useful things to include:

- the version (crate or Python package) and the storage backend
- what an attacker needs: write access to the bucket, read access, a crafted
  object, a crafted query
- a reproduction, even a rough one

You should get a first reply within a week. Once a fix is ready it ships in a
patch release, and the advisory is published with credit to you unless you
would rather stay anonymous.

## Scope

itzma trusts whoever can write to the bucket: anyone with write access can
change or delete a namespace, and that is not a vulnerability on its own. In
scope are things like a crafted object that makes a reader panic, hang,
allocate without bound or return wrong results silently, a query that reads
past its byte budget, and anything that breaks the guarantees listed in the
README.
