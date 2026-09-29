# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately using GitHub's private
vulnerability reporting for
[`vineethsudhir/indic-book-translator`](https://github.com/vineethsudhir/indic-book-translator/security/advisories/new).
Avoid opening a public issue for an unpatched vulnerability. Include the
affected version, impact, and steps to reproduce where possible.

## Supported versions

Security fixes are accepted for the current `0.1.x` release line.

## Security model

- The local app server binds to `127.0.0.1`, checks the `Host` header, and
  requires a per-launch token on every API call.
- API keys are stored in `secrets.env` in the per-user app-data folder. On
  macOS and Linux, the file is created with mode `0600`. The API reports
  whether a key is set, never the key value.
- Uploaded books and generated outputs are stored in
  `Documents/KannadaBookTranslator` by default. Private app settings and
  secrets are kept separately in the OS app-data folder.
- Book text is sent to the cloud providers selected by the user for the
  configured translation, editing, quality-check, or narration stages. Local
  providers process their stage on the user's computer.
