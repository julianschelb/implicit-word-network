# Security Policy

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported |
|---|---|
| 0.1.x | yes |
| 0.0.x | no |

## Reporting a vulnerability

This library processes local text and loads machine-learning models from the
Hugging Face Hub / spaCy; it does not open network services. If you find a
security-relevant issue (for example unsafe deserialisation in `ImplicitNetwork.load`
or a dependency problem), please report it privately:

- Use GitHub's private vulnerability reporting on this repository
  (Security → Report a vulnerability), or
- e-mail julian.schelb@uni-konstanz.de.

Please do not open a public issue for security reports. You can expect an
acknowledgement within a week and a fix or mitigation plan within 30 days.

## Notes for users

- `ImplicitNetwork.load` reads `.npz` archives with `allow_pickle=False`.
- Model checkpoints are downloaded by `gliner` / `sentence-transformers` /
  spaCy; pin model revisions in production and review their licenses.
