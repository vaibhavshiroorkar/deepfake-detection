# C2PA test fixtures

Source: https://github.com/contentauth/c2pa-python at commit ae8628afe957535011f4ac6675b3acac4206091a (path `tests/fixtures/`),
downloaded 2026-10-26. Licence: dual MIT OR Apache-2.0 (LICENSE-MIT and LICENSE-APACHE copied here from the same commit).

| File | Origin | Use | SHA-256 |
|---|---|---|---|
| A.jpg | c2pa-python tests/fixtures/A.jpg | unsigned JPEG, base for signing in tests | f999fd78bfe8a83c96e468a078830ba94485bc1bc6fd086fb94a43bd29dd0f23 |
| C.jpg | c2pa-python tests/fixtures/C.jpg | JPEG with an upstream-signed manifest (signer not on any trust list we configure) | 4cbac1af485155f878d12a610da851c2938f2ce886189c61d9da7a0a43bd00b5 |
| es256_certs.pem | c2pa-python tests/fixtures/es256_certs.pem | public test certificate chain ("C2PA Test Signing Cert"), also used as the trust anchor in one test | f18a293626ac261b18201e269ee0ed9c10bc30afb3b5c8708bb16e41a05bf040 |
| es256_private.key | c2pa-python tests/fixtures/es256_private.key | PUBLIC TEST private key from the upstream repo, for tests only; never use it for real signing | f547ec8dbf24e50b51cf9924e58c250d246ed8844304408a072801fcf70a95d7 |

Signed, tampered and corrupted variants are generated at test time with the library Builder and A.jpg; no media is uploaded anywhere.
