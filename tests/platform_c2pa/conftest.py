"""Ephemeral ES256 test PKI generated at test time with the cryptography package (no git-ignored key needed).

Profile copied from the upstream c2pa-python test chain: root CA -> intermediate CA (CA:TRUE, keyCertSign) -> end-entity
(CA:FALSE, digitalSignature + contentCommitment, EKU emailProtection). The signer chain is end-entity + intermediate; the root
is the trust anchor. The keys exist only in the test process and are never written to the repository."""
import datetime as dt

import pytest

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
except Exception:           # pragma: no cover
    x509 = None


def _name(cn, org):
    return x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, "US"), x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
                      x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "FOR TESTING_ONLY"), x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _cert(subject, issuer, pub, signing_key, ca, ku, eku=None, issuer_pub=None):
    now = dt.datetime.now(dt.timezone.utc)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(pub).serial_number(x509.random_serial_number())
         .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), True).add_extension(ku, True)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(pub), False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_pub or pub), False))
    if eku:
        b = b.add_extension(x509.ExtendedKeyUsage([eku]), True)
    return b.sign(signing_key, hashes.SHA256())


def _ku(**kw):
    base = dict(digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False,
                key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False)
    base.update(kw)
    return x509.KeyUsage(**base)


@pytest.fixture(scope="session")
def pki():
    if x509 is None:
        pytest.skip("cryptography package not importable")
    rk, ik, lk = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = _cert(_name("Ephemeral Root CA", "DFD Test Root"), _name("Ephemeral Root CA", "DFD Test Root"), rk.public_key(), rk, True,
                 _ku(key_cert_sign=True, crl_sign=True, digital_signature=True))
    inter = _cert(_name("Ephemeral Intermediate CA", "DFD Test Intermediate"), root.subject, ik.public_key(), rk, True,
                  _ku(key_cert_sign=True, crl_sign=True, digital_signature=True), issuer_pub=rk.public_key())
    leaf = _cert(_name("DFD Ephemeral Signer", "DFD Test Signer"), inter.subject, lk.public_key(), ik, False,
                 _ku(digital_signature=True, content_commitment=True), ExtendedKeyUsageOID.EMAIL_PROTECTION, issuer_pub=ik.public_key())
    pem = lambda c: c.public_bytes(serialization.Encoding.PEM)
    key = lk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return {"chain": pem(leaf) + pem(inter), "key": key, "root_pem": pem(root).decode("ascii")}
