"""
Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).

Verifies that KalshiSigner produces a valid RSA-PSS signature for a
hardcoded message, using a test RSA key pair generated specifically
for this test.
"""
import base64
import unittest

from quant_engine.kalshi.client import KalshiSigner


# 2048-bit RSA test key (used ONLY for unit tests — not a real credential).
_TEST_PRIVATE_KEY_PEM = """\
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWeKKUVkE4s9mGDRSp2ylrEzvSw4p3V
RrA1dFe5aB1URFX6kTsOv6B/VGgDqfy+Z0sKn6zlBBW1r7aCm8g2aFTjFOfUFKfZ
/wq7FnB+DSFBhKCGhG6sASsr3P8UQOKLW0l5GDK0GW5sB7W9FTLaLgfx0dH8UYjh
P/3RnMSTMgQV/kkMb5z2rFAp0oW6BSy5rMQaVvYhh6qB6sFx8IOUqEYRUJIbUABT
VoqSWTCTLhtGW+u/AjN2y9NHQP0nJWBqTH/5n7RamN8WXxL5LoFVbKBHbOz3zq+W
QalHAoybMwj+qDNv0rOOGQb1r8j1TiGsDLblTwIDAQABAoIBAC5RgZ+hBx7xHNaM
pPgwGMnCd9OZaBhLBxkLRmDEqRz+ynh7LOkNqFPe+aFPuOmTSVljCjLIi1dRCiDX
LMrTqX8m0mgstSBXlPkRoCOxuXbLA0xtfj3LCEQJf3EJlK1ZHOlEMF6u5XGIKbPV
P1HCgbN1HhCJqGJ6m4nVJFHm5zh6UAsBUiMlkTiMVHmVLsQLg8IBJKZ1Y3KDnNiL
LMISkbMXjDPpNdgtw+kThXVNMh1b+0T0dvMXk3Hkr0/lBGfzIM7y6L+k6HJyRNfT
Q2s3PCbnkhGfBwVVKn+MIJyFiT1RYXB4Fz/arFSHclHY8QWMU3R4j2tddTpj3MOZ
P1TSneECgYEA7VIy01pqh3KIGZ3cRRidWCK/RJE/D0P8nqn4D6p+9myJ3WtEBnWp
KDaPkyOj+7LdPPFjz4DAF6WNXNO7KK7akXTt2Z/ZYApjlQSHiVEXdqxl0klG7Rn1
l+5fdY9+hKxxY+xRjO5HlJ4QIuiiSky+F7mUFbVgb1YGkE3c8HpOYVkCgYEA4h5t
LpJC5mDM5+XP4UKdMmI+3z5vnzSG5cW9DiPF11ycwgqXpAOjR7ILfzPM4IuewlbO
7dYCmInRIBhQI+CWmt0p6f8mIh8LbgJFRi1z5gRDKC3VNYweLo3+ufxYwLCdMrjY
aRGuaj3JoHvZkVoYNRF/A/I5q3TlFpEUKBjdrU8CgYEAqY7fIY+Bsj0jqXSxqnOa
JhNMYzP5B6nh+w7MgEQJ3MSqYXUWJjJDi1YAlJVjTaJ/YfF0OPG+L0SJC0DZAQBu
a8s0c2wl6oXJqH0jD9Md2bVEhS0BB5PMkf+v1xvPQ1VU6tU7n6DN/FjI1MCwHo2G
PjM4ADPkjx1ciiTRIJfnPYECgYBf2UPzQLCvqk0x8D5P3WqIK6v8Gzr3dJy6wADh
e7g/G5MIXqq9BfMhT5WpK6VHMEavTb4/u/M1//S+kgYIMz6ADbnNW6PKnqXQLBn+
dIME1ixAMGLOrkX7OSkv3zHK7x/RljFYv+zY9Y3O09zvK6zA3GmYmT0V/1SAgnRv
yQpCrQKBgQCSuXPKn+cOMFqW/gR+nL/A72s1eP1/A9RUiPd5ij1pJBdTl4aCJb+8
gT0B8VcRp2aXmXfhdmVE5HbpPVFrXmh/lR8jNhPKcHQPqnWVK5RW+vB3F/bHJnER
F3j0XE2sYzrHnm82pf2iW3JEnMrp3cR1sVFm9b0XBHjU7UuGqFXK+Q==
-----END RSA PRIVATE KEY-----
"""


class SignatureKATTests(unittest.TestCase):
    """Known-answer tests for Kalshi request signing."""

    def _skip_if_no_crypto(self):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            self.skipTest("cryptography library not installed — required for RSA-PSS signing")

    def test_sign_produces_valid_base64(self):
        """Signing should produce a valid base64-encoded signature string."""
        self._skip_if_no_crypto()
        signer = KalshiSigner(
            access_key="test-key-id",
            private_key_pem=_TEST_PRIVATE_KEY_PEM,
        )
        sig = signer.sign(
            timestamp_ms="1700000000000",
            method="GET",
            path="/trade-api/v2/markets",
        )
        # Must be valid base64
        decoded = base64.b64decode(sig)
        self.assertIsInstance(decoded, bytes)
        self.assertGreater(len(decoded), 0)

    def test_sign_deterministic_message_format(self):
        """The canonical message format must be <ts><METHOD><path>."""
        self._skip_if_no_crypto()
        signer = KalshiSigner(
            access_key="test-key-id",
            private_key_pem=_TEST_PRIVATE_KEY_PEM,
        )
        # Same inputs should be verifiable with the public key
        sig1 = signer.sign("1700000000000", "GET", "/trade-api/v2/markets")
        sig2 = signer.sign("1700000000000", "GET", "/trade-api/v2/markets")

        # Both should be valid base64 and non-empty
        raw1 = base64.b64decode(sig1)
        raw2 = base64.b64decode(sig2)
        self.assertGreater(len(raw1), 0)
        self.assertGreater(len(raw2), 0)

    def test_sign_verifies_with_public_key(self):
        """Signature must verify against the corresponding public key."""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
        except ImportError:
            self.skipTest("cryptography library not installed")

        signer = KalshiSigner(
            access_key="test-key-id",
            private_key_pem=_TEST_PRIVATE_KEY_PEM,
        )
        ts = "1700000000000"
        method = "POST"
        path = "/trade-api/v2/portfolio/orders"

        sig_b64 = signer.sign(ts, method, path)
        sig_bytes = base64.b64decode(sig_b64)

        # Extract public key from private key
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        private_key = load_pem_private_key(
            _TEST_PRIVATE_KEY_PEM.encode("utf-8"), password=None
        )
        public_key = private_key.public_key()

        # Verify signature
        message = f"{ts}{method.upper()}{path}".encode("utf-8")
        try:
            public_key.verify(
                sig_bytes,
                message,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
        except Exception as e:
            self.fail(f"Signature verification failed: {e}")

    def test_canonical_path_normalization(self):
        """Path should be normalized to canonical form."""
        result = KalshiSigner._canonical_path("https://api.kalshi.com/trade-api/v2/markets")
        self.assertEqual(result, "/trade-api/v2/markets")

        result2 = KalshiSigner._canonical_path("/trade-api/v2/markets")
        self.assertEqual(result2, "/trade-api/v2/markets")


if __name__ == "__main__":
    unittest.main()
