import os
import socket
import ssl
from flask import Flask
from flask_socketio import SocketIO
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta


# Function to generate a self-signed certificate
def generate_self_signed_cert(cert_file, key_file):
    """
    Generate a self-signed SSL certificate and its corresponding private key.

    Parameters:
    - cert_file (str): The file path where the generated certificate will be saved.
    - key_file (str): The file path where the generated private key will be saved.

    Returns:
    - None

    Actions:
    1. Generates a private key using RSA algorithm with a key size of 2048 bits.
    2. Creates a certificate subject and issuer with common name set to an empty string.
    3. Constructs a self-signed X.509 certificate valid for one year from the current date.
    4. Adds a Subject Alternative Name extension with 'localhost' as the DNS name.
    5. Signs the certificate using the generated private key and SHA256 hashing algorithm.
    6. Writes the private key to the specified key_file in PEM format without encryption.
    7. Writes the certificate to the specified cert_file in PEM format.
    """
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "XX"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "X"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "X"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "X"),
            x509.NameAttribute(NameOID.COMMON_NAME, "X"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(
            # Certificate is valid for 1 year
            datetime.utcnow()
            + timedelta(days=365)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # Write the private key to a file
    with open(key_file, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write the certificate to a file
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
