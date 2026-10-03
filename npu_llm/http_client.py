"""Credential destination policy shared by command-line API clients."""

import ipaddress
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener, urlopen


def validate_api_url(url, api_key):
    """Allow bearer tokens only over HTTPS or explicit loopback HTTP."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("API URL must use HTTP or HTTPS and include a host")
    if parts.username is not None or parts.password is not None:
        raise ValueError("API URL must not contain credentials")
    host = parts.hostname.lower()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if api_key and parts.scheme != "https" and not loopback:
        raise ValueError("Authenticated remote API requests require HTTPS")


class _NoCredentialRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Reject every redirect, including HTTPS downgrades and origin changes.
        return None


def open_api_request(request, api_key=None, *, timeout=30):
    """Validate the destination and prevent redirects for authenticated requests."""
    validate_api_url(request.full_url, api_key)
    if api_key:
        request.add_unredirected_header("Authorization", f"Bearer {api_key}")
        return build_opener(_NoCredentialRedirects()).open(request, timeout=timeout)
    return urlopen(request, timeout=timeout)
