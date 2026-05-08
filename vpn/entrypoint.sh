#!/bin/sh

SECRET=/run/secrets/openvpn_config

sed -n '1p' "$SECRET" | tr -d '\r' >  /tmp/vpn_creds
sed -n '2p' "$SECRET" | tr -d '\r' >> /tmp/vpn_creds
chmod 600 /tmp/vpn_creds
tail -n +3 "$SECRET" > /tmp/vpn.conf

# Ensure auth-user-pass always points to our credentials file,
# regardless of what the config says (handles bare 'auth-user-pass' with no path).
grep -v '^auth-user-pass' /tmp/vpn.conf > /tmp/vpn.conf.tmp
echo "auth-user-pass /tmp/vpn_creds" >> /tmp/vpn.conf.tmp
mv /tmp/vpn.conf.tmp /tmp/vpn.conf

# Forward :8000 -> 10.64.82.60:8000 across tun0 so external callers
# (e.g. host CLIs reaching us via the published 127.0.0.1:8000 port) can hit
# the VPN-side llama endpoint. phoebe-api / phoebe-sandbox already reach it
# directly through the shared netns and are unaffected.
# Self-restarts if socat exits; failures while tun0 is down are harmless.
(
    while true; do
        socat TCP-LISTEN:8000,fork,reuseaddr TCP:10.64.82.60:8000
        echo "socat exited (code $?), retrying in 5s..." >&2
        sleep 5
    done
) &

# Retry loop: keep the container alive if openvpn exits or fails to connect.
# phoebe-api and phoebe-sandbox share this container's network namespace, so it must
# stay running at all times.
while true; do
    openvpn --config /tmp/vpn.conf
    echo "OpenVPN exited (code $?), retrying in 5s..." >&2
    sleep 5
done
