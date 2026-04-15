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

# Retry loop: keep the container alive if openvpn exits or fails to connect.
# mab-api and mab-sandbox share this container's network namespace, so it must
# stay running at all times.
while true; do
    openvpn --config /tmp/vpn.conf
    echo "OpenVPN exited (code $?), retrying in 5s..." >&2
    sleep 5
done
