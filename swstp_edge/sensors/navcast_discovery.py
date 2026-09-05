"""
sensors/navcast_discovery.py — Automatic USB Tethering IP & Port Detection for NavCast.

Automatically detects the connected phone's IP address and NavCast TCP server port
across Android (RNDIS/USB tethering, Wi-Fi hotspot) and iOS, on both Linux
(Raspberry Pi) and Windows environments.

Eliminates the need to manually look up or configure the phone's IP and port.
"""

import concurrent.futures
import os
import re
import socket
import struct
import subprocess
import sys
import time

# Standard candidate ports used by GPS / NMEA streaming apps
DEFAULT_CANDIDATE_PORTS = [10110, 2947, 11123, 50000, 8080]

# Known default Android/iOS tethering gateways
COMMON_TETHER_GATEWAYS = [
    "192.168.42.129",   # Android USB tethering default (Samsung, Pixel, OnePlus, etc.)
    "192.168.42.1",     # Android USB tethering alternate
    "192.168.43.1",     # Android Wi-Fi hotspot default
    "192.168.44.1",     # Android USB tethering subnet variant
    "172.20.10.1",      # iOS USB / Wi-Fi tethering gateway
]

RE_IPV4 = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")


# ---------------------------------------------------------------------------
# Linux Network Route & Neighbor Parsers (/proc/net)
# ---------------------------------------------------------------------------

def _hex_to_ipv4(hex_str: str) -> str | None:
    """Convert little-endian hex string from /proc/net/route to dotted IPv4."""
    try:
        val = int(hex_str, 16)
        return socket.inet_ntoa(struct.pack("<L", val))
    except Exception:
        return None


def _get_linux_proc_routes() -> list[tuple[str, str, str]]:
    """
    Parse /proc/net/route.
    Returns list of (iface, destination_ip, gateway_ip).
    """
    routes = []
    if not os.path.exists("/proc/net/route"):
        return routes
    try:
        with open("/proc/net/route", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] != "Iface":
                    iface = parts[0]
                    dest = _hex_to_ipv4(parts[1])
                    gw = _hex_to_ipv4(parts[2])
                    if dest and gw:
                        routes.append((iface, dest, gw))
    except Exception:
        pass
    return routes


def _get_linux_proc_arp() -> list[tuple[str, str]]:
    """
    Parse /proc/net/arp.
    Returns list of (ip, iface).
    """
    entries = []
    if not os.path.exists("/proc/net/arp"):
        return entries
    try:
        with open("/proc/net/arp", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0] != "IP":
                    ip = parts[0]
                    flags = parts[2]
                    iface = parts[5]
                    # flags 0x2 = complete entry (device answered ARP)
                    if flags != "0x0" and RE_IPV4.match(ip):
                        entries.append((ip, iface))
    except Exception:
        pass
    return entries


def _get_linux_ip_command_routes() -> list[str]:
    """Fallback: query `ip route` and `ip neighbor` via shell."""
    found_ips = []
    for cmd in (["ip", "route"], ["ip", "neighbor"]):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0)
            text = out.decode("ascii", errors="ignore")
            for match in RE_IPV4.findall(text):
                if not match.endswith(".0") and not match.endswith(".255"):
                    found_ips.append(match)
        except Exception:
            pass
    return found_ips


# ---------------------------------------------------------------------------
# Windows Network Route & ARP Parsers
# ---------------------------------------------------------------------------

def _get_windows_gateways() -> list[str]:
    """Find default gateways and ARP entries on Windows."""
    found_ips = []
    # 1. ARP table
    try:
        out = subprocess.check_output(["arp", "-a"], stderr=subprocess.DEVNULL, timeout=1.0)
        text = out.decode("ascii", errors="ignore")
        for match in RE_IPV4.findall(text):
            if not match.endswith(".255") and not match.startswith("224.") and not match.startswith("239."):
                found_ips.append(match)
    except Exception:
        pass

    # 2. Route print 0.0.0.0
    try:
        out = subprocess.check_output(["route", "print", "0.0.0.0"], stderr=subprocess.DEVNULL, timeout=1.0)
        text = out.decode("ascii", errors="ignore")
        for match in RE_IPV4.findall(text):
            if match != "0.0.0.0" and not match.endswith(".255"):
                found_ips.append(match)
    except Exception:
        pass

    return found_ips


# ---------------------------------------------------------------------------
# Local Subnet IP Extraction
# ---------------------------------------------------------------------------

def _get_local_ips() -> list[str]:
    """Get all local IPv4 addresses assigned to interfaces."""
    ips = set()
    try:
        host_info = socket.gethostbyname_ex(socket.gethostname())
        for ip in host_info[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    # Connect dummy UDP socket to find primary interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    return list(ips)


# ---------------------------------------------------------------------------
# Candidate IP Collection
# ---------------------------------------------------------------------------

def get_candidate_ips(fallback_host: str | None = None) -> list[str]:
    """
    Assemble an ordered list of candidate phone IP addresses.
    Prioritizes USB tethering interfaces, default gateways, ARP neighbors,
    and common tether subnets.
    """
    usb_gateways = []
    other_gateways = []
    arp_ips = []

    # 1. Linux routes
    routes = _get_linux_proc_routes()
    for iface, dest, gw in routes:
        if gw != "0.0.0.0":
            if any(k in iface.lower() for k in ("usb", "rndis", "enx", "eth1")):
                usb_gateways.append(gw)
            else:
                other_gateways.append(gw)

    # 2. Linux ARP
    for ip, iface in _get_linux_proc_arp():
        if any(k in iface.lower() for k in ("usb", "rndis", "enx", "eth1")):
            usb_gateways.append(ip)
        else:
            arp_ips.append(ip)

    # 3. Linux ip command fallback
    if not usb_gateways and sys.platform.startswith("linux"):
        for ip in _get_linux_ip_command_routes():
            if ip.startswith("192.168.42.") or ip.startswith("10."):
                usb_gateways.append(ip)
            else:
                arp_ips.append(ip)

    # 4. Windows routes & ARP
    if sys.platform.startswith("win"):
        for ip in _get_windows_gateways():
            if ip.startswith("192.168.42.") or ip.startswith("10."):
                usb_gateways.append(ip)
            else:
                other_gateways.append(ip)

    # 5. Derive probable gateways from local interface IPs (.1, .129)
    derived_gateways = []
    for local_ip in _get_local_ips():
        parts = local_ip.split(".")
        if len(parts) == 4:
            prefix = ".".join(parts[:3])
            derived_gateways.append(f"{prefix}.129")   # Android RNDIS default
            derived_gateways.append(f"{prefix}.1")     # Standard gateway
            derived_gateways.append(f"{prefix}.254")

    # Combine all in strict priority order (no duplicates)
    ordered = []
    seen = set()

    def _add(ip: str | None):
        if ip and ip not in seen and ip != "0.0.0.0" and not ip.startswith("127."):
            seen.add(ip)
            ordered.append(ip)

    # Highest priority: direct USB interface gateways & neighbors
    for ip in usb_gateways:
        _add(ip)

    # User-configured fallback host (e.g. from config.py)
    _add(fallback_host)

    # Derived gateways from current network subnets
    for ip in derived_gateways:
        _add(ip)

    # Common tethering defaults (Android 192.168.42.129, 192.168.42.1, etc.)
    for ip in COMMON_TETHER_GATEWAYS:
        _add(ip)

    # General gateways and ARP neighbors
    for ip in other_gateways:
        _add(ip)
    for ip in arp_ips:
        _add(ip)

    return ordered


# ---------------------------------------------------------------------------
# Fast Probe & NMEA Stream Verification
# ---------------------------------------------------------------------------

def probe_navcast_server(host: str, port: int, timeout: float = 0.3) -> tuple[bool, bool]:
    """
    Probe a target (host, port) to check:
    1. Is TCP port open?
    2. Is it sending NMEA sentences ($GGA, $RMC, $GSA, etc.)?

    Returns (port_open: bool, is_nmea: bool)
    """
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(0.5)
        # Try to read a sample to verify NMEA stream
        try:
            chunk = sock.recv(256).decode("ascii", errors="replace")
            is_nmea = any(sig in chunk for sig in ("$G", "$P", "GGA", "RMC", "GSA", "GSV", "VTG"))
            return (True, is_nmea)
        except Exception:
            # Port is open even if no data arrived in 0.5s
            return (True, False)
    except Exception:
        return (False, False)
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# High-Level Discovery Function
# ---------------------------------------------------------------------------

def discover_navcast(
    preferred_host: str | None = None,
    preferred_port: int = 10110,
    candidate_ports: list[int] | None = None,
    max_scan_ips: int = 12,
    timeout: float = 0.3,
) -> tuple[str, int] | tuple[None, None]:
    """
    Automatically discovers the NavCast TCP server host IP and port.

    Scans candidate IPs and ports concurrently for fast response (< 0.8s).
    Returns (host, port) if found, or (None, None) if unavailable.
    """
    ports = []
    if preferred_port:
        ports.append(preferred_port)
    for p in (candidate_ports or DEFAULT_CANDIDATE_PORTS):
        if p not in ports:
            ports.append(p)

    candidates = get_candidate_ips(fallback_host=preferred_host)[:max_scan_ips]
    if not candidates:
        return (None, None)

    # Build targets list: (host, port)
    targets = [(ip, p) for ip in candidates for p in ports]

    best_target = None

    # Probe concurrently with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        future_map = {
            pool.submit(probe_navcast_server, ip, p, timeout): (ip, p)
            for ip, p in targets
        }
        for future in concurrent.futures.as_completed(future_map):
            ip, p = future_map[future]
            try:
                is_open, is_nmea = future.result()
                if is_open:
                    if is_nmea:
                        # Definite NMEA stream verified! Return immediately
                        return (ip, p)
                    elif best_target is None:
                        # Open port found, keep as candidate unless confirmed NMEA arrives
                        best_target = (ip, p)
            except Exception:
                pass

    return best_target if best_target else (None, None)


def auto_detect_navcast(
    preferred_host: str | None = None,
    preferred_port: int = 10110,
    candidate_ports: list[int] | None = None,
    log_prefix: str = "[GNSS]",
) -> tuple[str, int]:
    """
    Convenience wrapper: returns discovered (host, port) or falls back to
    (preferred_host, preferred_port).
    """
    discovered_host, discovered_port = discover_navcast(
        preferred_host=preferred_host,
        preferred_port=preferred_port,
        candidate_ports=candidate_ports,
    )
    if discovered_host and discovered_port:
        print(f"{log_prefix} Auto-detected NavCast server @ {discovered_host}:{discovered_port}")
        return (discovered_host, discovered_port)

    # Fallback to configured default
    host = preferred_host or "10.208.43.190"
    port = preferred_port or 10110
    print(f"{log_prefix} Auto-detect found no active server; using configured default {host}:{port}")
    return (host, port)


# ---------------------------------------------------------------------------
# CLI test utility
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NavCast Auto-Discovery Scanner")
    print("==============================")
    print("[*] Detecting candidate tethering IPs...")
    ips = get_candidate_ips(fallback_host="10.208.43.190")
    for i, ip in enumerate(ips, 1):
        print(f"  {i}. {ip}")

    print("\n[*] Probing for active NavCast servers...")
    host, port = discover_navcast(preferred_host="10.208.43.190", preferred_port=10110)
    if host and port:
        print(f"\n[+] SUCCESS: Found NavCast at {host}:{port}")
    else:
        print("\n[-] No active NavCast server found on candidate IPs.")
