"""
notifier.py - Node-Side Predictive Maintenance Notification Engine.

Receives actual ML predictive-maintenance alerts from the Central Server,
formats clear operational warnings, and triggers lightweight, non-blocking
native Windows desktop notifications (Toast / Balloon) on the affected node.
Includes deduplication so repeat polling cycles do not spam duplicate popups.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def show_windows_desktop_notification(title: str, message: str) -> None:
    """
    Triggers a lightweight, non-blocking native Windows desktop notification.
    Uses Windows.UI.Notifications Action Center Toast with fallback to
    System.Windows.Forms.NotifyIcon balloon tip.
    Runs asynchronously in a background thread so the telemetry collector
    loop is never blocked.
    """
    if platform.system() != "Windows":
        return

    def _worker():
        clean_title = title.replace('"', '`"').replace("'", "''")
        clean_msg = message.replace('"', '`"').replace("'", "''").replace("\r", " ").replace("\n", " ")

        ps_script = f"""
        try {{
            $xml = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>{clean_title}</text>
            <text>{clean_msg}</text>
        </binding>
    </visual>
    <audio src="ms-winsoundevent:Notification.Warning" />
</toast>
"@
            $xmlDoc = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xmlDoc.LoadXml($xml)
            $toast = New-Object Windows.UI.Notifications.ToastNotification $xmlDoc
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AdaptiveCloud.NodeAgent")
            $notifier.Show($toast)
        }} catch {{
            [reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null
            [reflection.assembly]::loadwithpartialname('System.Drawing') | Out-Null
            $notify = New-Object System.Windows.Forms.NotifyIcon
            $notify.Icon = [System.Drawing.SystemIcons]::Warning
            $notify.Visible = $true
            $notify.BalloonTipTitle = "{clean_title}"
            $notify.BalloonTipText = "{clean_msg}"
            $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
            $notify.ShowBalloonTip(4500)
            Start-Sleep -Seconds 2
            $notify.Dispose()
        }}
        """
        # CREATE_NO_WINDOW flag = 0x08000000
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                creationflags=creation_flags,
                timeout=10,
                check=False
            )
        except Exception:
            pass

    # Launch daemon worker thread
    t = threading.Thread(target=_worker, daemon=True, name="node-alert-desktop-notify")
    t.start()


def _safe_print(text: str) -> None:
    """Safely print text even if Windows console is set to non-UTF8 code page (cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        clean = text.replace("⚠", "[!]").replace("🚨", "[!]")
        try:
            print(clean)
        except Exception:
            print(clean.encode("ascii", errors="replace").decode("ascii"))


class NodeAlertNotifier:
    """
    Manages node-side notification display and deduplication.
    Ensures that active alerts are clearly displayed on the affected Windows node,
    while suppressing repetitive popups across routine polling ticks.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.last_notified_alert_id: Optional[str] = None
        self.last_notified_risk: Optional[str] = None
        self.active_alert_ongoing: bool = False

    def process_server_alert(self, alert_payload: Optional[Dict[str, Any]]) -> bool:
        """
        Processes alert returned by the Central Server for this node.
        
        Args:
            alert_payload: The alert dictionary if risk is HIGH/CRITICAL, or None if healthy.
            
        Returns:
            True if a new desktop/console alert notification was triggered, False otherwise.
        """
        if not alert_payload or not isinstance(alert_payload, dict):
            if self.active_alert_ongoing:
                _safe_print(f"\n  [ALERT CLEARED] Node '{self.node_id}' telemetry nominal. Active maintenance alert resolved.")
                self.active_alert_ongoing = False
                self.last_notified_alert_id = None
                self.last_notified_risk = None
            return False

        alert_id = alert_payload.get("alert_id", "UNKNOWN_ALERT")
        risk_level = alert_payload.get("risk_level", "HIGH").upper()
        prob = float(alert_payload.get("failure_probability") or 0.0)
        prob_pct = f"{prob * 100:.0f}%"
        node = alert_payload.get("node_id", self.node_id)
        reason = alert_payload.get("reason", "Elevated resource pressure and machine strain.")
        action = alert_payload.get("recommended_action", "Inspect the node and reduce workload.")
        reps = alert_payload.get("repeat_count", 1)

        # ── Deduplication Check ──────────────────────────────────────────────
        if alert_id == self.last_notified_alert_id and not alert_payload.get("is_new", False):
            # Same incident cycle already notified
            _safe_print(f"  [ALERT ACTIVE] {alert_id} ({risk_level} - {prob_pct}) ongoing (tick repeat #{reps}) - popup suppressed.")
            self.active_alert_ongoing = True
            return False

        # ── New Alert Notification ───────────────────────────────────────────
        self.last_notified_alert_id = alert_id
        self.last_notified_risk = risk_level
        self.active_alert_ongoing = True

        # 1. Prominent Console Notification Box
        banner_lines = [
            "",
            "*" * 68,
            "⚠ PREDICTIVE MAINTENANCE ALERT",
            "",
            f"Node: {node}",
            f"Risk: {risk_level}",
            f"Failure Probability: {prob_pct}",
            "",
            "The system predicts an elevated probability of failure.",
            "",
            "Reason:",
            f"{reason}",
            "",
            "Recommended action:",
            f"{action}",
            "*" * 68,
            ""
        ]
        banner = "\n".join(banner_lines)
        _safe_print(banner)

        # 2. Windows Native Desktop Notification
        toast_title = f"[!] PREDICTIVE MAINTENANCE ALERT - {risk_level}"
        toast_msg = (
            f"Node: {node} | Risk: {risk_level} | Failure Probability: {prob_pct}\n"
            f"The system predicts an elevated probability of failure.\n"
            f"Recommended action: {action}"
        )
        show_windows_desktop_notification(toast_title, toast_msg)

        return True
