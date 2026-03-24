"""
alert_engine.py — Multi-channel alert system with per-zone cooldown.

Supports sound, email (smtplib), and SMS (Twilio).  All channels degrade
gracefully if credentials are missing.
"""

import os
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class AlertEngine:
    """
    Dispatches alerts across multiple channels with per-zone cooldown.

    Parameters
    ----------
    cooldown_seconds : float
        Minimum seconds between consecutive alerts *for the same zone*.
    alarm_sound_path : str | None
        Path to an audio file for sound alerts.
    email_enabled : bool
        Whether to attempt email alerts.
    sms_enabled : bool
        Whether to attempt SMS alerts.
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        alarm_sound_path: Optional[str] = None,
        email_enabled: bool = False,
        sms_enabled: bool = False,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.alarm_sound_path = alarm_sound_path
        self.email_enabled = email_enabled
        self.sms_enabled = sms_enabled

        # per-zone last-alert timestamp
        self._last_alert: Dict[str, float] = {}

        # email config from env
        self._smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self._smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self._smtp_user = os.getenv("SMTP_USER", "")
        self._smtp_pass = os.getenv("SMTP_PASS", "")
        self._email_to = os.getenv("ALERT_EMAIL_TO", "")

        # twilio config from env
        self._twilio_sid = os.getenv("TWILIO_SID", "")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self._twilio_from = os.getenv("TWILIO_FROM", "")
        self._sms_to = os.getenv("ALERT_SMS_TO", "")

    # ── public API ───────────────────────────────────────────────

    def trigger(
        self,
        zone_name: str,
        class_name: str,
        confidence: float,
        email_to_override: str = "",
        sms_to_override: str = "",
    ) -> bool:
        """
        Attempt to fire an alert for *zone_name*.

        Returns True if the alert actually fired (cooldown passed),
        False if suppressed by cooldown.
        """
        now = time.time()
        last = self._last_alert.get(zone_name, 0.0)

        if now - last < self.cooldown_seconds:
            return False  # still in cooldown

        self._last_alert[zone_name] = now

        message = (
            f"🚨 ALERT: {class_name} detected in '{zone_name}' "
            f"(confidence {confidence:.0%})"
        )

        # Sound (always attempted if path set)
        if self.alarm_sound_path:
            self._play_sound()

        # Email
        if self.email_enabled:
            recipient = email_to_override or self._email_to
            if recipient:
                self._send_email(recipient, message)

        # SMS
        if self.sms_enabled:
            recipient = sms_to_override or self._sms_to
            if recipient:
                self._send_sms(recipient, message)

        return True

    def reset_cooldown(self, zone_name: Optional[str] = None):
        """Clear cooldown for a specific zone or all zones."""
        if zone_name:
            self._last_alert.pop(zone_name, None)
        else:
            self._last_alert.clear()

    # ── private channels ─────────────────────────────────────────

    def _play_sound(self):
        """Play alarm sound on a background thread."""
        def _worker():
            try:
                from playsound import playsound
                playsound(self.alarm_sound_path)
            except Exception:
                pass  # degrade silently

        threading.Thread(target=_worker, daemon=True).start()

    def _send_email(self, to: str, body: str):
        """Send an email alert via SMTP on a background thread."""
        if not (self._smtp_user and self._smtp_pass):
            return

        def _worker():
            try:
                msg = MIMEMultipart()
                msg["From"] = self._smtp_user
                msg["To"] = to
                msg["Subject"] = "🚨 Surveillance Alert"
                msg.attach(MIMEText(body, "plain"))

                with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                    server.starttls()
                    server.login(self._smtp_user, self._smtp_pass)
                    server.send_message(msg)
            except Exception:
                pass  # degrade silently

        threading.Thread(target=_worker, daemon=True).start()

    def _send_sms(self, to: str, body: str):
        """Send an SMS alert via Twilio on a background thread."""
        if not (self._twilio_sid and self._twilio_token and self._twilio_from):
            return

        def _worker():
            try:
                from twilio.rest import Client
                client = Client(self._twilio_sid, self._twilio_token)
                client.messages.create(body=body, from_=self._twilio_from, to=to)
            except Exception:
                pass  # degrade silently

        threading.Thread(target=_worker, daemon=True).start()
