"""
Notification Service
Handles email and web push notifications for task assignments.
Works with Gmail SMTP or any SMTP server.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict
from pathlib import Path

# Try to import pywebpush for push notifications
try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    print("⚠️ pywebpush not installed. Push notifications disabled.")

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


class NotificationService:
    """
    Service for sending notifications via email and web push.
    """
    
    def __init__(self):
        # Email configuration from environment
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        
        # Web push configuration
        self.vapid_public_key = os.getenv("VAPID_PUBLIC_KEY", "")
        self.vapid_private_key = os.getenv("VAPID_PRIVATE_KEY", "")
        self.vapid_email = os.getenv("VAPID_EMAIL", "mailto:admin@example.com")
        
        # Check if email is configured
        self.email_configured = bool(self.smtp_username and self.smtp_password)
        self.push_configured = bool(self.vapid_private_key and PUSH_AVAILABLE)
        
        if not self.email_configured:
            print("⚠️ Email not configured. Set SMTP_USERNAME and SMTP_PASSWORD in .env")
        if not self.push_configured:
            print("⚠️ Web push not configured. Set VAPID keys in .env")
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: str = None
    ) -> Dict:
        """
        Send an email notification.
        
        Returns:
            Dict with 'success' boolean and 'message' string
        """
        if not self.email_configured:
            return {
                "success": False,
                "message": "Email not configured. Set SMTP credentials in .env"
            }
        
        if not to_email:
            return {
                "success": False,
                "message": "No recipient email address provided"
            }
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email
            
            # Add plain text version
            if body_text:
                msg.attach(MIMEText(body_text, "plain"))
            
            # Add HTML version
            msg.attach(MIMEText(body_html, "html"))
            
            # Send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())
            
            return {
                "success": True,
                "message": f"Email sent to {to_email}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }
    
    def send_push(
        self,
        subscription: Dict,
        title: str,
        body: str,
        data: Dict = None
    ) -> Dict:
        """
        Send a web push notification.
        
        Args:
            subscription: Push subscription object from browser
            title: Notification title
            body: Notification body text
            data: Additional data payload
            
        Returns:
            Dict with 'success' boolean and 'message' string
        """
        if not self.push_configured:
            return {
                "success": False,
                "message": "Push notifications not configured"
            }
        
        if not subscription:
            return {
                "success": False,
                "message": "No push subscription provided"
            }
        
        try:
            import json
            
            payload = json.dumps({
                "title": title,
                "body": body,
                "data": data or {},
                "icon": "/icon-192.png",
                "badge": "/badge-72.png"
            })
            
            webpush(
                subscription_info=subscription,
                data=payload,
                vapid_private_key=self.vapid_private_key,
                vapid_claims={
                    "sub": self.vapid_email
                }
            )
            
            return {
                "success": True,
                "message": "Push notification sent"
            }
            
        except WebPushException as e:
            return {
                "success": False,
                "message": f"Push failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Push error: {str(e)}"
            }
    
    def send_task_notification(
        self,
        worker_name: str,
        worker_email: str,
        push_subscription: Dict,
        task_title: str,
        task_description: str,
        task_location: str,
        start_date: str,
        assigned_by: str
    ) -> Dict:
        """
        Send task assignment notification via all available channels.
        
        Returns:
            Dict with results for each notification type
        """
        results = {
            "email": {"success": False, "message": "Not attempted"},
            "push": {"success": False, "message": "Not attempted"}
        }
        
        # Build email content
        email_subject = f"🔔 New Task Assigned: {task_title}"
        
        email_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .content {{ padding: 30px; }}
                .task-card {{ background: #f8fafc; border-left: 4px solid #667eea; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .task-title {{ font-size: 20px; color: #1a1a2e; margin: 0 0 10px 0; }}
                .task-detail {{ color: #666; margin: 8px 0; }}
                .task-detail strong {{ color: #333; }}
                .cta-button {{ display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px 30px; text-decoration: none; border-radius: 30px; font-weight: 600; margin-top: 20px; }}
                .footer {{ background: #f8fafc; padding: 20px; text-align: center; color: #888; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📋 New Task Assigned</h1>
                </div>
                <div class="content">
                    <p>Hello <strong>{worker_name}</strong>,</p>
                    <p>You have been assigned a new task. Please review the details below:</p>
                    
                    <div class="task-card">
                        <h2 class="task-title">{task_title}</h2>
                        <p class="task-detail"><strong>📝 Description:</strong> {task_description or 'No description provided'}</p>
                        <p class="task-detail"><strong>📍 Location:</strong> {task_location or 'TBD'}</p>
                        <p class="task-detail"><strong>📅 Start Date:</strong> {start_date or 'TBD'}</p>
                        <p class="task-detail"><strong>👤 Assigned By:</strong> {assigned_by}</p>
                    </div>
                    
                    <p>Please log in to your Worker Dashboard to accept or review this task.</p>
                    
                    <center>
                        <a href="#" class="cta-button">View Task Details</a>
                    </center>
                </div>
                <div class="footer">
                    <p>This is an automated notification from the Worker Allocation System.</p>
                    <p>© 2026 Neural Worker AI</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        email_text = f"""
        New Task Assigned!
        
        Hello {worker_name},
        
        You have been assigned a new task:
        
        Task: {task_title}
        Description: {task_description or 'No description provided'}
        Location: {task_location or 'TBD'}
        Start Date: {start_date or 'TBD'}
        Assigned By: {assigned_by}
        
        Please log in to your Worker Dashboard to accept or review this task.
        
        - Neural Worker AI
        """
        
        # Send email
        if worker_email:
            results["email"] = self.send_email(
                to_email=worker_email,
                subject=email_subject,
                body_html=email_html,
                body_text=email_text
            )
        else:
            results["email"]["message"] = "Worker email not provided"
        
        # Send push notification
        if push_subscription:
            results["push"] = self.send_push(
                subscription=push_subscription,
                title=f"🔔 New Task: {task_title}",
                body=f"Location: {task_location or 'TBD'} | Start: {start_date or 'TBD'}",
                data={
                    "type": "new_task",
                    "task_title": task_title
                }
            )
        else:
            results["push"]["message"] = "Push subscription not available"
        
        return results


# Singleton instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get or create the notification service singleton."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


# For testing
if __name__ == "__main__":
    service = get_notification_service()
    print(f"Email configured: {service.email_configured}")
    print(f"Push configured: {service.push_configured}")
    
    # Test email (if configured)
    if service.email_configured:
        result = service.send_email(
            to_email="test@example.com",
            subject="Test Email",
            body_html="<h1>Hello!</h1><p>This is a test.</p>",
            body_text="Hello! This is a test."
        )
        print(f"Email test result: {result}")
