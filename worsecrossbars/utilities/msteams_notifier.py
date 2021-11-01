"""
msteams_notifier.py:
An internal module used to send notifications
to the user via Microsoft Teams.
"""
import pymsteams

class MSTeamsNotifier:
    """
    MSTeamsNotifier (class):
    An instance of this class holds the
    webhook necessary to send messages to the
    correct channel.
    """

    def __init__ (self, webhook):
        self.webhook_url = webhook

    def __call__ (self):
        pass

    def send_message(self, message, title=None, color=None):
        """
        send_message(message, title, color):
        Send a message to the channel indicated by the webhook URL.
        A title and color may be specified, but they are not necessary.
        """
        msteams_message = pymsteams.connectorcard(self.webhook_url)
        msteams_message.text(message)
        if title:
            msteams_message.title(title)
        if color:
            msteams_message.color(color)
        msteams_message.send()
