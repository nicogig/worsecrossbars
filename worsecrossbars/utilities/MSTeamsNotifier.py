import pymsteams

class MSTeamsNotifier:

    def __init__ (self, webhook):
        self.webhook_url = webhook

    def __call__ (self):
        pass

    def send_message(self, message, title=None, color=None):
        msteams_message = pymsteams.connectorcard(self.webhook_url)
        msteams_message.text(message)
        if title:
            msteams_message.title(title)
        if color:
            msteams_message.color(color)
        msteams_message.send() 