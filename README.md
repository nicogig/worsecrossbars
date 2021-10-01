# worse-crossbars

A tool for simulating faulty devices in a memristive-based neural network

## Usage

```
usage: MLP.py [-h] [-hl HIDDEN_LAYERS] [-ft FAULT_TYPE] [-a ANNS] [-s SIMULATIONS] [-l LOG] [-d DROPBOX] [-t MSTEAMS]

optional arguments:
  -h, --help         show this help message and exit
  -hl HIDDEN_LAYERS  Number of hidden layers in the ANN
  -ft FAULT_TYPE     Identifier of the fault type. 1: Cannot electroform, 2: Stuck at HRS, 3: Stuck at LRS
  -a ANNS            Number of ANNs being simulated
  -s SIMULATIONS     Number of simulations being run
  -l LOG             Enable logging the output in a separate file
  -d DROPBOX         Enable Dropbox integration
  -t MSTEAMS         Enable MS Teams integration
```
The optional arguments ```-d``` and ```-t``` require additional configuration, as they connect to Third-Party providers. This only needs to be done once.

## Enabling Internet Integrations

### Enabling Dropbox Integration

The Dropbox integration facilitates the export of the outputted files. This can be beneficial in case computation is performed off-site and retreiving the data is cumbersome. To get started, set the optional argument ```-d True```. Please be aware that the initial setup is **NOT** unattended, and will require you to interact with the terminal, as well as opening a web browser.

Upon running the command, the package will look for the configuration file ```user_secrets.json```. If not found, you will be presented with the following prompt:

```
1. Go to: [AUTH_URL]
2. Click "Allow" (you might have to log in first).
3. Copy the authorization code.
Enter the authorization code here: 
```

Follow the URL to generate a token that the package will then use to upload the data on Dropbox. You will only need to do this once.

### Enabling MS Teams Integration

The MS Teams Integration allows you to receive notifications when jobs start and finish. This is great if you make the tasks run in the background. To get started, set the optional argument ```-t True```. Please be aware that the initial setup is **NOT** unattended, and will require you to interact with the terminal, as well as opening a web browser or your Microsoft Teams application.

To enable this integration, you will also need to setup a Webhook Connection in Microsoft Teams. For more information, visit [Microsoft's Website](https://docs.microsoft.com/en-us/outlook/actionable-messages/send-via-connectors#creating-messages-through-office-365-connectors-in-microsoft-teams). You will only need to configure the package once.