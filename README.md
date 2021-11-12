# worsecrossbars

<p align="center" width="100%">
    <img width="100%" src="docs/imgs/logo.gif">
</p>

### A tool for simulating faulty devices in a memristive-based neural network.

## Installation

To get started, pull this repository in your environment. Then, install using

```
python3 -m pip install .
```

In the repository's root folder.

## Quick Start

Before you can perform computations using this framework, you'll need to set it up for internet access. This can be achieved using the command

```
python3 -m worsecrossbars.compute --setup True
```

In your favourite terminal emulator. For more information on what this command performs, or if you get lost during setup, consult our wiki.

After setup has completed, run a demo simulation using

```
python3 -m worsecrossbars.compute --config example.json
```

The output of the simulation can be found under ```~\worsecrossbars\outputs```.

## More Information

To get more detailed information on the files produced by worsecrossbars, its inputs, and its configuration stages, please consult the wiki.