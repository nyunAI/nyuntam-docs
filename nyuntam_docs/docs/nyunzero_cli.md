# Nyuntam CLI

Nyuntam also provides a command-line interface tool that provides a convenient way to initialize and manage your Nyuntam workspace, as well as run various scripts and algorithms supported by Nyuntam.

## Installation

To install Nyuntam, you need to have Python 3.6 or later installed on your system. You can then install the CLI tool using pip:

```shell
pip install nyuntam
```

## Usage

After installation, you can use the `nyuntam` command to access the available commands. Run `nyuntam --help` to see the list of available commands and their descriptions.

### Initializing the Workspace

Before you can run any scripts or algorithms, you need to initialize your Nyuntam workspace. You can do this using the `init` command:

```shell
nyuntam init [WORKSPACE_PATH] [CUSTOM_DATA_PATH] [OPTIONS]
```
- `WORKSPACE_PATH`: The path to the workspace directory. If not provided, the current working directory will be used.
- `CUSTOM_DATA_PATH`: The path to the custom data directory. If not provided, a default directory will be created within the workspace.
- `OPTIONS`:
  - `--overwrite`, `-o`: Overwrite the existing workspace spec if it already exists.
  - `--extensions`, `-e`: Specify the extensions to install. Defaults to installing all available extensions. Available extensions are:
    - `"all"`: Install all available extensions.
    - `"none"`: Don't install any extension.

Example:

```shell
# mkdir ~/my-workspace
nyun init ~/my-workspace ~/my-data
```

This command initializes a new workspace at `~/my-workspace` and sets the custom data directory to `~/my-data`, installing the Nyun Kompress Vision extension.

### Running Scripts

Once your workspace is initialized, you can run scripts using the `run` command in the workspace directory. The command syntax is as follows:

```shell
nyun run [SCRIPT_PATH]
```

- `SCRIPT_PATH`: The path(s) to the YAML or JSON script file you want to run.

Example:

```shell
nyuntam run ~/my-script.yaml
```

This command runs the script located at `~/my-script.yaml` within your initialized workspace.

To run chained scripts, you can provide multiple script paths in the order of execution:

```shell
nyuntam run ~/my-script1.yaml ~/my-script2.yaml
```

### Checking Version

To check the version of the Nyuntam CLI you have installed, use the `version` command:

```shell
nyuntam version
```

<!-- ## Contributing

If you'd like to contribute to the Nyun CLI project, please follow the standard GitHub workflow:

1. Fork the repository 
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive commit messages
4. Push your changes to your forked repository
5. Create a pull request against the main repository

We welcome all contributions, whether they are bug fixes, feature requests, or documentation improvements. -->

<!-- ## License

The Nyun CLI is released under the [MIT License](LICENSE). -->
