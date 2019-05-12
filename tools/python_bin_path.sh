if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No VIRTUAL_ENV set"
    export PYTHON_BIN_PATH="/usr/bin/python"
else
    echo "VIRTUAL_ENV is set"
    export PYTHON_BIN_PATH="${VIRTUAL_ENV}/bin/python"
fi
