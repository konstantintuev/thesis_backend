#!/bin/bash

PARENT_DIR=$(pwd)

echo "PARENT_DIR: $PARENT_DIR"

SCRIPT_NAME="${PARENT_DIR}/pdf_2_md_chunkable_internal.py"

VENV_DIR="${PARENT_DIR}/venv_pdf_2_md_chunkable"

REQUIREMENTS_FILE="${PARENT_DIR}/requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        deactivate
        exit 1
    fi
else
    echo "$REQUIREMENTS_FILE not found. Skipping installation of dependencies."
fi

echo "Executing $SCRIPT_NAME with arguments: $1 $2"
python "$SCRIPT_NAME" "$1" "$2"

if [ $? -ne 0 ]; then
    echo "Error: Script execution failed."
    deactivate
    exit 1
fi

deactivate

echo "Script executed successfully."
