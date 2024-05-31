#!/bin/bash
INIT_FILE=$1
NEW_VERSION=$2
sed -i -e "/__version__ =/ s/= .*/= \"${NEW_VERSION}\"/" "$INIT_FILE"
