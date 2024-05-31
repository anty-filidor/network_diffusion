#!/bin/bash
NEW_VERSION=$1
sed -i -e "/version =/ s/= .*/= \"${NEW_VERSION}\"/" pyproject.toml
