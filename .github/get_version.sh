#!/bin/bash
cat pyproject.toml| grep "version =" | cut -d'"' -f2
