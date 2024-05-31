#!/bin/bash
cat network_diffusion/__init__.py | grep __version__ | cut -d'"' -f2
