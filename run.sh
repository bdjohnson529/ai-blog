#!/bin/bash
mkdir -p .log
timestamp=$(date +%Y%m%d_%H%M)
bundle exec jekyll serve >> .log/output_${timestamp}.log 2>&1