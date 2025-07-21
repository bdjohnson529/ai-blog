#!/bin/bash
mkdir -p .log
timestamp=$(date +%Y%m%d_%H%M)
bundle exec jekyll serve 2>&1 >> tee .log/output_${timestamp}.log 