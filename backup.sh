#!/usr/bin/env bash

# Directory for backups
BACKUP_DIR=~/backups/interviews_db
# Source database
DB_PATH=~/src/slop/data/interviews.db
# Date format for backup files
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create backup using sqlite3's .backup command
sqlite3 $DB_PATH ".backup '$BACKUP_DIR/interviews_$DATE.db'"
