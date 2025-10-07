#!/bin/bash
# filepath: /work/ab1412/uftp_transfer_slurm.sh

#SBATCH --job-name=uftp_transfer
#SBATCH --account=ab1412
#SBATCH --partition=shared
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=uftp_transfer_%j.log
#SBATCH --error=uftp_transfer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# UFTP Configuration
UFTP_USER="scholle1"
UFTP_KEY="$HOME/.uftp/uftpkey"
UFTP_AUTH_URL="https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/auth/JUDAC:"
UFTP_SOURCE="/p/data1/slmet/met_data/ecmwf/era5/zarr/era5_y2010_2021_res025.zarr"
UFTP_DEST="/work/ab1412/atmorep/data/era5_y2010_2021_res025.zarr"

# Create destination directory if it doesn't exist
mkdir -p "$(dirname "$UFTP_DEST")

# Load any necessary modules
module load uftp-client || echo "No UFTP module found, using system installation"

# Setup logging
LOGDIR="/work/ab1412/uftp_logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${LOGDIR}/uftp_transfer_${TIMESTAMP}.log"

echo "=============================================" | tee -a "$LOGFILE"
echo "UFTP Transfer Job started at $(date)" | tee -a "$LOGFILE"
echo "Source: $UFTP_AUTH_URL$UFTP_SOURCE" | tee -a "$LOGFILE"
echo "Destination: $UFTP_DEST" | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"

# Check if destination already exists and has content
if [ -d "$UFTP_DEST" ] && [ "$(ls -A "$UFTP_DEST" 2>/dev/null)" ]; then
    echo "Destination directory already exists and has content." | tee -a "$LOGFILE"
    echo "Will attempt to resume transfer by skipping existing files." | tee -a "$LOGFILE"
    RESUME_FLAG="--skip-existing"
else
    echo "Starting new transfer." | tee -a "$LOGFILE"
    RESUME_FLAG=""
fi

# Start time for duration calculation
START_TIME=$(date +%s)

# Run the UFTP transfer with appropriate flags
# Note: Using -r is sufficient, -R is redundant when used with -r
echo "Starting UFTP transfer..." | tee -a "$LOGFILE"
uftp cp -R -v -u "$UFTP_USER" --identity "$UFTP_KEY" -P "$PASSPHRASE" -t4 $RESUME_FLAG \
    "${UFTP_AUTH_URL}${UFTP_SOURCE}" \
    "$UFTP_DEST" 2>&1 | tee -a "$LOGFILE"

TRANSFER_STATUS=${PIPESTATUS[0]}

# Calculate and log duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=============================================" | tee -a "$LOGFILE"
if [ $TRANSFER_STATUS -eq 0 ]; then
    echo "UFTP Transfer completed successfully." | tee -a "$LOGFILE"
else
    echo "UFTP Transfer failed with status $TRANSFER_STATUS." | tee -a "$LOGFILE"
fi
echo "Transfer duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"

exit $TRANSFER_STATUS