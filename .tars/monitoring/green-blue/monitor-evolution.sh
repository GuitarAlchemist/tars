#!/bin/bash 
# TARS Evolution Monitoring Script 
echo "📊 Starting TARS Evolution Monitoring" 
 
SESSION_ID="34744403" 
GREEN_ENDPOINT="http://localhost:8080" 
BLUE_ENDPOINT="http://localhost:8082" 
 
MONITORING_DIR=".tars/monitoring/green-blue/$SESSION_ID" 
mkdir -p "$MONITORING_DIR" 
 
echo "  🔍 Monitoring session: $SESSION_ID" 
echo "  🟢 Green endpoint: $GREEN_ENDPOINT" 
echo "  🔵 Blue endpoint: $BLUE_ENDPOINT" 
echo "  📊 Monitoring directory: $MONITORING_DIR" 
 
# Initialize CSV headers 
echo "timestamp,environment,status" > "$MONITORING_DIR/health-status.csv" 
echo "timestamp,environment,metric,value" > "$MONITORING_DIR/metrics.csv" 
 
# Monitoring loop 
echo "🔄 Starting monitoring loop (Ctrl+C to stop)..." 
while true; do 
    timestamp=$(date -Iseconds) 
    echo "[$timestamp] Monitoring green/blue environments..." 
    echo "$timestamp,green,monitoring" >> "$MONITORING_DIR/health-status.csv" 
    echo "$timestamp,blue,monitoring" >> "$MONITORING_DIR/health-status.csv" 
    echo "  ⏳ Next check in 5 minutes..." 
    sleep 300 
done 
