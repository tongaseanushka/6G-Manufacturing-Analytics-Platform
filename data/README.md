# Data Directory

Place the `Thales_Group_Manufacturing.csv` dataset in this directory.

## Dataset Requirements

- **Filename**: `Thales_Group_Manufacturing.csv`
- **Size**: 100,000 rows × 14 columns
- **Date Format**: DD-MM-YYYY (e.g., "15-03-2025")
- **Time Range**: 2025-01-01 to 2025-03-10 (approximately 69 days)
- **Machines**: 50 unique machine IDs

## Required Columns

1. Timestamp (Date, DD-MM-YYYY format)
2. Machine_ID (String)
3. Network_Latency_ms (Float, non-negative)
4. Packet_Loss_% (Float, 0-100)
5. Efficiency_Status (Categorical: Low/Medium/High)
6. Production_Speed (Float)
7. Error_Rate (Float)
8. Quality_Control_Defect_Rate (Float)
9. Operation_Mode (Categorical: Active/Idle/Maintenance)
10. Additional telemetry columns (14 total)

## Note

CSV files in this directory are excluded from version control per `.gitignore` configuration.