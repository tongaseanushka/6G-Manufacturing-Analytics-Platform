"""
Data Preparation Module for Thales 6G Manufacturing Analytics Platform.

This module provides data loading, validation, and preprocessing functions for the
manufacturing telemetry dataset. It ensures data integrity through comprehensive
schema and value validation, and handles explicit date parsing to prevent ambiguity.

Key Features:
    - Explicit DD-MM-YYYY date parsing
    - Comprehensive schema validation (row count, column count, required columns)
    - Value range validation for numeric columns
    - Categorical value validation
    - Streamlit caching for performance optimization

Requirements Addressed:
    - 1.1: Load CSV dataset
    - 1.2: Parse Timestamp with DD-MM-YYYY format
    - 1.3: Validate 100,000 rows and 14 columns
    - 1.4: Validate required columns exist
    - 1.5: Raise descriptive errors for missing columns
    - 1.6: Validate Efficiency_Status categorical values
    - 1.7: Validate Operation_Mode categorical values
    - 1.8: Validate Network_Latency is numeric and non-negative
    - 1.9: Validate Packet_Loss is numeric and in range [0, 100]
    - 1.10: Complete loading within 5 seconds
    - 20.2: Include docstrings for all public functions
    - 20.3: Include detailed docstrings with formulas
    - 20.4: Include docstrings explaining date parsing
    - 20.6: Include type hints for all functions
"""

import pandas as pd
import os as _os

# Only import Streamlit (and its slow runtime initialization) when actually running
# inside a Streamlit server process. In pytest / CLI contexts, use a no-op decorator.
if _os.environ.get("STREAMLIT_SERVER_PORT") or _os.environ.get("STREAMLIT_RUN_TARGET"):
    import streamlit as st
    _cache_data = st.cache_data
else:
    import functools as _functools
    def _cache_data(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            @_functools.wraps(fn)
            def wrapper(*a, **kw):
                return fn(*a, **kw)
            return wrapper
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator


def classify_network_quality(latency_ms: float, packet_loss_pct: float) -> str:
    """
    Classify network performance into Low/Medium/High quality bands based on 6G URLLC requirements.
    
    This classification is designed for smart manufacturing environments where network
    performance directly impacts real-time control systems, monitoring infrastructure,
    and coordination tasks. The thresholds are derived from 6G Ultra-Reliable Low-Latency
    Communication (URLLC) specifications and manufacturing tolerance requirements.
    
    Threshold Rationale (6G URLLC Manufacturing Requirements):
    
    **High Quality (latency < 20ms AND packet_loss < 1%)**:
        - Ultra-low latency required for real-time manufacturing control systems
        - Example applications: robotic arm coordination, emergency stop systems,
          precision assembly line synchronization
        - 6G URLLC target: <1ms for critical control, <10ms for time-sensitive operations
        - Manufacturing tolerance: <20ms for reliable real-time responsiveness
        - <1% packet loss ensures virtually no data loss in control signals
    
    **Medium Quality (latency < 50ms AND packet_loss < 5%)**:
        - Acceptable for most monitoring, coordination, and supervisory tasks
        - Example applications: quality inspection systems, production monitoring dashboards,
          machine status reporting, operator coordination
        - Latency <50ms provides responsive feedback for human-in-the-loop operations
        - <5% packet loss allows some tolerance for non-critical telemetry data
    
    **Low Quality (latency >= 50ms OR packet_loss >= 5%)**:
        - Degraded performance may impact time-sensitive manufacturing operations
        - Example scenarios: network congestion, infrastructure issues, poor signal quality
        - Latency >=50ms introduces noticeable delays in control loops and feedback systems
        - Packet loss >=5% causes significant data gaps affecting decision-making accuracy
        - May require network optimization or infrastructure upgrades
    
    Classification Logic:
        The classification uses AND logic for High and Medium quality bands to ensure
        BOTH latency and packet loss meet requirements. A single poor metric (either
        high latency OR high packet loss) results in Low quality classification.
    
    Args:
        latency_ms: Network latency in milliseconds. Expected range: [0, 200+]
                   Typical values: 5-100ms depending on network conditions
        packet_loss_pct: Packet loss percentage in range [0, 100]
                        Typical values: 0-10% depending on network quality
    
    Returns:
        str: One of {'High', 'Medium', 'Low'} representing network quality band
    
    Examples:
        >>> classify_network_quality(15.0, 0.5)
        'High'
        >>> classify_network_quality(35.0, 3.0)
        'Medium'
        >>> classify_network_quality(60.0, 2.0)
        'Low'
        >>> classify_network_quality(25.0, 6.0)
        'Low'
    
    Notes:
        - This function is deterministic and stateless
        - Classification is strict: boundary values (exactly 20ms, exactly 50ms) are
          excluded from higher quality bands
        - For example: latency=20.0ms is NOT High quality (requires <20ms)
    
    Requirements Addressed:
        - 2.1: Compute Network_Quality_Band for each row
        - 2.2: Use defined thresholds for Low/Medium/High categories
        - 2.3: Document exact threshold values
        - 2.5: Expose classification logic as reusable function
    """
    if latency_ms < 20 and packet_loss_pct < 1.0:
        return 'High'
    elif latency_ms < 50 and packet_loss_pct < 5.0:
        return 'Medium'
    else:
        return 'Low'


class DataValidationError(Exception):
    """
    Custom exception for data validation failures.
    
    Raised when the dataset fails schema validation, value range validation,
    or categorical value validation.
    
    Examples:
        >>> raise DataValidationError("Row count mismatch: expected 100,000, got 95,000")
        >>> raise DataValidationError("Missing required column: Network_Latency_ms")
    """
    pass


@_cache_data
def load_and_prepare_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the Thales manufacturing telemetry dataset.
    
    This function performs comprehensive data loading with explicit date parsing
    and validation. It ensures the dataset meets all schema, type, and value
    requirements before being used for analysis.
    
    Date Parsing Strategy:
        The Timestamp column is parsed using DD-MM-YYYY format (European date format)
        with explicit format specification to prevent day/month ambiguity. For example,
        "15-03-2025" is correctly interpreted as March 15, 2025, not as an invalid
        date with month 15.
        
        Method: pd.to_datetime(format='%d-%m-%Y', dayfirst=True)
    
    Validation Checks:
        1. Row count validation: exactly 100,000 rows
        2. Column count validation: exactly 14 columns
        3. Required columns presence check
        4. Numeric type and range validation for network metrics
        5. Categorical value validation for status fields
    
    Performance:
        - Uses Streamlit @st.cache_data decorator for caching
        - Target: complete loading within 5 seconds
        - Cached result persists across Streamlit reruns
    
    Args:
        csv_path: Path to the Thales_Group_Manufacturing.csv file.
                 Can be absolute or relative path.
    
    Returns:
        pd.DataFrame: Validated DataFrame with the following guaranteed properties:
            - Shape: (100000, 14 base columns + Network_Quality_Band)
            - Required columns present and correctly typed
            - Timestamp column parsed as datetime64[ns]
            - All numeric columns have valid ranges
            - All categorical columns have valid values
            - Network_Quality_Band column added (Low/Medium/High classification)
    
    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.
        DataValidationError: If any validation check fails. Error message includes
                            specific details about the validation failure.
        
    Examples:
        >>> df = load_and_prepare_dataset("data/Thales_Group_Manufacturing.csv")
        >>> print(df.shape)
        (100000, 15)
        >>> print(df['Timestamp'].dtype)
        datetime64[ns]
        >>> print(df['Network_Quality_Band'].unique())
        ['Low' 'Medium' 'High']
        
    Requirements Validated:
        - 1.1: Dataset loading from CSV
        - 1.2: DD-MM-YYYY date parsing
        - 1.3: 100,000 rows, 14 columns
        - 1.4: Required columns exist
        - 1.5: Descriptive error for missing columns
        - 1.6: Efficiency_Status ∈ {Low, Medium, High}
        - 1.7: Operation_Mode ∈ {Active, Idle, Maintenance}
        - 1.8: Network_Latency_ms >= 0
        - 1.9: Packet_Loss_% ∈ [0, 100]
        - 1.10: Loading completes within 5 seconds
        - 2.1: Compute Network_Quality_Band for each row
        - 2.4: Assign exactly one Network_Quality_Band value per row
    """
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file not found at path: {csv_path}. "
            f"Please ensure the Thales_Group_Manufacturing.csv file exists."
        )
    
    # Parse Timestamp column with explicit DD-MM-YYYY format
    try:
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'],
            format='%d-%m-%Y',
            dayfirst=True
        )
    except Exception as e:
        raise DataValidationError(
            f"Failed to parse Timestamp column with DD-MM-YYYY format. "
            f"Error: {str(e)}"
        )
    
    # Validation 1: Row count
    expected_rows = 100000
    actual_rows = len(df)
    if actual_rows != expected_rows:
        raise DataValidationError(
            f"Row count validation failed: expected {expected_rows:,} rows, "
            f"but got {actual_rows:,} rows."
        )
    
    # Validation 2: Column count
    expected_cols = 14
    actual_cols = len(df.columns)
    if actual_cols != expected_cols:
        raise DataValidationError(
            f"Column count validation failed: expected {expected_cols} columns, "
            f"but got {actual_cols} columns."
        )
    
    # Validation 3: Required columns presence
    required_columns = [
        'Network_Latency_ms',
        'Packet_Loss_%',
        'Efficiency_Status',
        'Production_Speed',
        'Error_Rate',
        'Quality_Control_Defect_Rate',
        'Operation_Mode',
        'Timestamp',
        'Machine_ID'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing_columns)}. "
            f"Dataset must contain all required columns: {', '.join(required_columns)}."
        )
    
    # Validation 4: Efficiency_Status categorical values
    valid_efficiency_statuses = {'Low', 'Medium', 'High'}
    actual_efficiency_statuses = set(df['Efficiency_Status'].unique())
    invalid_efficiency = actual_efficiency_statuses - valid_efficiency_statuses
    
    if invalid_efficiency:
        raise DataValidationError(
            f"Invalid Efficiency_Status value(s): {', '.join(invalid_efficiency)}. "
            f"Valid values are: {', '.join(sorted(valid_efficiency_statuses))}."
        )
    
    # Validation 5: Operation_Mode categorical values
    valid_operation_modes = {'Active', 'Idle', 'Maintenance'}
    actual_operation_modes = set(df['Operation_Mode'].unique())
    invalid_modes = actual_operation_modes - valid_operation_modes
    
    if invalid_modes:
        raise DataValidationError(
            f"Invalid Operation_Mode value(s): {', '.join(invalid_modes)}. "
            f"Valid values are: {', '.join(sorted(valid_operation_modes))}."
        )
    
    # Validation 6: Network_Latency_ms numeric and non-negative
    if not pd.api.types.is_numeric_dtype(df['Network_Latency_ms']):
        raise DataValidationError(
            f"Network_Latency_ms must be numeric, but has type: {df['Network_Latency_ms'].dtype}"
        )
    
    negative_latency_count = (df['Network_Latency_ms'] < 0).sum()
    if negative_latency_count > 0:
        raise DataValidationError(
            f"Network_Latency_ms must be non-negative, but found {negative_latency_count} "
            f"negative value(s). Min value: {df['Network_Latency_ms'].min()}"
        )
    
    # Validation 7: Packet_Loss_% numeric and in range [0, 100]
    if not pd.api.types.is_numeric_dtype(df['Packet_Loss_%']):
        raise DataValidationError(
            f"Packet_Loss_% must be numeric, but has type: {df['Packet_Loss_%'].dtype}"
        )
    
    out_of_range_loss = ((df['Packet_Loss_%'] < 0) | (df['Packet_Loss_%'] > 100)).sum()
    if out_of_range_loss > 0:
        min_loss = df['Packet_Loss_%'].min()
        max_loss = df['Packet_Loss_%'].max()
        raise DataValidationError(
            f"Packet_Loss_% must be in range [0, 100], but found {out_of_range_loss} "
            f"out-of-range value(s). Range found: [{min_loss}, {max_loss}]"
        )
    
    # Apply Network Quality Band classification
    df['Network_Quality_Band'] = df.apply(
        lambda row: classify_network_quality(
            row['Network_Latency_ms'],
            row['Packet_Loss_%']
        ),
        axis=1
    )
    
    return df
