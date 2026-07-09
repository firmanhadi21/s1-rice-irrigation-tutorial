#!/usr/bin/env python3
"""
Period Utilities for Sentinel-1 Time Series Processing

Handles conversion between dates and period numbers based on Sentinel-1's
12-day revisit cycle.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_period_lookup(csv_file):
    """
    Load period lookup table from CSV

    Args:
        csv_file: Path to period lookup CSV

    Returns:
        DataFrame with columns: period, start_date, end_date
    """
    try:
        df = pd.read_csv(csv_file)

        # Ensure date columns are datetime
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded period lookup with {len(df)} periods")
        return df

    except Exception as e:
        logger.error(f"Error loading period lookup: {e}")
        raise


def get_period_from_date(date, periods_df):
    """
    Get period number for a given date

    Args:
        date: datetime object or string
        periods_df: DataFrame from load_period_lookup

    Returns:
        Period number (int) or None if not found
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    # Try different matching strategies

    # Strategy 1: Exact date match
    if 'date' in periods_df.columns:
        match = periods_df[periods_df['date'] == date]
        if not match.empty:
            return int(match.iloc[0]['period'])

    # Strategy 2: Date range match
    if 'start_date' in periods_df.columns and 'end_date' in periods_df.columns:
        match = periods_df[
            (periods_df['start_date'] <= date) &
            (periods_df['end_date'] >= date)
        ]
        if not match.empty:
            return int(match.iloc[0]['period'])

    # Strategy 3: Find nearest date
    if 'date' in periods_df.columns:
        periods_df['date_diff'] = abs((periods_df['date'] - date).dt.days)
        nearest = periods_df.loc[periods_df['date_diff'].idxmin()]

        # Only accept if within reasonable range (6 days = half period)
        if nearest['date_diff'] <= 6:
            return int(nearest['period'])

    logger.warning(f"Could not find period for date {date}")
    return None


def create_period_lookup(year, start_date=None, n_periods=31, period_days=12):
    """
    Create period lookup table for a year

    Args:
        year: Year (e.g., 2024)
        start_date: Start date (default: Jan 1 of year)
        n_periods: Number of periods (default: 31 for ~365 days)
        period_days: Days per period (default: 12 for Sentinel-1)

    Returns:
        DataFrame with period lookup
    """
    if start_date is None:
        start_date = datetime(year, 1, 1)
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    periods = []

    for period_num in range(1, n_periods + 1):
        period_start = start_date + timedelta(days=(period_num - 1) * period_days)
        period_end = period_start + timedelta(days=period_days - 1)
        period_center = period_start + timedelta(days=period_days // 2)

        periods.append({
            'period': period_num,
            'date': period_center,
            'start_date': period_start,
            'end_date': period_end,
            'year': year
        })

    df = pd.DataFrame(periods)
    logger.info(f"Created period lookup for {year} with {n_periods} periods")

    return df


def save_period_lookup(df, csv_file):
    """
    Save period lookup to CSV

    Args:
        df: Period lookup DataFrame
        csv_file: Output CSV path
    """
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved period lookup to {csv_file}")


def get_period_date_range(period, periods_df):
    """
    Get date range for a period

    Args:
        period: Period number
        periods_df: Period lookup DataFrame

    Returns:
        Tuple of (start_date, end_date) or None
    """
    match = periods_df[periods_df['period'] == period]

    if match.empty:
        return None

    row = match.iloc[0]

    if 'start_date' in row and 'end_date' in row:
        return (row['start_date'], row['end_date'])
    elif 'date' in row:
        # Approximate range from center date
        center = row['date']
        start = center - timedelta(days=6)
        end = center + timedelta(days=6)
        return (start, end)

    return None


def validate_period_lookup(periods_df):
    """
    Validate period lookup table

    Args:
        periods_df: Period lookup DataFrame

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    if 'period' not in periods_df.columns:
        raise ValueError("Period lookup must have 'period' column")

    if 'date' not in periods_df.columns:
        if 'start_date' not in periods_df.columns or 'end_date' not in periods_df.columns:
            raise ValueError("Period lookup must have 'date' or 'start_date'/'end_date' columns")

    # Check period numbers are sequential
    periods = sorted(periods_df['period'].unique())
    expected = list(range(1, len(periods) + 1))

    if periods != expected:
        logger.warning(f"Period numbers are not sequential: {periods}")

    logger.info(f"Period lookup validation passed: {len(periods)} periods")
    return True


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create period lookup for 2024
    periods_2024 = create_period_lookup(year=2024, n_periods=31)

    print("\nFirst 5 periods:")
    print(periods_2024.head())

    print("\nLast 5 periods:")
    print(periods_2024.tail())

    # Test date lookup
    test_date = datetime(2024, 3, 15)
    period = get_period_from_date(test_date, periods_2024)
    print(f"\nDate {test_date.date()} is in period: {period}")

    # Save example
    save_period_lookup(periods_2024, 'period_lookup_2024_example.csv')
