"""
This file contains the DateTimeTool class, which provides current date and time
information for different time zones.
"""

from datetime import datetime
import pytz
from langchain.tools import Tool

class DateTimeTool:
    def get_current_datetime(self, timezone: str = "UTC") -> str:
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return f"Current date and time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Unknown timezone: {timezone}. Please provide a valid timezone."

    def get_tool(self) -> Tool:
        return Tool(
            name="Date and Time",
            func=self.get_current_datetime,
            description="Useful for getting current date and time in different time zones. Input should be a timezone string (e.g., 'UTC', 'US/Pacific', 'Europe/London')."
        )