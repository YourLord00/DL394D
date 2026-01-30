from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        return self.data.min(dim=1).values, self.data.max(dim=1).values

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        daily_avg = self.data.mean(dim=1)
        day_diff_over_days = torch.diff(daily_avg)
        
        return torch.min(day_diff_over_days)

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        daily_avg = self.data.mean(dim=1, keepdim=True)
        diffs = torch.abs(self.data - daily_avg)
        extreme_values, idxs = diffs.max(dim=1)

        return self.data[torch.arange(self.data.shape[0]), idxs]

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """

        return self.data[-k:].max(dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """

        return self.data[-k:].mean(dim=1).mean()

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """

        diffs = torch.abs(self.data -t)
        distances = diffs.sum(dim=1)

        return torch.argmin(distances)


if __name__ == "__main__":
    # Debug code - run with: python -m homework.weather_forecast

    data = [
        [74.8, 88.4, 54.4, 56.6, 65.3, 81.7, 74.5, 94.8, 72.7, 81.6],  # Day 0
        [67.4, 70.0, 51.1, 58.4, 64.6, 75.9, 84.8, 90.0, 58.0, 64.1],  # Day 1
    ]

    wf = WeatherForecast(data)

    print("=" * 50)
    print("DATA STRUCTURE")
    print("=" * 50)
    print(f"self.data:\n{wf.data}\n")
    print(f"Shape: {wf.data.shape}")
    print(f"  -> {wf.data.shape[0]} days, {wf.data.shape[1]} measurements per day\n")

    print("=" * 50)
    print("MIN/MAX WITH dim=1")
    print("=" * 50)
    min_result = wf.data.min(dim=1)
    max_result = wf.data.max(dim=1)

    print(f"min(dim=1) full result:\n{min_result}\n")
    print(f"min(dim=1).values:  {min_result.values}")
    print(f"min(dim=1).indices: {min_result.indices}  <- column where min found\n")
    print(f"max(dim=1).values:  {max_result.values}")
    print(f"max(dim=1).indices: {max_result.indices}\n")

    print("=" * 50)
    print("MEAN PER DAY & DIFF")
    print("=" * 50)
    daily_avg = wf.data.mean(dim=1)
    print(f"Daily averages: {daily_avg}")
    print(f"Diff between days: {torch.diff(daily_avg)}")

    # Test the actual functions (breakpoints inside methods will trigger here)
    print("=" * 50)
    print("TESTING find_min_and_max_per_day()")
    print("=" * 50)
    min_temps, max_temps = wf.find_min_and_max_per_day()
    print(f"Min temps: {min_temps}")
    print(f"Max temps: {max_temps}")

    print("=" * 50)
    print("TESTING find_the_most_extreme_day()")
    print("=" * 50)
    extreme = wf.find_the_most_extreme_day()
    print(f"Extreme measurements: {extreme}")
    print(f"Expected: [94.8, 90.0] (the actual temps, not the differences)")
