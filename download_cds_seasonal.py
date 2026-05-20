"""Download seasonal forecast variables from the Copernicus Climate Data Store.

Before running this script, configure CDS API credentials according to the
Copernicus Climate Data Store instructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originating-centre", default="ecmwf")
    parser.add_argument("--system", default="5")
    parser.add_argument("--variables", nargs="+", default=["2m_temperature", "total_precipitation", "evaporation"])
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--day", default="1")
    parser.add_argument("--leadtime-hour", type=int, default=5160)
    parser.add_argument("--time-step", type=int, default=24)
    parser.add_argument("--area", nargs=4, type=float, metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--output-dir", default="data/raw/cds")
    parser.add_argument("--catchment", default="A")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.originating_centre.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_times = list(range(args.time_step, args.leadtime_hour + args.time_step, args.time_step))
    client = cdsapi.Client()

    for year in range(args.start_year, args.end_year + 1):
        for month in range(args.start_month, args.end_month + 1):
            request = {
                "format": "netcdf",
                "originating_centre": args.originating_centre,
                "system": args.system,
                "variable": args.variables,
                "year": str(year),
                "month": str(month),
                "day": args.day,
                "leadtime_hour": lead_times,
            }
            if args.area:
                request["area"] = args.area

            output_file = output_dir / f"{args.catchment}_{year}_{month:02d}_{args.originating_centre}.nc"
            client.retrieve("seasonal-original-single-levels", request, str(output_file))


if __name__ == "__main__":
    main()
