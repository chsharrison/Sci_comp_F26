"""Surface-weather animation of Ida using local ERA5 and IBTrACS files.

From the workspace root, run:
    final_project/.venv-data/bin/python week7/ida_weather_movie.py

The local .venv-data environment contains imageio-ffmpeg for video export.
Use --preview-only to check a single frame before making the movie.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedLocator

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "final_project"
RAW = PROJECT / "data" / "raw"
cartopy.config["data_dir"] = PROJECT / "data" / "cartopy"
GEO = ccrs.PlateCarree()
PROJECTION = ccrs.LambertConformal(
    central_longitude=-85, central_latitude=28, standard_parallels=(20, 40)
)
EXTENT = [-98, -70, 14, 43]
WIND_NORM = Normalize(0, 35)
WIND_CMAP = LinearSegmentedColormap.from_list("surface_wind", [
    "#f8fcff", "#d7eaf5", "#83c5dc", "#46aa9a", "#83bf68",
    "#d5da66", "#f7c34a", "#ef8848", "#cc3e3b",
])


def load_ida():
    """Select real six-hourly best-track records, using the lab's time window."""
    with xr.open_dataset(RAW / "IBTrACS.NA.v04r01.nc") as source:
        names = np.char.strip(source["name"].values.astype(str))
        indices = np.flatnonzero((source["season"].values == 2021) & (names == "IDA"))
        if len(indices) != 1:
            raise ValueError("Expected exactly one 2021 Ida record.")
        storm = source.isel(storm=int(indices[0]))
        ida = pd.DataFrame({
            "time": pd.to_datetime(storm["time"].values, utc=True).round("s"),
            "lat": storm["lat"].values, "lon": storm["lon"].values,
            "wind_kt": storm["usa_wind"].values,
        })
    ida = ida[(ida.time >= "2021-08-26") & (ida.time < "2021-09-02")
              & ida.time.dt.hour.isin([0, 6, 12, 18])
              & (ida.time.dt.minute == 0)].copy()
    ida.loc[ida.wind_kt < 0, "wind_kt"] = np.nan
    return ida.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)


def load_weather(ida):
    """Read only the needed region and align ERA5 exactly to IBTrACS times."""
    times = ida.time.dt.tz_localize(None).to_numpy()
    parts = []
    for month in ["202108", "202109"]:
        with xr.open_dataset(RAW / f"era5_surface_{month}.nc") as source:
            part = source[["u10", "v10", "msl"]].sel(
                valid_time=slice(times[0], times[-1]),
                latitude=slice(51, 7), longitude=slice(-105, -62),
            ).reset_coords(drop=True).load()
            parts.append(part)
    weather = xr.concat(parts, dim="valid_time").sortby("valid_time")
    weather = weather.sel(valid_time=times)  # Exact match; no time interpolation.
    if not np.array_equal(weather.valid_time.values, times):
        raise ValueError("ERA5 and IBTrACS timestamps do not match.")
    for variable in ["u10", "v10", "msl"]:
        if not np.isfinite(weather[variable].values).all():
            raise ValueError(f"Non-finite weather values in {variable}.")
    weather["speed"] = np.hypot(weather.u10, weather.v10)
    weather["speed"].attrs = {"units": "m s-1", "long_name": "10 m wind speed"}
    weather["pressure"] = weather.msl / 100  # Pa to hPa.
    weather["pressure"].attrs = {"units": "hPa", "long_name": "Mean sea-level pressure"}
    return weather


def make_figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    axes = [fig.add_axes(box, projection=PROJECTION) for box in [
        [0.048, 0.225, 0.43, 0.63], [0.532, 0.225, 0.43, 0.63]
    ]]
    fig.suptitle("Hurricane Ida (2021) — surface weather and track", y=0.975, fontsize=21)
    clock = fig.text(0.5, 0.919, "", ha="center", fontsize=12)
    cax = fig.add_axes([0.19, 0.127, 0.47, 0.018])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=WIND_NORM, cmap=WIND_CMAP),
                       cax=cax, orientation="horizontal", ticks=np.arange(0, 36, 5))
    bar.set_label("ERA5 10 m wind speed (m s$^{-1}$)", labelpad=4)
    fig.text(0.5, 0.053,
             "Contours: sea-level pressure (hPa), every 4 hPa   |   Solid line / circle: IBTrACS track / center",
             ha="center", color="0.25", fontsize=10)
    fig.text(0.5, 0.025,
             "ERA5: 0.25° grid, 6-hourly snapshots   •   NOAA IBTrACS: best track   •   One second = six hours",
             ha="center", color="0.4", fontsize=9)
    return fig, axes, clock


def draw_frame(frame, fig, axes, clock, ida, weather):
    """Redraw the two maps for one actual six-hourly weather snapshot."""
    current = ida.iloc[frame]
    field = weather.isel(valid_time=frame)
    zoom = [current.lon - 6, current.lon + 6, current.lat - 5, current.lat + 5]
    trail = ida.iloc[:frame + 1]
    for panel, (ax, extent, stride) in enumerate(zip(axes, [EXTENT, zoom], [8, 3])):
        ax.clear()
        ax.set_extent(extent, crs=GEO)
        # Plot only the nearby grid, including a small margin for map corners.
        crop = field if panel == 0 else field.sel(
            latitude=slice(extent[3] + 2, extent[2] - 2),
            longitude=slice(extent[0] - 2, extent[1] + 2))
        lon, lat = np.meshgrid(crop.longitude.values, crop.latitude.values)
        ax.contourf(lon, lat, crop.speed.values, levels=np.arange(0, 36, 1),
                    cmap=WIND_CMAP, norm=WIND_NORM, transform=GEO,
                    transform_first=True, extend="max", zorder=1)
        contours = ax.contour(lon, lat, crop.pressure.values,
                              levels=np.arange(960, 1045, 4), colors="0.3",
                              linewidths=0.65, transform=GEO, zorder=3)
        label_levels = [1008, 1016, 1024, 1032] if panel == 0 else contours.levels[::2]
        ax.clabel(contours, label_levels, inline=True, fontsize=8, fmt="%.0f")
        sampled = crop.isel(latitude=slice(None, None, stride),
                            longitude=slice(None, None, stride))
        x, y = np.meshgrid(sampled.longitude.values, sampled.latitude.values)
        u = np.ma.masked_where(sampled.speed.values < 1.5, sampled.u10.values)
        v = np.ma.masked_where(sampled.speed.values < 1.5, sampled.v10.values)
        # u/v are geographic east/north components. Cartopy rotates the arrows.
        arrows = ax.quiver(x, y, u, v, transform=GEO, scale_units="inches", scale=80,
                           width=0.0027, color="0.14", alpha=0.83, zorder=4,
                           headwidth=3.2, headlength=4.5, pivot="middle")
        if panel == 1:
            ax.quiverkey(arrows, 0.76, 0.132, 20, "20 m s$^{-1}$",
                         coordinates="figure", labelpos="E", fontproperties={"size": 10})
        ax.coastlines("50m", linewidth=0.8, color="0.15", zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45,
                       edgecolor="0.35", zorder=5)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35,
                       edgecolor="0.45", zorder=5)
        grid = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, rotate_labels=False,
                            linewidth=0.4, color="0.5",
                            alpha=0.45, linestyle=":", zorder=2)
        grid.top_labels = grid.right_labels = False
        grid.xlocator = FixedLocator(np.arange(-110, -59, 5 if panel == 0 else 2))
        grid.ylocator = FixedLocator(np.arange(0, 61, 5 if panel == 0 else 2))
        grid.xlabel_style = grid.ylabel_style = {"size": 9, "color": "0.25"}
        line, = ax.plot(trail.lon, trail.lat, color="0.1", linewidth=1.6,
                         transform=GEO, zorder=7)
        line.set_path_effects([pe.Stroke(linewidth=3.3, foreground="white"), pe.Normal()])
        ax.plot(current.lon, current.lat, "o", markersize=6.5,
                markerfacecolor="white", markeredgecolor="0.1", markeredgewidth=1.3,
                transform=GEO, zorder=8)
        ax.set_title(["Regional view", "Storm-following view"][panel], fontsize=13, pad=13)
    axes[0].plot([zoom[0], zoom[1], zoom[1], zoom[0], zoom[0]],
                  [zoom[2], zoom[2], zoom[3], zoom[3], zoom[2]],
                  transform=GEO, color="0.15", linewidth=0.9, linestyle="--", zorder=6)
    # Label the actual ERA5 local minimum near the reference center.
    local = field.pressure.sel(latitude=slice(current.lat + 2, current.lat - 2),
                               longitude=slice(current.lon - 2, current.lon + 2))
    iy, ix = np.unravel_index(np.argmin(local.values), local.shape)
    low = float(local.values[iy, ix])
    low_lon, low_lat = float(local.longitude[ix]), float(local.latitude[iy])
    label = axes[1].annotate(f"L\n{low:.0f}", (low_lon, low_lat), xycoords=GEO._as_mpl_transform(axes[1]),
                             xytext=(10, 9), textcoords="offset points", fontsize=11,
                             color="#922c30", fontweight="bold", ha="center", zorder=9)
    label.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
    clock.set_text(f"{current.time:%d %b %Y  %H:%M UTC}    |    IBTrACS maximum sustained wind: {current.wind_kt:.0f} kt")
    return axes


def make_animation(ida, weather):
    fig, axes, clock = make_figure()
    animation = FuncAnimation(fig, draw_frame, frames=len(ida), interval=1000,
                              fargs=(fig, axes, clock, ida, weather), blit=False,
                              repeat=True, cache_frame_data=False)
    return fig, animation


def save_movie(animation, output_dir=HERE):
    """Keep each real snapshot on screen for one second; export GIF and MP4."""
    output_dir = Path(output_dir)
    gif_path, mp4_path = output_dir / "Ida_weather.gif", output_dir / "Ida_weather.mp4"
    # The local environment provides FFmpeg even when the notebook uses python3.
    ffmpeg = subprocess.check_output([
        str(PROJECT / ".venv-data" / "bin" / "python"), "-c",
        "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
    ], text=True).strip()
    # A normal 24 fps video repeats the same snapshots; it does not invent data.
    with plt.rc_context({"animation.ffmpeg_path": ffmpeg}):
        animation.save(mp4_path, writer=FFMpegWriter(
            fps=1, codec="libx264", extra_args=["-r", "24", "-crf", "18",
                                                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        ), dpi=100)
    animation.save(gif_path, writer=PillowWriter(fps=1), dpi=100)
    return gif_path, mp4_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    ida = load_ida()
    weather = load_weather(ida)
    peak = int(ida.wind_kt.idxmax())
    fig, axes, clock = make_figure()
    draw_frame(peak, fig, axes, clock, ida, weather)
    fig.savefig(HERE / "Ida_weather_preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Preview saved; {len(ida)} exact 6-hourly frames.", flush=True)
    if not args.preview_only:
        fig, animation = make_animation(ida, weather)
        print("Rendering weather movie...", flush=True)
        for path in save_movie(animation):
            print(f"Saved {path.name}: {path.stat().st_size:,} bytes", flush=True)
        plt.close(fig)
        metadata = {
            "frames": len(ida), "data_interval_hours": 6,
            "display_seconds_per_snapshot": 1, "interpolated_weather": False,
            "start_utc": str(ida.time.iloc[0]), "end_utc": str(ida.time.iloc[-1]),
            "variables": {"shading": "ERA5 10 m wind speed (m/s)",
                          "arrows": "ERA5 u10/v10; reference 20 m/s",
                          "contours": "ERA5 MSLP (hPa), interval 4 hPa",
                          "track": "NOAA IBTrACS, usa_wind in knots"},
            "low_label": "ERA5 minimum MSLP within +/-2 degrees of the IBTrACS center",
            "note": "ERA5 gridded wind is not the IBTrACS maximum sustained wind.",
        }
        (HERE / "Ida_weather_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
