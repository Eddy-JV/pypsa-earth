# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging
import os
import shutil
from pathlib import Path

import atlite
import country_converter as coco
import numpy as np
import pandas as pd
import geopandas as gpd
from dask.distributed import Client
import progressbar as pgb
import requests
from io import StringIO
from _helpers import BASE_DIR, configure_logging, create_logger
from build_renewable_profiles import check_cutout_match

cc = coco.CountryConverter()

logger = create_logger(__name__)

COPERNICUS_CRS = "EPSG:4326"


def download_BioenergyPotential():
    """
    Download the global bioenergy potential dataset.

    The Excel file is hosted on Zenodo and contains global bioenergy
    potential data for both scenarios (Buisiness as Usual (BAU) and Towards Sustainability Scenario (TSS)).

    Source:
    https://zenodo.org/records/15273606
    """

    file_path = Path(BASE_DIR).joinpath("data","BioenergyPotential_GlobalFile_Final.xlsx")
    sheet = "BioPot_Global"
    header_rows = [575, 576, 577]  # Excel rows 576–578

    if file_path.exists():
        bioenergy_df = pd.read_excel(file_path, sheet_name=sheet, header=header_rows,engine="openpyxl", index_col=0,)

    else:
        url = (
            "https://zenodo.org/records/15273606/files/"
            "BioenergyPotential_GlobalFile_Final.xlsx?download=1"
        )

        bioenergy_df = pd.read_excel(url, sheet_name=sheet, header=header_rows, engine="openpyxl", index_col=0,)

    return bioenergy_df


def prepare_BioenergyPotential(bioenergy_df):
    """
    Prepare the bioenergy potential dataset for further analysis.

    This function processes the raw bioenergy potential DataFrame by
    cleaning, renaming columns, and aggregating data as needed.
    """

    # 1) Clean MultiIndex header (fix merged cells + remove "Unnamed")
    cols = bioenergy_df.columns.to_frame(index=False)

    # turn "Unnamed: ..." into NaN so forward-fill works
    cols = cols.apply(lambda s: s.map(lambda x: np.nan if isinstance(x, str) and x.startswith("Unnamed") else x))

    # forward-fill merged header cells in each level
    cols = cols.ffill(axis=0)

    bioenergy_df.columns = pd.MultiIndex.from_frame(cols)

    # 2) Fix the YEAR level: coerce to numeric
    cols = bioenergy_df.columns.to_frame(index=False)

    year = pd.to_numeric(cols.iloc[:, 2], errors="coerce")

    # 3) Snap weird floats like 2050.1 -> 2050 (round to nearest int)
    year_int = year.round().astype("Int64")   # keeps <NA> where conversion failed

    cols.iloc[:, 2] = year_int

    # 4) Keep only valid years (adjust as you like)
    valid_years = [2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
    mask = cols.iloc[:, 2].isin(valid_years)

    bioenergy_df = bioenergy_df.loc[:, mask.to_numpy()]
    bioenergy_df.columns = pd.MultiIndex.from_frame(cols.loc[mask].reset_index(drop=True))
    bioenergy_df = bioenergy_df.dropna(axis=1, how="all")
    bioenergy_df_T = bioenergy_df.T
    bioenergy_df_T.index.names = ["sources", "scenario", "year"]

    # keep only columns with a valid (non-NaN) column name
    bioenergy_df_T = bioenergy_df_T.loc[:, ~pd.isna(bioenergy_df_T.columns)]

    # convert country names to ISO2 codes
    new_cols = cc.convert(
        list(bioenergy_df_T.columns),
        to="ISO2",
        not_found=None
    )

    bioenergy_df_T.columns = new_cols

    # drop columns that couldn't be converted (None / NaN)
    bioenergy_df_T = bioenergy_df_T.loc[:, pd.notna(bioenergy_df_T.columns)]

    return bioenergy_df_T


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_biomass_potentials",
            planning_horizons="2050",)
    configure_logging(snakemake)

    pgb.streams.wrap_stderr()

    config = snakemake.config
    paths = snakemake.input
    nprocesses = int(snakemake.threads)
    noprogress = not snakemake.config["enable"]["progress_bar"]

    countries = snakemake.params.countries

    # crs
    area_crs = snakemake.params.crs["area_crs"]

    # Load regions shapefile
    regions = gpd.read_file(paths.regions)
    regions = regions.set_index("shape_id").rename_axis("bus")

    assert not regions.empty, (
        f"List of regions in {snakemake.input.regions} is empty, please "
        "check regions before continuing."
    )

    # Grid codes for bioenergy potentials in Copernicus dataset: https://land.copernicus.eu/en/technical-library/global-dynamic-land-cover-product-user-manual-v3.0/@@download/file
    grid_codes = [40, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126]

    # Load cutout
    if nprocesses > 1:
        client = Client(n_workers=nprocesses, threads_per_worker=1)
    else:
        client = None

    cutout = atlite.Cutout(paths["cutout"])
    check_cutout_match(cutout=cutout, geodf=regions)

    excluder = atlite.ExclusionContainer(crs=area_crs, res=100)

    if "copernicus" in config:
        excluder.add_raster(
                paths.copernicus,
                codes=grid_codes,
                invert=True,
                crs=COPERNICUS_CRS,
            )
        
    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)

    # Calcualate availability
    availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

    availability_bus_sum = availability.sum(dim=["y", "x"]).to_dataframe(name="bioenergy_availability")
    availability_bus_sum = (availability_bus_sum / availability_bus_sum.sum()).round(2)
    logger.info("Bioenergy availability per region:\n%s", availability_bus_sum)

    # Save biomass availability to CSV
    availability_bus_sum.to_csv(snakemake.output.bioenergy_availability, sep=",", encoding="utf-8", header=True)


    # Download global bioenergy potential dataset
    bioenergy_df = download_BioenergyPotential().copy()

    bioenergy_df_T = prepare_BioenergyPotential(bioenergy_df)

    # Save prepared biomass potentials to CSV
    bioenergy_df_T.to_csv(snakemake.output.bioenergy_global_potentials, sep=",", encoding="utf-8", header="true")

    # Filter biomass and biogas potentials for selected countries
    biomass_potential_scenario = snakemake.params.get("biomass_potential_scenario", "BAU")
    scenario_label = f"{biomass_potential_scenario} [TWh]"
    year = int(snakemake.wildcards.planning_horizons)
    
    # countries_biomass_pot = bioenergy_df_T.loc[(["Crop residues", "Forest residues"], scenario_label, year), countries].sum()
    #-----
    bioenergy_df_T = bioenergy_df_T.T.groupby(level=0).sum().T
    countries_biomass_pot = (bioenergy_df_T.loc[(["Crop residues", "Forest residues"], scenario_label, year)].reindex(columns=countries).sum())
    #-------

    # countries_biogas_pot = bioenergy_df_T.loc[(["Biogas"], scenario_label, year), countries].sum()
    #-----
    countries_biogas_pot = (bioenergy_df_T.loc[(["Biogas"], scenario_label, year)].reindex(columns=countries).sum())
    #-----

    availability_bus_sum_biomass = availability_bus_sum.copy()
    availability_bus_sum_biomass.index = availability_bus_sum_biomass.index.astype(str) + "_AC solid biomass"

    availability_bus_sum_biogas = availability_bus_sum.copy()
    availability_bus_sum_biogas.index = availability_bus_sum_biogas.index.astype(str) + "_AC biogas"

    # Distribute over the regions based on availablity
    regions_biomass_pot = countries_biomass_pot.sum() * availability_bus_sum_biomass   # in TWh/yr
    regions_biogas_pot = countries_biogas_pot.sum() * availability_bus_sum_biogas   # in TWh/yr

    regions_biomass_pot = regions_biomass_pot.rename(columns={"bioenergy_availability": "biomass_potential_TWh_per_year"})
    regions_biogas_pot = regions_biogas_pot.rename(columns={"bioenergy_availability": "biogas_potential_TWh_per_year"})

    # Save filtered biomass potentials to CSV
    regions_biomass_pot.to_csv(snakemake.output.biomass_potentials, sep=",", encoding="utf-8", header="true")
    regions_biogas_pot.to_csv(snakemake.output.biogas_potentials, sep=",", encoding="utf-8", header="true")


