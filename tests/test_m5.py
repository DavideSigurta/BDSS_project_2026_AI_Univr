"""M5-final grid: fixed size, budgets and selectors."""

from bdss import campaign, config


def test_m5_final_grid_is_frozen():
    grid = campaign.final_grid_specs()
    assert len(grid) == 54 * len(config.M5_FINAL_BUDGETS) * len(config.M5_FINAL_SELECTORS)
    assert {spec[2] for spec in grid} == set(config.M5_FINAL_BUDGETS)
    assert {spec[3] for spec in grid} == set(config.M5_FINAL_SELECTORS)
