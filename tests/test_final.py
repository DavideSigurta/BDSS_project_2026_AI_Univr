"""Final grid: fixed size, budgets and selectors."""

from bdss import campaign, config


def test_final_grid_is_frozen():
    grid = campaign.final_grid_specs()
    assert len(grid) == 54 * len(config.FINAL_BUDGETS) * len(config.FINAL_SELECTORS)
    assert {spec[2] for spec in grid} == set(config.FINAL_BUDGETS)
    assert {spec[3] for spec in grid} == set(config.FINAL_SELECTORS)
